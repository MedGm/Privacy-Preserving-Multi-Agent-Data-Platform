import asyncio
import os
import random

from common.local_trainer import LocalTrainer
from spade.agent import Agent
from spade.behaviour import FSMBehaviour, State
from spade.message import Message

from common.logger import setup_logger
from common.messages import (
    MSG_TYPE_JOIN,
    MSG_TYPE_ROUND_START,
    MSG_TYPE_MODEL_UPDATE,
    MSG_TYPE_GLOBAL_UPDATE,
    MessageHeader,
    validate_schema,
    build_message,
    parse_message,
)

logger = setup_logger("Agent")

# --- Define State Names Constants ---
STATE_INIT = "Init"
STATE_IDLE = "Idle"
STATE_READY = "Ready"
STATE_COMPUTING = "Computing"
STATE_REPORTING = "Reporting"
STATE_WAITING = "Waiting"
STATE_FAILED = "Failed"


class InitState(State):
    async def run(self):
        logger.info(f"[{STATE_INIT}] Initialization started.")
        # We no longer need a CSV data path. The data loader handles memory sharding.
        self.agent.data_path = None

        # Send JOIN to coordinator
        msg = Message(to=self.agent.coordinator_jid)
        header = MessageHeader(
            msg_id=f"join_{random.randint(1000, 9999)}",
            sender_id=self.agent.agent_id,
            round_id=0,
            msg_type=MSG_TYPE_JOIN,
            task=self.agent.configured_task,
        )
        msg.body = build_message(header, {})
        await self.send(msg)

        self.set_next_state(STATE_IDLE)


class IdleState(State):
    async def run(self):
        logger.debug(f"[{STATE_IDLE}] Waiting for coordinator signals...")
        msg = await self.receive(timeout=10)

        if msg:
            try:
                header, payload = parse_message(msg.body)
                if header.msg_type == MSG_TYPE_ROUND_START:
                    if header.task != self.agent.configured_task:
                        logger.debug(
                            f"[{STATE_IDLE}] Ignoring ROUND_START for task {header.task}"
                        )
                        self.set_next_state(STATE_IDLE)
                        return

                    logger.info(
                        f"[{STATE_IDLE}] Received ROUND_START for round {header.round_id} (Task: {header.task})"
                    )
                    self.agent.current_round = header.round_id
                    self.agent.current_round_config = payload
                    self.set_next_state(STATE_READY)
                else:
                    self.set_next_state(STATE_IDLE)
            except Exception as e:
                logger.error(f"[{STATE_IDLE}] Error parsing message: {e}")
                self.set_next_state(STATE_IDLE)
        else:
            self.set_next_state(STATE_IDLE)


class ReadyState(State):
    async def run(self):
        logger.info(
            f"[{STATE_READY}] Validating local resources for round {self.agent.current_round}"
        )
        self.set_next_state(STATE_COMPUTING)


class ComputingState(State):
    async def run(self):
        logger.info(f"[{STATE_COMPUTING}] Executing local computation... (Isolated)")

        # Train on isolated data shard
        global_weights = self.agent.global_model if self.agent.global_model else None
        result = self.agent.trainer.train(
            self.agent.agent_id, global_weights=global_weights
        )

        self.agent.local_output = result
        logger.info(
            f"[{STATE_COMPUTING}] Local metrics: "
            f"acc={result['metrics']['accuracy']:.4f}, "
            f"loss={result['metrics']['loss']:.4f}"
        )
        self.set_next_state(STATE_REPORTING)


class ReportingState(State):
    async def run(self):
        logger.info(f"[{STATE_REPORTING}] Validating and emitting local update...")
        output = self.agent.local_output

        # Privacy Firewall Check
        if not validate_schema(output, MSG_TYPE_MODEL_UPDATE):
            logger.critical(
                f"[{STATE_REPORTING}] Privacy violation detected. Data structure invalid. Halting."
            )
            self.set_next_state(STATE_FAILED)
            return

        msg = Message(to=self.agent.coordinator_jid)
        header = MessageHeader(
            msg_id=f"upd_{self.agent.current_round}_{random.randint(1000, 9999)}",
            sender_id=self.agent.agent_id,
            round_id=self.agent.current_round,
            msg_type=MSG_TYPE_MODEL_UPDATE,
            task=self.agent.configured_task,
        )
        msg.body = build_message(header, output)
        await self.send(msg)
        self.set_next_state(STATE_WAITING)


class WaitingState(State):
    async def run(self):
        logger.debug(f"[{STATE_WAITING}] Waiting for global update...")
        msg = await self.receive(timeout=20)
        if msg:
            try:
                header, payload = parse_message(msg.body)
                if header.msg_type == MSG_TYPE_GLOBAL_UPDATE:
                    if header.task != self.agent.configured_task:
                        logger.debug(
                            f"[{STATE_WAITING}] Ignoring GLOBAL_UPDATE for task {header.task}"
                        )
                        self.set_next_state(STATE_WAITING)
                        return

                    logger.info(
                        f"[{STATE_WAITING}] Received Global Update "
                        f"(dim={len(payload.get('weights', []))}) for task {header.task}"
                    )
                    self.agent.global_model = {
                        "weights": payload.get("weights", []),
                        "intercept": payload.get("intercept", [0.0]),
                    }
                    self.set_next_state(STATE_IDLE)
                else:
                    self.set_next_state(STATE_WAITING)
            except Exception as e:
                logger.error(f"[{STATE_WAITING}] Parse error: {e}")
                self.set_next_state(STATE_WAITING)
        else:
            # Could transition to Failed if strict timeout needed
            self.set_next_state(STATE_IDLE)


class FailedState(State):
    async def run(self):
        logger.error(f"[{STATE_FAILED}] Agent encountered fatal error. Halting.")
        await self.agent.stop()


class ClientAgent(Agent):
    def __init__(self, jid, password, agent_id, coordinator_jid):
        super().__init__(jid, password)
        self.agent_id = agent_id
        self.coordinator_jid = coordinator_jid
        self.current_round = 0
        self.global_model = None
        self.configured_task = os.environ.get("AGENT_TASK", "breast_cancer")

        if os.environ.get("USE_PYTORCH", "false").lower() == "true":
            from agent.pytorch_trainer import PyTorchTrainer

            self.trainer = PyTorchTrainer()
            logger.info("Initialized with PyTorchTrainer")
        elif os.environ.get("USE_SPARK", "false").lower() == "true":
            from common.spark_trainer import SparkTrainer

            self.trainer = SparkTrainer()
            logger.info("Initialized with PySpark SparkTrainer")
        else:
            self.trainer = LocalTrainer()
            logger.info("Initialized with scikit-learn LocalTrainer")

    async def setup(self):
        logger.info(f"Agent {self.agent_id} starting up...")
        fsm = FSMBehaviour()

        fsm.add_state(name=STATE_INIT, state=InitState(), initial=True)
        fsm.add_state(name=STATE_IDLE, state=IdleState())
        fsm.add_state(name=STATE_READY, state=ReadyState())
        fsm.add_state(name=STATE_COMPUTING, state=ComputingState())
        fsm.add_state(name=STATE_REPORTING, state=ReportingState())
        fsm.add_state(name=STATE_WAITING, state=WaitingState())
        fsm.add_state(name=STATE_FAILED, state=FailedState())

        # Transitions
        fsm.add_transition(source=STATE_INIT, dest=STATE_IDLE)
        fsm.add_transition(source=STATE_IDLE, dest=STATE_READY)
        fsm.add_transition(source=STATE_IDLE, dest=STATE_IDLE)
        fsm.add_transition(source=STATE_READY, dest=STATE_COMPUTING)
        fsm.add_transition(source=STATE_READY, dest=STATE_FAILED)
        fsm.add_transition(source=STATE_COMPUTING, dest=STATE_REPORTING)
        fsm.add_transition(source=STATE_REPORTING, dest=STATE_WAITING)
        fsm.add_transition(source=STATE_REPORTING, dest=STATE_FAILED)
        fsm.add_transition(source=STATE_WAITING, dest=STATE_IDLE)
        fsm.add_transition(source=STATE_WAITING, dest=STATE_WAITING)

        self.add_behaviour(fsm)


async def main():
    import os

    agent_id = os.environ.get("AGENT_ID", "agent1")
    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")
    jid = f"{agent_id}@{xmpp_server}"
    password = "password"
    coordinator_jid = f"coordinator@{xmpp_server}"

    logger.info(f"Connecting to Breast Cancer Federation as {agent_id}")

    agent = ClientAgent(jid, password, agent_id, coordinator_jid)
    agent.verify_security = False
    await agent.start(auto_register=True)

    try:
        while agent.is_alive():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
