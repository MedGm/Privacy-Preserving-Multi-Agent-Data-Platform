import asyncio
import os
import random
import time
from spade.agent import Agent
from spade.behaviour import FSMBehaviour, State
from spade.message import Message

from common.logger import setup_logger
from common.federation_store import store
from common.messages import (
    MSG_TYPE_JOIN,
    MSG_TYPE_ROUND_START,
    MSG_TYPE_MODEL_UPDATE,
    MSG_TYPE_GLOBAL_UPDATE,
    MessageHeader,
    build_message,
    parse_message,
)

logger = setup_logger("Coordinator")

STATE_INIT = "Init"
STATE_IDLE = "Idle"
STATE_REGISTRATION = "Registration"
STATE_ROUND_SETUP = "RoundSetup"
STATE_AGGREGATING = "Aggregating"
STATE_BROADCASTING = "Broadcasting"
STATE_TERMINATED = "Terminated"

MIN_AGENTS = int(os.environ.get("MIN_AGENTS", 2))
# MAX_ROUNDS is now dynamic per run, but we keep a default fallback
DEFAULT_MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", 5))
ROUND_TIMEOUT = 60.0


class InitState(State):
    async def run(self):
        logger.info(f"[{STATE_INIT}] Coordinator initialized.")
        self.agent.registry = set()
        self.set_next_state(STATE_IDLE)


class IdleState(State):
    async def run(self):
        store.update_status("Idle - Waiting for Start")

        start_requested, target_rounds = store.pop_start_request()
        if start_requested:
            logger.info(
                f"[{STATE_IDLE}] Start requested. Target rounds: {target_rounds}"
            )
            self.agent.current_round = 0
            self.agent.max_rounds = (
                target_rounds if target_rounds > 0 else DEFAULT_MAX_ROUNDS
            )
            self.agent.global_model = {"weights": None, "intercept": None}
            store.reset_metrics()

            from common.mlflow_tracker import FederationTracker

            self.agent.tracker = FederationTracker()
            self.agent.target_rounds = self.agent.max_rounds # Use max_rounds as target_rounds

            # Log setup config and tags to MLflow
            self.agent.tracker.log_setup(
                params={
                    "target_rounds": self.agent.target_rounds,
                    "min_agents": os.environ.get("MIN_AGENTS", "5"),
                    "dp_epsilon": os.environ.get("DP_EPSILON", "1.0"),
                    "dp_clip_norm": os.environ.get("DP_CLIP_NORM", "1.0"),
                }
            )

            self.set_next_state(STATE_REGISTRATION)
        else:
            await asyncio.sleep(1)
            self.set_next_state(STATE_IDLE)


class RegistrationState(State):
    async def run(self):
        logger.info(
            f"[{STATE_REGISTRATION}] Waiting for {MIN_AGENTS} agents to join..."
        )
        msg = await self.receive(timeout=10)

        if msg:
            try:
                header, payload = parse_message(msg.body)
                if header.msg_type == MSG_TYPE_JOIN:
                    logger.info(
                        f"[{STATE_REGISTRATION}] Agent {header.sender_id} joined."
                    )
                    self.agent.registry.add(str(msg.sender))
                    store.set_agents(list(self.agent.registry))
            except Exception as e:
                logger.error(f"[{STATE_REGISTRATION}] Error parsing message: {e}")

        if store.pop_stop_request():
            logger.info(f"[{STATE_REGISTRATION}] Run aborted during registration.")
            self.set_next_state(STATE_IDLE)
            return

        if len(self.agent.registry) >= MIN_AGENTS:
            self.set_next_state(STATE_ROUND_SETUP)
        else:
            self.set_next_state(STATE_REGISTRATION)


class RoundSetupState(State):
    async def run(self):
        if store.pop_stop_request():
            logger.info(f"[{STATE_ROUND_SETUP}] Stop requested. Aborting run.")
            self.set_next_state(STATE_IDLE)
            return

        self.agent.current_round += 1
        store.update_status("RoundSetup")
        store.update_round(self.agent.current_round, self.agent.max_rounds)
        logger.info(f"[{STATE_ROUND_SETUP}] Starting Round {self.agent.current_round}")

        # Broadcast round start
        for agent_jid in self.agent.registry:
            msg = Message(to=agent_jid)
            header = MessageHeader(
                msg_id=f"start_{self.agent.current_round}_{random.randint(1000, 9999)}",
                sender_id="coordinator",
                round_id=self.agent.current_round,
                msg_type=MSG_TYPE_ROUND_START,
            )
            msg.body = build_message(
                header, {"task": "dummy_regression", "hyperparams": {}}
            )
            await self.send(msg)

        self.set_next_state(STATE_AGGREGATING)


class AggregatingState(State):
    async def run(self):
        logger.info(f"[{STATE_AGGREGATING}] Waiting for model updates...")
        start_time = time.time()
        received_updates = []

        while time.time() - start_time < ROUND_TIMEOUT:
            msg = await self.receive(timeout=1)
            if msg:
                try:
                    header, payload = parse_message(msg.body)
                    if (
                        header.msg_type == MSG_TYPE_MODEL_UPDATE
                        and header.round_id == self.agent.current_round
                    ):
                        logger.info(
                            f"[{STATE_AGGREGATING}] Received update from {header.sender_id} "
                            f"(n={payload.get('num_samples', '?')})"
                        )
                        payload["sender_id"] = header.sender_id
                        received_updates.append(payload)
                except Exception as e:
                    logger.error(f"[{STATE_AGGREGATING}] Error parsing update: {e}")

            if len(received_updates) == len(self.agent.registry):
                break

        if len(received_updates) > 0:
            # Weighted FedAvg: global_w = sum(n_i * w_i) / sum(n_i)
            total_samples = sum(u.get("num_samples", 1) for u in received_updates)
            num_features = len(received_updates[0]["weights"])
            num_intercept = len(received_updates[0].get("intercept", [0.0]))

            avg_weights = [0.0] * num_features
            avg_intercept = [0.0] * num_intercept

            for update in received_updates:
                n = update.get("num_samples", 1)
                weight = n / total_samples
                for j in range(num_features):
                    avg_weights[j] += weight * update["weights"][j]
                for j in range(num_intercept):
                    avg_intercept[j] += weight * update.get("intercept", [0.0])[j]

                # Store per-agent metrics
                store.add_agent_metrics(
                    update["sender_id"],
                    self.agent.current_round,
                    update.get("metrics", {}),
                    n
                )

            self.agent.global_model = {
                "weights": avg_weights,
                "intercept": avg_intercept,
            }

            # Log convergence metrics
            avg_acc = sum(
                u.get("metrics", {}).get("accuracy", 0) for u in received_updates
            ) / len(received_updates)
            avg_loss = sum(
                u.get("metrics", {}).get("loss", 0) for u in received_updates
            ) / len(received_updates)
            avg_prec = sum(
                u.get("metrics", {}).get("precision", 0) for u in received_updates
            ) / len(received_updates)
            avg_rec = sum(
                u.get("metrics", {}).get("recall", 0) for u in received_updates
            ) / len(received_updates)

            logger.info(
                f"[{STATE_AGGREGATING}] FedAvg complete (n={total_samples}). "
                f"Avg acc={avg_acc:.4f}, loss={avg_loss:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}"
            )
            store.add_round_metrics(
                self.agent.current_round, avg_acc, avg_loss, avg_prec, avg_rec, total_samples
            )
            store.update_global_model(self.agent.global_model)

            if hasattr(self.agent, "tracker"):
                # Log individual and global metrics to MLflow
                avg_metrics = {
                    "accuracy": avg_acc,
                    "loss": avg_loss,
                    "precision": avg_prec,
                    "recall": avg_rec,
                }
                self.agent.tracker.log_round(self.agent.current_round, avg_metrics, len(received_updates), received_updates)

            self.set_next_state(STATE_BROADCASTING)
        else:
            logger.warning(
                f"[{STATE_AGGREGATING}] Round failed: zero participation. Retrying."
            )
            self.agent.current_round -= 1
            self.set_next_state(STATE_ROUND_SETUP)


class BroadcastingState(State):
    async def run(self):
        store.update_status("Broadcasting")
        logger.info(f"[{STATE_BROADCASTING}] Broadcasting global model...")
        for agent_jid in self.agent.registry:
            msg = Message(to=agent_jid)
            header = MessageHeader(
                msg_id=f"glob_{self.agent.current_round}_{random.randint(1000, 9999)}",
                sender_id="coordinator",
                round_id=self.agent.current_round,
                msg_type=MSG_TYPE_GLOBAL_UPDATE,
            )
            msg.body = build_message(
                header,
                {
                    "weights": self.agent.global_model["weights"],
                    "intercept": self.agent.global_model["intercept"],
                },
            )
            await self.send(msg)

        if self.agent.current_round >= getattr(
            self.agent, "max_rounds", DEFAULT_MAX_ROUNDS
        ):
            self.set_next_state(STATE_TERMINATED)
        else:
            await asyncio.sleep(2)  # Small buffer between rounds
            self.set_next_state(STATE_ROUND_SETUP)


class TerminatedState(State):
    async def run(self):
        store.update_status("Terminated")
        logger.info(f"[{STATE_TERMINATED}] Target rounds completed.")
        if hasattr(self.agent, "tracker"):
            self.agent.tracker.log_final_model(self.agent.global_model)
        logger.info(f"[{STATE_TERMINATED}] Run finished. Heading to Idle.")
        self.set_next_state(STATE_IDLE)


class CoordinatorAgent(Agent):
    async def setup(self):
        logger.info("Coordinator starting up...")
        fsm = FSMBehaviour()

        fsm.add_state(name=STATE_INIT, state=InitState(), initial=True)
        fsm.add_state(name=STATE_IDLE, state=IdleState())
        fsm.add_state(name=STATE_REGISTRATION, state=RegistrationState())
        fsm.add_state(name=STATE_ROUND_SETUP, state=RoundSetupState())
        fsm.add_state(name=STATE_AGGREGATING, state=AggregatingState())
        fsm.add_state(name=STATE_BROADCASTING, state=BroadcastingState())
        fsm.add_state(name=STATE_TERMINATED, state=TerminatedState())

        fsm.add_transition(source=STATE_INIT, dest=STATE_IDLE)
        fsm.add_transition(source=STATE_IDLE, dest=STATE_IDLE)
        fsm.add_transition(source=STATE_IDLE, dest=STATE_REGISTRATION)
        fsm.add_transition(source=STATE_REGISTRATION, dest=STATE_REGISTRATION)
        fsm.add_transition(source=STATE_REGISTRATION, dest=STATE_IDLE)  # Abort
        fsm.add_transition(source=STATE_REGISTRATION, dest=STATE_ROUND_SETUP)
        fsm.add_transition(source=STATE_ROUND_SETUP, dest=STATE_IDLE)  # Abort
        fsm.add_transition(source=STATE_ROUND_SETUP, dest=STATE_AGGREGATING)
        fsm.add_transition(source=STATE_AGGREGATING, dest=STATE_BROADCASTING)
        fsm.add_transition(source=STATE_AGGREGATING, dest=STATE_ROUND_SETUP)
        fsm.add_transition(source=STATE_BROADCASTING, dest=STATE_ROUND_SETUP)
        fsm.add_transition(source=STATE_BROADCASTING, dest=STATE_TERMINATED)
        fsm.add_transition(source=STATE_TERMINATED, dest=STATE_IDLE)

        self.add_behaviour(fsm)


async def main():
    import os
    import threading
    from dashboard.app import start_dashboard

    xmpp_server = os.environ.get("XMPP_SERVER", "localhost")
    jid = f"coordinator@{xmpp_server}"
    password = "password"

    # Start Flask UI in background thread
    dashboard_thread = threading.Thread(
        target=start_dashboard, args=(8080,), daemon=True
    )
    dashboard_thread.start()
    logger.info("Dashboard started on port 8080")

    agent = CoordinatorAgent(jid, password)
    agent.verify_security = False
    await agent.start(auto_register=True)

    try:
        while agent.is_alive():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
