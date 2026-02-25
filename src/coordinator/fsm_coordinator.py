import asyncio
import os
import time
import uuid
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
from common.aggregation_validator import validate_update

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
        self.agent.agent_tasks = {}
        self.agent.global_models = {
            "breast_cancer": {"weights": None, "intercept": None},
            "pneumonia_xray": {"weights": None, "intercept": None},
        }
        self.set_next_state(STATE_IDLE)


class IdleState(State):
    async def run(self):
        store.update_status("Idle - Waiting for Start")

        start_requested, target_rounds, _ = store.pop_start_request()
        if start_requested:
            logger.info(
                f"[{STATE_IDLE}] Start requested. Target rounds: {target_rounds}. Running all combined tasks."
            )
            self.agent.current_round = 0
            self.agent.max_rounds = (
                target_rounds if target_rounds > 0 else DEFAULT_MAX_ROUNDS
            )
            self.agent.global_models = {
                "breast_cancer": {"weights": None, "intercept": None},
                "pneumonia_xray": {"weights": None, "intercept": None},
            }
            store.reset_metrics()

            from common.mlflow_tracker import FederationTracker

            self.agent.tracker = FederationTracker()
            self.agent.target_rounds = (
                self.agent.max_rounds
            )  # Use max_rounds as target_rounds

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
                    task = getattr(header, "task", "breast_cancer")
                    self.agent.agent_tasks[str(msg.sender)] = task
                    logger.info(
                        f"[{STATE_REGISTRATION}] Agent {header.sender_id} joined for task: {task}."
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
            task = self.agent.agent_tasks.get(agent_jid, "breast_cancer")
            msg = Message(to=agent_jid)
            header = MessageHeader(
                msg_id=str(uuid.uuid4()),
                sender_id="coordinator",
                round_id=self.agent.current_round,
                msg_type=MSG_TYPE_ROUND_START,
                task=task,
            )

            # Comply with frozen ROUND_START schema
            gmodel = self.agent.global_models.get(
                task, {"weights": None, "intercept": None}
            )
            payload = {
                "weights": gmodel.get("weights") or [],
                "intercept": gmodel.get("intercept") or [0.0],
            }
            msg.body = build_message(header, payload)
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
            # Pre-aggregation sanity checks per task
            valid_updates_by_task = {}
            for update in received_updates:
                task = self.agent.agent_tasks.get(update["sender_id"], "breast_cancer")
                is_valid, reason = validate_update(
                    update, self.agent.global_models.get(task, {})
                )
                if is_valid:
                    valid_updates_by_task.setdefault(task, []).append(update)
                else:
                    logger.warning(
                        f"[{STATE_AGGREGATING}] Dropped {task} update "
                        f"from {update.get('sender_id', '?')}: {reason}"
                    )

            if not valid_updates_by_task:
                logger.warning(
                    f"[{STATE_AGGREGATING}] All updates rejected. Retrying round."
                )
                self.agent.current_round -= 1
                self.set_next_state(STATE_ROUND_SETUP)
                return

            sum_acc, sum_loss, sum_prec, sum_rec = 0.0, 0.0, 0.0, 0.0
            num_valid_all = 0
            total_samples_all = 0

            # Concurrent Multi-Task FedAvg
            for task, valid_updates in valid_updates_by_task.items():
                total_samples = sum(u.get("num_samples", 1) for u in valid_updates)
                total_samples_all += total_samples
                num_features = len(valid_updates[0]["weights"])
                num_intercept = len(valid_updates[0].get("intercept", [0.0]))

                avg_weights = [0.0] * num_features
                avg_intercept = [0.0] * num_intercept

                for update in valid_updates:
                    n = update.get("num_samples", 1)
                    weight = n / total_samples
                    for j in range(num_features):
                        avg_weights[j] += weight * update["weights"][j]
                    for j in range(num_intercept):
                        avg_intercept[j] += weight * update.get("intercept", [0.0])[j]

                    sum_acc += update.get("metrics", {}).get("accuracy", 0)
                    sum_loss += update.get("metrics", {}).get("loss", 0)
                    sum_prec += update.get("metrics", {}).get("precision", 0)
                    sum_rec += update.get("metrics", {}).get("recall", 0)
                    num_valid_all += 1

                    # Store per-agent metrics
                    store.add_agent_metrics(
                        update["sender_id"],
                        self.agent.current_round,
                        update.get("metrics", {}),
                        n,
                    )

                self.agent.global_models[task] = {
                    "weights": avg_weights,
                    "intercept": avg_intercept,
                }

            # Log unified convergence metrics across all multi-task models
            avg_acc = sum_acc / num_valid_all if num_valid_all else 0.0
            avg_loss = sum_loss / num_valid_all if num_valid_all else 0.0
            avg_prec = sum_prec / num_valid_all if num_valid_all else 0.0
            avg_rec = sum_rec / num_valid_all if num_valid_all else 0.0

            logger.info(
                f"[{STATE_AGGREGATING}] FedAvg complete (Total n={total_samples_all}). "
                f"Avg acc={avg_acc:.4f}, loss={avg_loss:.4f}, prec={avg_prec:.4f}, rec={avg_rec:.4f}"
            )
            store.add_round_metrics(
                self.agent.current_round,
                avg_acc,
                avg_loss,
                avg_prec,
                avg_rec,
                total_samples_all,
            )
            store.update_global_model(
                self.agent.global_models.get("breast_cancer")
                or self.agent.global_models.get("pneumonia_xray")
                or {"weights": [], "intercept": []}
            )

            if hasattr(self.agent, "tracker"):
                # Log individual and global metrics to MLflow
                avg_metrics = {
                    "accuracy": avg_acc,
                    "loss": avg_loss,
                    "precision": avg_prec,
                    "recall": avg_rec,
                }
                self.agent.tracker.log_round(
                    self.agent.current_round,
                    avg_metrics,
                    len(
                        received_updates
                    ),  # Use received_updates count for total participants
                    received_updates,
                )

            # Track per-agent privacy budgets
            agent_budgets = {}
            for (
                update
            ) in (
                received_updates
            ):  # Iterate over all received updates, not just valid ones
                aid = update.get("sender_id", "unknown").split("@")[0]
                remaining = update.get("privacy_remaining")
                if remaining is not None:
                    agent_budgets[aid] = remaining
                exhausted = update.get("budget_exhausted", False)
                if exhausted:
                    logger.warning(
                        f"[{STATE_AGGREGATING}] {aid} "
                        f"has EXHAUSTED its privacy budget!"
                    )

            if agent_budgets:
                store.update_privacy_state(agent_budgets)
                min_remaining = min(agent_budgets.values())
                if hasattr(self.agent, "tracker"):
                    self.agent.tracker.log_privacy(
                        self.agent.current_round,
                        min_remaining,
                    )

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
            task = self.agent.agent_tasks.get(agent_jid, "breast_cancer")
            msg = Message(to=agent_jid)
            header = MessageHeader(
                msg_id=str(uuid.uuid4()),
                sender_id="coordinator",
                round_id=self.agent.current_round,
                msg_type=MSG_TYPE_GLOBAL_UPDATE,
                task=task,
            )
            gmodel = self.agent.global_models.get(
                task, {"weights": None, "intercept": None}
            )
            payload = {
                "weights": gmodel.get("weights") or [],
                "intercept": gmodel.get("intercept") or [0.0],
            }
            msg.body = build_message(header, payload)
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

        # Gather final metrics from the last round
        history = store.state.get("rounds_history", [])
        final_metrics = {}
        if history:
            last = history[-1]
            final_metrics = {
                "accuracy": last.get("accuracy", 0),
                "loss": last.get("loss", 0),
                "precision": last.get("precision", 0),
                "recall": last.get("recall", 0),
            }

        # Log to MLflow
        run_id = ""
        if hasattr(self.agent, "tracker"):
            run_id = (
                self.agent.tracker.run.info.run_id if self.agent.tracker.run else ""
            )
            self.agent.tracker.log_final_model(self.agent.global_models, final_metrics)

        # Register version in store
        v = store.register_model_version(final_metrics, run_id)
        logger.info(f"[{STATE_TERMINATED}] Model v{v} registered.")

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
