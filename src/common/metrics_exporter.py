"""
Prometheus metrics exporter for the federation platform.

Exposes system-level and ML-level metrics for scraping.
"""

from common.logger import setup_logger

try:
    from prometheus_client import Gauge, Counter, generate_latest

    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

logger = setup_logger("PrometheusMetrics")

if PROM_AVAILABLE:
    # -- Federation-level metrics --
    FED_ROUND = Gauge(
        "federation_current_round",
        "Current federation round number",
    )
    FED_AGENTS = Gauge(
        "federation_connected_agents",
        "Number of connected agents",
    )
    FED_STATUS = Gauge(
        "federation_status_code",
        "Federation status (0=idle, 1=running, 2=terminated)",
    )

    # -- Aggregation metrics --
    FED_ACCURACY = Gauge(
        "federation_global_accuracy",
        "Latest global model accuracy",
    )
    FED_LOSS = Gauge(
        "federation_global_loss",
        "Latest global model loss",
    )
    FED_PRECISION = Gauge(
        "federation_global_precision",
        "Latest global model precision",
    )
    FED_RECALL = Gauge(
        "federation_global_recall",
        "Latest global model recall",
    )

    # -- Privacy metrics --
    PRIVACY_MIN_REMAINING = Gauge(
        "federation_privacy_min_remaining",
        "Minimum remaining privacy budget across agents",
    )
    PRIVACY_ANY_EXHAUSTED = Gauge(
        "federation_privacy_any_exhausted",
        "1 if any agent has exhausted their privacy budget",
    )

    # -- Aggregation validation --
    UPDATES_REJECTED = Counter(
        "federation_updates_rejected_total",
        "Total number of rejected model updates",
    )
    MODEL_VERSIONS = Gauge(
        "federation_model_versions_total",
        "Total number of registered model versions",
    )


def update_metrics_from_store(store):
    """Pull latest state from FederationStore and push to Prometheus gauges."""
    if not PROM_AVAILABLE:
        return

    state = store.state

    # Federation state
    FED_ROUND.set(state.get("current_round", 0))
    FED_AGENTS.set(len(state.get("connected_agents", [])))

    status_str = state.get("status", "Idle").lower()
    status_map = {"idle": 0, "running": 1, "terminated": 2}
    FED_STATUS.set(status_map.get(status_str, 0))

    # Latest round metrics
    history = state.get("rounds_history", [])
    if history:
        last = history[-1]
        FED_ACCURACY.set(last.get("accuracy", 0))
        FED_LOSS.set(last.get("loss", 0))
        FED_PRECISION.set(last.get("precision", 0))
        FED_RECALL.set(last.get("recall", 0))

    # Privacy
    ps = state.get("privacy_state", {})
    min_rem = ps.get("min_remaining")
    if min_rem is not None:
        PRIVACY_MIN_REMAINING.set(min_rem)
    PRIVACY_ANY_EXHAUSTED.set(1 if ps.get("any_exhausted") else 0)

    # Model versions
    MODEL_VERSIONS.set(len(state.get("model_versions", [])))


def get_metrics_text(store):
    """Generate Prometheus text exposition format."""
    if not PROM_AVAILABLE:
        return "# prometheus_client not installed\n"
    update_metrics_from_store(store)
    return generate_latest().decode("utf-8")
