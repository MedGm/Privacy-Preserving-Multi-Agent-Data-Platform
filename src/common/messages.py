import json
from dataclasses import dataclass, asdict
from typing import Any, Dict
import time

# Defined Message Types
MSG_TYPE_JOIN = "JOIN"
MSG_TYPE_ROUND_START = "ROUND_START"
MSG_TYPE_MODEL_UPDATE = "MODEL_UPDATE"
MSG_TYPE_GLOBAL_UPDATE = "GLOBAL_UPDATE"
MSG_TYPE_ERROR = "ERROR"

ALLOWED_TYPES = {
    MSG_TYPE_JOIN,
    MSG_TYPE_ROUND_START,
    MSG_TYPE_MODEL_UPDATE,
    MSG_TYPE_GLOBAL_UPDATE,
    MSG_TYPE_ERROR,
}


@dataclass
class MessageHeader:
    msg_id: str
    sender_id: str
    round_id: int
    msg_type: str
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


def validate_schema(payload: Dict[str, Any], msg_type: str) -> bool:
    """
    Privacy Firewall: Validate payload structure against allowed schema.
    Returns True if valid, False otherwise.
    """
    if msg_type not in ALLOWED_TYPES:
        return False

    if msg_type == MSG_TYPE_MODEL_UPDATE:
        # Must only contain aggregated weights and optionally a sample count (scalar)
        if "weights" not in payload:
            return False
        # Privacy check: Ensure weights is a list of floats, not raw dicts/records
        if not isinstance(payload["weights"], list):
            return False
        # Optional: check if there is any disallowed extra key
        allowed_keys = {"weights", "intercept", "num_samples", "metrics"}
        if not set(payload.keys()).issubset(allowed_keys):
            return False

    return True


def build_message(header: MessageHeader, payload: Dict[str, Any]) -> str:
    """Serialize the header and payload to a JSON string."""
    if not validate_schema(payload, header.msg_type):
        raise ValueError(
            f"Schema validation failed for payload of type {header.msg_type}"
        )

    return json.dumps({"header": asdict(header), "payload": payload})


def parse_message(body: str) -> tuple[MessageHeader, Dict[str, Any]]:
    """Parse JSON string back to header and payload."""
    try:
        data = json.loads(body)
        header_data = data["header"]
        # Python dataclass initialization
        # 'type' is mapped to 'msg_type' in our dataclass to avoid reserved keyword conflicts if strictly typed
        if "msg_type" not in header_data and "type" in header_data:
            header_data["msg_type"] = header_data.pop("type")

        header = MessageHeader(**header_data)
        payload = data["payload"]

        if not validate_schema(payload, header.msg_type):
            raise ValueError(
                f"Privacy Violation: Incoming message payload failed "
                f"schema validation for {header.msg_type}"
            )

        return header, payload
    except Exception as e:
        raise ValueError(f"Malformed message: {e}")
