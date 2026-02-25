import json
from dataclasses import dataclass, asdict
from typing import Any, Dict
import time

from common.schemas import (
    PROTOCOL_VERSION,
    PAYLOAD_SCHEMAS,
    validate_payload,
)

# Defined Message Types
MSG_TYPE_JOIN = "JOIN"
MSG_TYPE_ROUND_START = "ROUND_START"
MSG_TYPE_MODEL_UPDATE = "MODEL_UPDATE"
MSG_TYPE_GLOBAL_UPDATE = "GLOBAL_UPDATE"
MSG_TYPE_ERROR = "ERROR"

ALLOWED_TYPES = set(PAYLOAD_SCHEMAS.keys())


@dataclass
class MessageHeader:
    msg_id: str
    sender_id: str
    round_id: int
    msg_type: str
    task: str = "breast_cancer"
    timestamp: float = 0.0
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


def validate_schema(payload: Dict[str, Any], msg_type: str) -> bool:
    """
    Privacy Firewall: Validate payload structure against frozen schema.
    Delegates to schemas.validate_payload().
    """
    return validate_payload(payload, msg_type)


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

        # Backward compat: old messages may not have protocol_version
        if "protocol_version" not in header_data:
            header_data["protocol_version"] = PROTOCOL_VERSION

        if "task" not in header_data:
            header_data["task"] = "breast_cancer"

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
