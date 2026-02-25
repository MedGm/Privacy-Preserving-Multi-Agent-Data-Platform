"""
Frozen Protocol Contracts — v1.0

JSON Schema definitions for all inter-agent message types.
These schemas are the canonical specification of the federation protocol.

Protocol version: 1.0
Last frozen: 2026-02-24
"""

PROTOCOL_VERSION = "1.1"

# -- Schema Definitions --

HEADER_SCHEMA = {
    "type": "object",
    "required": ["msg_id", "sender_id", "round_id", "msg_type", "task"],
    "properties": {
        "msg_id": {"type": "string"},
        "sender_id": {"type": "string"},
        "round_id": {"type": "integer"},
        "msg_type": {
            "type": "string",
            "enum": [
                "JOIN",
                "ROUND_START",
                "MODEL_UPDATE",
                "GLOBAL_UPDATE",
                "ERROR",
            ],
        },
        "task": {"type": "string"},
        "timestamp": {"type": "number"},
        "protocol_version": {"type": "string"},
    },
    "additionalProperties": False,
}

JOIN_PAYLOAD_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}

ROUND_START_PAYLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "weights": {"type": "array", "items": {"type": "number"}},
        "intercept": {
            "type": "array",
            "items": {"type": "number"},
        },
    },
    "additionalProperties": False,
}

MODEL_UPDATE_PAYLOAD_SCHEMA = {
    "type": "object",
    "required": ["weights"],
    "properties": {
        "weights": {"type": "array", "items": {"type": "number"}},
        "intercept": {
            "type": "array",
            "items": {"type": "number"},
        },
        "num_samples": {"type": "integer"},
        "metrics": {
            "type": "object",
            "properties": {
                "accuracy": {"type": "number"},
                "loss": {"type": "number"},
                "precision": {"type": "number"},
                "recall": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "budget_exhausted": {"type": "boolean"},
        "privacy_remaining": {"type": "number"},
    },
    "additionalProperties": False,
}

GLOBAL_UPDATE_PAYLOAD_SCHEMA = {
    "type": "object",
    "required": ["weights"],
    "properties": {
        "weights": {"type": "array", "items": {"type": "number"}},
        "intercept": {
            "type": "array",
            "items": {"type": "number"},
        },
    },
    "additionalProperties": False,
}

ERROR_PAYLOAD_SCHEMA = {
    "type": "object",
    "properties": {
        "error": {"type": "string"},
    },
    "additionalProperties": False,
}

# Lookup by message type
PAYLOAD_SCHEMAS = {
    "JOIN": JOIN_PAYLOAD_SCHEMA,
    "ROUND_START": ROUND_START_PAYLOAD_SCHEMA,
    "MODEL_UPDATE": MODEL_UPDATE_PAYLOAD_SCHEMA,
    "GLOBAL_UPDATE": GLOBAL_UPDATE_PAYLOAD_SCHEMA,
    "ERROR": ERROR_PAYLOAD_SCHEMA,
}


def validate_payload(payload: dict, msg_type: str) -> bool:
    """
    Validate a message payload against the frozen protocol schema.

    Uses jsonschema if available, falls back to key-based checks.
    Returns True if valid, False otherwise.
    """
    schema = PAYLOAD_SCHEMAS.get(msg_type)
    if schema is None:
        return False

    try:
        from jsonschema import validate

        validate(instance=payload, schema=schema)
        return True
    except ImportError:
        # Fallback: basic key-set check
        allowed = set(schema.get("properties", {}).keys())
        return set(payload.keys()).issubset(allowed)
    except Exception:
        return False
