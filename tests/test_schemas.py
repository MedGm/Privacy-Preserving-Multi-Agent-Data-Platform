"""Tests for frozen protocol schemas."""

from common.schemas import (
    PROTOCOL_VERSION,
    validate_payload,
    PAYLOAD_SCHEMAS,
)
from common.messages import (
    MessageHeader,
    build_message,
    parse_message,
)

# -- Protocol Version --


def test_protocol_version_exists():
    assert PROTOCOL_VERSION == "1.1"


def test_all_message_types_have_schemas():
    expected = {"JOIN", "ROUND_START", "MODEL_UPDATE", "GLOBAL_UPDATE", "ERROR"}
    assert set(PAYLOAD_SCHEMAS.keys()) == expected


# -- Schema Validation --


def test_join_payload():
    assert validate_payload({}, "JOIN") is True


def test_join_rejects_extra_keys():
    assert validate_payload({"hack": True}, "JOIN") is False


def test_model_update_valid():
    payload = {
        "weights": [0.1, 0.2],
        "intercept": [0.05],
        "num_samples": 100,
        "metrics": {"accuracy": 0.95, "loss": 0.05},
        "budget_exhausted": False,
        "privacy_remaining": 7.0,
    }
    assert validate_payload(payload, "MODEL_UPDATE") is True


def test_model_update_missing_weights():
    payload = {"intercept": [0.05]}
    # jsonschema validates 'required', so this should fail
    assert validate_payload(payload, "MODEL_UPDATE") is False


def test_model_update_extra_key():
    payload = {"weights": [0.1], "raw_data": [1, 2, 3]}
    assert validate_payload(payload, "MODEL_UPDATE") is False


def test_global_update_valid():
    payload = {"weights": [0.1, 0.2], "intercept": [0.05]}
    assert validate_payload(payload, "GLOBAL_UPDATE") is True


def test_unknown_type_rejected():
    assert validate_payload({}, "HACK") is False


# -- MessageHeader Protocol Version --


def test_header_has_protocol_version():
    h = MessageHeader(msg_id="test", sender_id="a", round_id=1, msg_type="JOIN")
    assert h.protocol_version == "1.1"


# -- Round-trip Serialization --


def test_build_parse_roundtrip():
    header = MessageHeader(
        msg_id="t1", sender_id="agent1", round_id=1, msg_type="MODEL_UPDATE"
    )
    payload = {
        "weights": [0.5, -0.3],
        "intercept": [0.1],
        "num_samples": 50,
    }
    raw = build_message(header, payload)
    h2, p2 = parse_message(raw)
    assert h2.protocol_version == "1.1"
    assert p2["weights"] == [0.5, -0.3]
