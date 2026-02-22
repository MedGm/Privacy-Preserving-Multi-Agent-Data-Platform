"""
Unit tests for the Privacy Firewall and Message Schema.

Tests validate:
  - Schema validation rejects forbidden payloads (privacy enforcement)
  - Message round-trip serialization (build → parse)
  - Edge cases: malformed JSON, missing headers, wrong types
"""

import sys
import os
import pytest
import json
import time

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from common.messages import (  # noqa: E402
    MSG_TYPE_JOIN,
    MSG_TYPE_ROUND_START,
    MSG_TYPE_MODEL_UPDATE,
    MSG_TYPE_GLOBAL_UPDATE,
    MSG_TYPE_ERROR,
    ALLOWED_TYPES,
    MessageHeader,
    validate_schema,
    build_message,
    parse_message,
)

# ---------------------------------------------------------------------------
# Privacy Firewall Tests
# ---------------------------------------------------------------------------


class TestPrivacyFirewall:
    """Ensure the schema validator blocks forbidden data structures."""

    def test_rejects_unknown_message_type(self):
        assert validate_schema({}, "RAW_DATA") is False
        assert validate_schema({}, "DUMP") is False

    def test_rejects_model_update_without_weights(self):
        payload = {"gradients": [0.1, 0.2]}
        assert validate_schema(payload, MSG_TYPE_MODEL_UPDATE) is False

    def test_rejects_model_update_with_non_list_weights(self):
        payload = {"weights": "not_a_list"}
        assert validate_schema(payload, MSG_TYPE_MODEL_UPDATE) is False

    def test_rejects_model_update_with_extra_keys(self):
        payload = {
            "weights": [0.1, 0.2],
            "raw_records": [{"id": 1, "name": "Alice"}],
        }
        assert validate_schema(payload, MSG_TYPE_MODEL_UPDATE) is False

    def test_accepts_valid_model_update(self):
        payload = {"weights": [0.5, 0.3], "num_samples": 100}
        assert validate_schema(payload, MSG_TYPE_MODEL_UPDATE) is True

    def test_accepts_model_update_with_metrics(self):
        payload = {
            "weights": [0.5, 0.3],
            "num_samples": 100,
            "metrics": {"loss": 0.01},
        }
        assert validate_schema(payload, MSG_TYPE_MODEL_UPDATE) is True

    def test_accepts_join_payload(self):
        payload = {"status": "available", "capabilities": ["regression"]}
        assert validate_schema(payload, MSG_TYPE_JOIN) is True

    def test_accepts_round_start_payload(self):
        payload = {"task": "regression", "hyperparams": {}}
        assert validate_schema(payload, MSG_TYPE_ROUND_START) is True

    def test_accepts_global_update_payload(self):
        payload = {"weights": [0.4, 0.6]}
        assert validate_schema(payload, MSG_TYPE_GLOBAL_UPDATE) is True


# ---------------------------------------------------------------------------
# Message Round-Trip Tests
# ---------------------------------------------------------------------------


class TestMessageRoundTrip:
    """Verify serialization → deserialization preserves data."""

    def _make_header(self, msg_type, round_id=1):
        return MessageHeader(
            msg_id="test_001",
            sender_id="agent1",
            round_id=round_id,
            msg_type=msg_type,
            timestamp=1000.0,
        )

    def test_model_update_round_trip(self):
        header = self._make_header(MSG_TYPE_MODEL_UPDATE)
        payload = {"weights": [0.1, 0.2, 0.3], "num_samples": 50}
        body = build_message(header, payload)

        parsed_header, parsed_payload = parse_message(body)
        assert parsed_header.msg_type == MSG_TYPE_MODEL_UPDATE
        assert parsed_header.sender_id == "agent1"
        assert parsed_payload["weights"] == [0.1, 0.2, 0.3]
        assert parsed_payload["num_samples"] == 50

    def test_join_round_trip(self):
        header = self._make_header(MSG_TYPE_JOIN, round_id=0)
        payload = {"status": "available", "capabilities": ["clustering"]}
        body = build_message(header, payload)

        parsed_header, parsed_payload = parse_message(body)
        assert parsed_header.msg_type == MSG_TYPE_JOIN
        assert parsed_payload["status"] == "available"

    def test_round_start_round_trip(self):
        header = self._make_header(MSG_TYPE_ROUND_START, round_id=3)
        payload = {"task": "anomaly_detection", "hyperparams": {"lr": 0.01}}
        body = build_message(header, payload)

        parsed_header, parsed_payload = parse_message(body)
        assert parsed_header.round_id == 3
        assert parsed_payload["task"] == "anomaly_detection"

    def test_global_update_round_trip(self):
        header = self._make_header(MSG_TYPE_GLOBAL_UPDATE)
        payload = {"weights": [0.5, 0.5]}
        body = build_message(header, payload)

        parsed_header, parsed_payload = parse_message(body)
        assert parsed_payload["weights"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test malformed inputs and boundary conditions."""

    def test_parse_malformed_json(self):
        with pytest.raises(ValueError, match="Malformed message"):
            parse_message("not valid json {{{")

    def test_parse_missing_header(self):
        body = json.dumps({"payload": {"weights": [1.0]}})
        with pytest.raises(ValueError, match="Malformed message"):
            parse_message(body)

    def test_parse_missing_payload(self):
        body = json.dumps(
            {
                "header": {
                    "msg_id": "x",
                    "sender_id": "a",
                    "round_id": 0,
                    "msg_type": MSG_TYPE_JOIN,
                    "timestamp": 1.0,
                }
            }
        )
        with pytest.raises(ValueError, match="Malformed message"):
            parse_message(body)

    def test_build_rejects_invalid_payload(self):
        header = MessageHeader(
            msg_id="x",
            sender_id="a",
            round_id=1,
            msg_type=MSG_TYPE_MODEL_UPDATE,
        )
        with pytest.raises(ValueError, match="Schema validation failed"):
            build_message(header, {"raw_data": [1, 2, 3]})

    def test_header_auto_timestamp(self):
        before = time.time()
        h = MessageHeader(msg_id="t", sender_id="a", round_id=0, msg_type=MSG_TYPE_JOIN)
        after = time.time()
        assert before <= h.timestamp <= after

    def test_allowed_types_set(self):
        expected = {
            MSG_TYPE_JOIN,
            MSG_TYPE_ROUND_START,
            MSG_TYPE_MODEL_UPDATE,
            MSG_TYPE_GLOBAL_UPDATE,
            MSG_TYPE_ERROR,
        }
        assert ALLOWED_TYPES == expected
