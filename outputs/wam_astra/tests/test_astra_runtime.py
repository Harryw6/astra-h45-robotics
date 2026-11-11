"""Fast is a service tier, independent of Astra reasoning effort."""
from copy import deepcopy
import importlib
import json

import pytest


class Client:
    model = "gpt-6-astra"
    effort = "low"

    def __init__(self, acknowledged="priority"):
        self.acknowledged = acknowledged
        self.calls = []

    def _request(self, method, params):
        self.calls.append((method, deepcopy(params)))
        if method in ("thread/start", "thread/resume"):
            return {"thread": {"id": "test-thread"}, "model": self.model,
                    "serviceTier": self.acknowledged, "reasoningEffort": self.effort}
        if method == "turn/start":
            return {"turn": {"id": "test-turn", "status": "inProgress"}}
        return {"unchanged": True}


class Session:
    def __init__(self, acknowledged="priority"):
        self._new_client = lambda: Client(acknowledged)
        self.client = self._new_client()


def api():
    return importlib.import_module("wam_astra.astra_runtime")


def test_thread_fast_configuration_preserves_model_effort_and_existing_config():
    session = Session()
    api().configure_astra_fast(session)
    params = {"model": "gpt-6-astra", "config": {"features.shell_tool": False}, "cwd": "synthetic"}
    before = deepcopy(params)
    session.client._request("thread/start", params)
    actual = session.client.calls[-1][1]
    assert actual["serviceTier"] == "priority"
    assert actual["config"]["features.fast_mode"] is True
    assert actual["config"]["model_reasoning_effort"] == "low"
    assert actual["config"]["features.shell_tool"] is False
    assert actual["model"] == "gpt-6-astra"
    assert session.client.effort == "low"
    assert params == before


def test_every_turn_requests_fast_without_overriding_effort_or_claiming_served_tier():
    session = Session()
    api().configure_astra_fast(session)
    session.client._request("thread/start", {})
    for _ in range(2):
        session.client._request("turn/start", {"threadId": "test-thread", "effort": "low",
                                                "serviceTierForTurn": "default"})
        params = session.client.calls[-1][1]
        assert params["serviceTier"] == "priority"
        assert params["serviceTierForTurn"] == "priority"
        assert params["effort"] == "low"
    receipt = api().astra_runtime_receipt(session)
    assert receipt["mode"] == "Fast"
    assert receipt["requested_service_tier"] == "priority"
    assert receipt["acknowledged_service_tier"] == "priority"
    assert receipt["thread_fast_acknowledged"] is True
    assert receipt["last_turn"]["request_accepted"] is True
    assert receipt["actual_inference_service_tier"] is None


@pytest.mark.parametrize("acknowledged", ["default", None, "flex", "ultrafast"])
def test_missing_or_other_tier_acknowledgement_is_explicit(acknowledged):
    session = Session(acknowledged)
    api().configure_astra_fast(session)
    with pytest.raises(RuntimeError, match="Fast service tier"):
        session.client._request("thread/start", {})
    receipt = api().astra_runtime_receipt(session)
    assert receipt["thread_fast_acknowledged"] is False
    assert receipt["acknowledged_service_tier"] == acknowledged


def test_fast_alias_ack_is_accepted():
    session = Session("fast")
    api().configure_astra_fast(session)
    session.client._request("thread/start", {})
    assert api().astra_runtime_receipt(session)["thread_fast_acknowledged"] is True


def test_recovered_client_reapplies_fast_before_resuming_same_thread():
    session = Session()
    api().configure_astra_fast(session)
    session.client._request("thread/start", {})
    session.client = session._new_client()
    assert api().astra_runtime_receipt(session)["thread_fast_acknowledged"] is False
    session.client._request("thread/resume", {"threadId": "test-thread"})
    assert session.client.calls[-1][1]["serviceTier"] == "priority"
    assert api().astra_runtime_receipt(session)["thread_id"] == "test-thread"


def test_turn_requires_current_client_thread_ack():
    session = Session()
    api().configure_astra_fast(session)
    with pytest.raises(RuntimeError, match="acknowledged"):
        session.client._request("turn/start", {"effort": "low"})
    assert session.client.calls == []


def test_unrelated_requests_are_untouched_and_receipt_is_a_copy():
    session = Session()
    api().configure_astra_fast(session)
    assert session.client._request("account/read", {"refreshToken": False}) == {"unchanged": True}
    assert session.client.calls == [("account/read", {"refreshToken": False})]
    result = api().astra_runtime_receipt(session)
    result["mode"] = "changed"
    assert api().astra_runtime_receipt(session)["mode"] == "Fast"


def test_idempotent_configuration_journals_only_metadata(tmp_path):
    session = Session()
    journal = tmp_path / "fast.jsonl"
    api().configure_astra_fast(session, event_log=journal)
    api().configure_astra_fast(session, event_log=journal)
    session.client._request("thread/start", {"baseInstructions": "SECRET_TEST_PROMPT"})
    session.client._request("turn/start", {"input": [{"text": "SECRET_TEST_IMAGE_OR_TEXT"}], "effort": "low"})
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["method"] == "thread/start"
    assert rows[1]["method"] == "turn/start"
    assert "SECRET_TEST" not in journal.read_text()


def test_helper_does_not_rewrite_a_different_model():
    session = Session()
    session.client.model = "different-model"
    with pytest.raises(ValueError, match="gpt-6-astra"):
        api().configure_astra_fast(session)
    assert session.client.model == "different-model"
