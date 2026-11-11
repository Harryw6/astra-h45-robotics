"""Per-session Astra Fast configuration for the pinned GPT-Policy client.

Codex's installed model catalog names the ``priority`` service tier "Fast";
``fast`` is accepted as an alias and normalizes to ``priority`` in thread ACKs.
Reasoning effort is independent and is preserved from the client.

References: https://learn.chatgpt.com/docs/agent-configuration/speed and
https://learn.chatgpt.com/docs/app-server. No global Codex configuration is
written. The app-server thread ACK proves its configured tier, not the actual
backend processing tier or achieved latency for an individual inference.
"""
from copy import deepcopy
import json
from pathlib import Path
import time


ASTRA_MODEL = "gpt-6-astra"
FAST_SERVICE_TIER = "priority"
_FAST_ALIASES = {"priority", "fast"}
_THREAD_METHODS = {"thread/start", "thread/resume"}


def astra_runtime_receipt(session):
    """Return configuration/request evidence without prompts or credentials."""
    return deepcopy(getattr(session, "_astra_runtime_state", None))


def configure_astra_fast(session, *, event_log=None):
    """Configure current and recovered clients before session.start().

Wrap only this session's native request method and recovery factory. Explicit
non-Fast or absent thread acknowledgements raise before any model turn can be
sent; no automatic Standard fallback is hidden from the caller. Callers may
handle that setup error before taking control of robot transports.
"""
    if hasattr(session, "_astra_runtime_state"):
        return astra_runtime_receipt(session)
    if session.client is not None and session.client.model != ASTRA_MODEL:
        raise ValueError("Fast Astra configuration requires model gpt-6-astra")
    journal = Path(event_log) if event_log is not None else None
    state = {
        "model": ASTRA_MODEL,
        "mode": "Fast",
        "requested_service_tier": FAST_SERVICE_TIER,
        "reasoning_effort": getattr(session.client, "effort", None),
        "acknowledged_service_tier": None,
        "thread_fast_acknowledged": False,
        "thread_id": None,
        "last_turn": None,
        "actual_inference_service_tier": None,
        "evidence_scope": "app_server_thread_configuration_and_request_acceptance",
    }
    session._astra_runtime_state = state

    def save_event(method, **fields):
        if journal is not None:
            journal.parent.mkdir(parents=True, exist_ok=True)
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"at_s": time.time(), "method": method,
                                         **astra_runtime_receipt(session), **fields},
                                        ensure_ascii=False, allow_nan=False) + "\n")

    def configure(client):
        if client.model != ASTRA_MODEL:
            raise ValueError("Fast Astra configuration requires model gpt-6-astra")
        original_request = client._request
        acknowledged = False
        state.update(reasoning_effort=client.effort, acknowledged_service_tier=None,
                     thread_fast_acknowledged=False, last_turn=None)

        def request(method, params):
            nonlocal acknowledged
            if method not in _THREAD_METHODS and method != "turn/start":
                return original_request(method, params)
            outgoing = dict(params)
            if outgoing.get("model", client.model) != ASTRA_MODEL:
                raise ValueError("Fast Astra request must preserve model gpt-6-astra")
            outgoing["serviceTier"] = FAST_SERVICE_TIER
            if method in _THREAD_METHODS:
                acknowledged = False
                state.update(thread_fast_acknowledged=False, acknowledged_service_tier=None)
                overrides = dict(outgoing.get("config") or {})
                overrides.update({"features.fast_mode": True, "model_reasoning_effort": client.effort})
                outgoing["config"] = overrides
            else:
                if not acknowledged:
                    raise RuntimeError("Fast service tier has not been acknowledged for this client thread")
                # A one-turn override takes precedence over the thread tier.
                outgoing["serviceTierForTurn"] = FAST_SERVICE_TIER
                state["last_turn"] = {"requested_service_tier": FAST_SERVICE_TIER,
                                      "reasoning_effort": outgoing.get("effort", client.effort),
                                      "request_accepted": False, "turn_id": None}
            try:
                result = original_request(method, outgoing)
            except Exception as exc:
                save_event(method, error_type=type(exc).__name__)
                raise
            if method in _THREAD_METHODS:
                tier = result.get("serviceTier")
                acknowledged = tier in _FAST_ALIASES and result.get("model") == ASTRA_MODEL
                state.update(acknowledged_service_tier=tier, thread_fast_acknowledged=acknowledged,
                             thread_id=(result.get("thread") or {}).get("id"))
                save_event(method)
                if not acknowledged:
                    raise RuntimeError(f"Fast service tier was not acknowledged: model={result.get('model')!r}, tier={tier!r}")
            else:
                state["last_turn"].update(request_accepted=True,
                                          turn_id=(result.get("turn") or {}).get("id"))
                save_event(method)
            return result

        client._request = request
        return client

    original_factory = session._new_client

    def new_client():
        return configure(original_factory())

    session._new_client = new_client
    if session.client is not None:
        configure(session.client)
    return astra_runtime_receipt(session)
