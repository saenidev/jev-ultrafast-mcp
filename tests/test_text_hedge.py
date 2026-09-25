"""The text helper's slow tail is cut by a second request; its answers and failures are unchanged."""
from __future__ import annotations

import threading
import time

import pytest

from jev_ultrafast_mcp import policy


@pytest.fixture(autouse=True)
def quick_hedge(monkeypatch):
    monkeypatch.setattr(policy, "TEXT_HEDGE_AFTER", 0.1)


def test_a_fast_answer_sends_one_request(monkeypatch):
    calls = []
    monkeypatch.setattr(policy, "_post", lambda *a: calls.append(a) or {"ok": 1})
    assert policy._hedged_post("u", "k", {}) == {"ok": 1}
    assert len(calls) == 1


def test_a_late_first_request_is_raced_by_a_second(monkeypatch):
    lock = threading.Lock()
    count = {"n": 0}

    def post(*_a):
        with lock:
            count["n"] += 1
            n = count["n"]
        if n == 1:
            time.sleep(2.0)
            return {"from": "first"}
        return {"from": "second"}

    monkeypatch.setattr(policy, "_post", post)
    started = time.monotonic()
    assert policy._hedged_post("u", "k", {}) == {"from": "second"}
    assert time.monotonic() - started < 1.0, "the slow first request was waited out"


def test_one_failed_request_does_not_fail_the_race(monkeypatch):
    count = {"n": 0}

    def post(*_a):
        count["n"] += 1
        if count["n"] == 1:
            time.sleep(0.3)
            raise policy.TurboUnavailable("HTTP 429")
        time.sleep(0.4)
        return {"from": "second"}

    monkeypatch.setattr(policy, "_post", post)
    assert policy._hedged_post("u", "k", {}) == {"from": "second"}


def test_both_failing_reports_the_failure(monkeypatch):
    def post(*_a):
        time.sleep(0.15)
        raise policy.TurboUnavailable("openrouter: HTTP 429")

    monkeypatch.setattr(policy, "_post", post)
    with pytest.raises(policy.TurboUnavailable, match="429"):
        policy._hedged_post("u", "k", {})


def test_decisions_are_never_hedged():
    import inspect
    assert "_hedged_post" not in inspect.getsource(policy.choose), (
        "the decision request is billed per call and must stay single")
