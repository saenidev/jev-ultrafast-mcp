"""An HTTP 400 from the decision model: diagnose it, write it down, and retry once with less.

Measured 2026-09-26 against api.typesafe.ai with a Google-Flights-like results page (250 elements,
link names of ~160 characters): the request was 96,979 bytes on the wire (the click head offered
120 targets), and TypeSafe answered

    HTTP 400 {"detail":{"error_type":"max_tokens_exceeded"}}

The same request with the click head cut to 80 targets was answered; 100 targets was not. So a 400
on a big page is the request being too big, and the fix is a smaller request: fewer targets in each
head and fewer elements in the state, goal-named and reachable ones kept. Halved once (52,263
bytes) the live API answered it.

No network: the provider is faked at `policy.CLIENT`.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from jev_ultrafast_mcp import policy
from jev_ultrafast_mcp.config import Config
from jev_ultrafast_mcp.observe import Element, Observation

FLIGHT = ("From {price} US dollars. 1 stop flight with Korean Air and Thai Airways. Leaves "
          "Suvarnabhumi Airport at 10:{minute:02d} PM on Sunday, October 11 and arrives at Incheon "
          "International Airport at 9:25 AM on Monday, October 12. Total duration 9 hr 30 min.")


def _results(n: int = 250) -> Observation:
    elements = [Element(ref=f"e{i}", role="button", name=name) for i, name in
                enumerate(["Where from?", "Where to?", "Search", "Stops", "Price", "Airlines"], 1)]
    for i in range(n - len(elements)):
        elements.append(Element(ref=f"e{100 + i}", role="link",
                                name=FLIGHT.format(price=198 + 3 * i, minute=i % 60)[:160],
                                context=f"Korean Air · BKK–ICN · ${198 + 3 * i}"))
    return Observation(
        url="https://www.google.com/travel/flights/search?tfs=secret-query", title="Flights",
        text="Flights results page. " * 250, elements=elements, digest="d", text_digest="t",
        page_key="k", scroll={"y": 0}, reachable=n, omitted=0, overlays=[], cross_frames=0,
        cross_frame_srcs=[],
    )


def _cfg(tmp_path) -> Config:
    return dataclasses.replace(Config.from_env(), typesafe_key="test-key", state_dir=tmp_path)


class _Response:
    def __init__(self, status: int, payload: object):
        self.status_code = status
        self.is_error = status >= 400
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def _answer(body: dict) -> dict:
    answers = {}
    for name, question in body["questions"].items():
        ids = sorted(question["criteria"])
        choice = "CLICK" if name == "operation" else ids[0]
        share = 0.4 / max(1, len(ids) - 1)
        probabilities = ({ids[0]: 1.0} if len(ids) == 1
                         else {item: (0.6 if item == choice else share) for item in ids})
        answers[name] = {"choice": choice, "confidence": 0.9, "probabilities": probabilities}
    return {"answers": answers}


class _Client:
    """Answers each request with the next scripted status; records every body sent."""

    def __init__(self, statuses: list[int]):
        self.statuses = list(statuses)
        self.bodies: list[dict] = []

    def post(self, _url, json=None, headers=None):  # noqa: A002 - httpx's own keyword
        self.bodies.append(json)
        status = self.statuses.pop(0)
        if status == 400:
            return _Response(400, {"detail": {"error_type": "max_tokens_exceeded"}})
        return _Response(200, _answer(json))


@pytest.fixture
def unbudgeted(monkeypatch):
    """Send the full request first, as before the pre-send trim, so the provider sees it whole."""
    monkeypatch.setattr(policy, "REQUEST_BUDGET_BYTES", 10**9)


def _size(body: dict) -> int:
    return len(json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode())


GOAL = "Open the Korean Air flight from 198 US dollars"


def test_a_400_is_retried_once_with_a_smaller_request(monkeypatch, tmp_path, unbudgeted):
    client = _Client([400, 200])
    monkeypatch.setattr(policy, "CLIENT", client)

    decision = policy.choose(_cfg(tmp_path), _results(), GOAL, [])

    assert decision["operation"] == "CLICK" and decision["ref"], decision
    assert len(client.bodies) == 2
    first, second = client.bodies
    assert _size(second) < _size(first)
    assert len(second["questions"]["click_target"]["criteria"]) <= \
        len(first["questions"]["click_target"]["criteria"]) // 2
    assert len(second["state"]["elements"]) < len(first["state"]["elements"])
    # The goal-named control stays in both: halving must not drop what the goal is about.
    for body in (first, second):
        assert "e100" in body["questions"]["click_target"]["criteria"]
    assert decision.get("reduced"), "the decision says it was made on a reduced request"


def test_two_400s_are_a_typed_failure_that_names_the_size(monkeypatch, tmp_path, unbudgeted):
    client = _Client([400, 400])
    monkeypatch.setattr(policy, "CLIENT", client)

    with pytest.raises(policy.TurboUnavailable) as caught:
        policy.choose(_cfg(tmp_path), _results(), GOAL, [])

    message = str(caught.value)
    assert len(client.bodies) == 2, "exactly one retry"
    assert "HTTP 400" in message and "no action executed" in message
    assert "max_tokens_exceeded" in message
    assert f"{_size(client.bodies[-1]):,} bytes" in message, message
    assert "elements" in message and "targets" in message


def test_a_400_leaves_a_redacted_dump(monkeypatch, tmp_path, unbudgeted):
    client = _Client([400, 200])
    monkeypatch.setattr(policy, "CLIENT", client)

    policy.choose(_cfg(tmp_path), _results(), GOAL, [])

    dump_path = tmp_path / "last-http-error.json"
    assert dump_path.exists()
    raw = dump_path.read_text(encoding="utf-8")
    dump = json.loads(raw)
    assert dump["status"] == 400
    assert "max_tokens_exceeded" in dump["response"]
    assert dump["request"]["bytes"] == _size(client.bodies[0])
    assert dump["request"]["elements"] == 250
    assert dump["request"]["questions"]["click_target"]["targets"] == 120
    # Shape only: no page text, no element names, no URL query, no key.
    for leak in ("Suvarnabhumi", "Flights results page", "secret-query", "test-key", "Korean Air"):
        assert leak not in raw, leak


def test_other_errors_are_not_retried_with_less(monkeypatch, tmp_path):
    class _Unauthorized(_Client):
        def post(self, _url, json=None, headers=None):  # noqa: A002
            self.bodies.append(json)
            return _Response(401, {"error": "bad key"})

    client = _Unauthorized([])
    monkeypatch.setattr(policy, "CLIENT", client)

    with pytest.raises(policy.TurboUnavailable, match="HTTP 401"):
        policy.choose(_cfg(tmp_path), _results(), GOAL, [])
    assert len(client.bodies) == 1


def test_a_small_page_is_sent_whole(monkeypatch, tmp_path):
    client = _Client([200])
    monkeypatch.setattr(policy, "CLIENT", client)

    policy.choose(_cfg(tmp_path), _results(40), GOAL, [])

    assert len(client.bodies) == 1
    assert len(client.bodies[0]["state"]["elements"]) == 40
    assert "elements_omitted" not in client.bodies[0]["state"]


def test_an_oversized_request_is_trimmed_before_it_is_sent(monkeypatch, tmp_path):
    """The measured page is over the limit every time: pay one request, not a 400 and a retry."""
    client = _Client([200])
    monkeypatch.setattr(policy, "CLIENT", client)

    decision = policy.choose(_cfg(tmp_path), _results(), GOAL, [])

    assert len(client.bodies) == 1, "a request known to be too big is not sent first"
    assert _size(client.bodies[0]) <= policy.REQUEST_BUDGET_BYTES
    assert client.bodies[0]["state"]["elements_omitted"] > 0
    assert decision.get("reduced")
