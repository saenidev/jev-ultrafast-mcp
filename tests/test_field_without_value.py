"""Two ways a goal used to die one field short of done, and how the loop handles them now.

1. A field the goal says nothing about. The decision model may pick TYPE_TEXT on an optional
   field (httpbin's "Delivery instructions", a newsletter box beside GitHub's search). The text
   helper rightly returns {"text": null} instead of inventing a value -- and that refusal used to
   be raised as TurboUnavailable, which ended the goal. Now it is a skipped step: nothing typed,
   the step recorded, the field withdrawn from TYPE_TEXT so it is not offered again.

2. A typed query that still has to be sent. Jev's operations had no way to press Enter, so on a
   search box whose suggestion list does not submit (GitHub's "Search or jump to") the model
   clicked the suggestion list until the stall detector ended the goal. SUBMIT is offered on a
   filled, non-secret editable field and runs `type` with `submit: true` on its current value.
"""

from __future__ import annotations

import dataclasses

import pytest

from jev_ultrafast_mcp import policy, server
from jev_ultrafast_mcp.config import Config
from jev_ultrafast_mcp.observe import Element, Observation


def _observation(elements: list[Element]) -> Observation:
    return Observation(
        url="https://example.test/", title="Example", text="Example page", elements=elements,
        digest="d", text_digest="t", page_key="k", scroll={"y": 0}, reachable=len(elements),
        omitted=0, overlays=[], cross_frames=0, cross_frame_srcs=[],
    )


def _answer(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _text_cfg() -> Config:
    return dataclasses.replace(Config.from_env(), text_model_key="test-key", text_model="test-model")


# ------------------------------------------------------------------ the text helper


def test_a_null_answer_is_its_own_refusal(monkeypatch):
    notes = Element(ref="e2", role="textbox", name="Delivery instructions", editable=True)
    monkeypatch.setattr(policy, "_post", lambda url, key, body: _answer('{"text": null}'))

    with pytest.raises(policy.NoValueForField) as caught:
        policy.text_for(_text_cfg(), "order a pizza", notes, _observation([notes]), [])

    assert "Delivery instructions" in str(caught.value)


def test_the_null_refusal_is_still_a_turbo_failure_to_older_callers():
    """Anything that only catches TurboUnavailable keeps working: the refusal is a subclass."""
    assert issubclass(policy.NoValueForField, policy.TurboUnavailable)


@pytest.mark.parametrize("content", ['{"text": "null"}', '{"text": null, "why": "x"}', '{"value": null}'])
def test_only_the_exact_null_answer_counts_as_no_value(monkeypatch, content):
    field = Element(ref="e1", role="textbox", name="Name", editable=True)
    monkeypatch.setattr(policy, "_post", lambda url, key, body: _answer(content))
    try:
        value = policy.text_for(_text_cfg(), "g", field, _observation([field]), [])
    except policy.NoValueForField:
        pytest.fail(f"{content!r} is not the documented null answer")
    except policy.TurboUnavailable:
        return
    assert value == "null", "a literal string 'null' is a value, not a refusal"


# ------------------------------------------------------------------ what is offered


def test_a_field_with_no_value_is_withdrawn_from_typing():
    name = Element(ref="e1", role="textbox", name="Name", editable=True)
    notes = Element(ref="e2", role="textbox", name="Notes", editable=True)
    history = [{"op": "type", "ref": "e2", "ok": False, "error": policy.NO_VALUE_ERROR}]

    _, heads = policy._operation_heads(_observation([name, notes]))
    policy.withdraw_valueless(heads, history)

    assert [e.ref for e in heads["TYPE_TEXT"]] == ["e1"]


def test_withdrawing_the_last_typeable_field_removes_the_operation():
    notes = Element(ref="e2", role="textbox", name="Notes", editable=True)
    order = Element(ref="e3", role="button", name="Order")
    history = [{"op": "type", "ref": "e2", "ok": False, "error": policy.NO_VALUE_ERROR}]

    _, heads = policy._operation_heads(_observation([notes, order]))
    policy.withdraw_valueless(heads, history)

    assert "TYPE_TEXT" not in heads, "offering it again only buys the same refusal"
    assert "e3" in [e.ref for e in heads["CLICK"]]


def test_submit_is_offered_only_on_a_filled_non_secret_field():
    filled = Element(ref="e1", role="searchbox", name="Search", editable=True, value="browser-use")
    empty = Element(ref="e2", role="textbox", name="Name", editable=True)
    secret = Element(ref="e3", role="textbox", name="Password", editable=True, value="x", secret=True)

    assert "SUBMIT" in filled.target_kinds()
    assert "SUBMIT" not in empty.target_kinds()
    assert "SUBMIT" not in secret.target_kinds()


def test_submit_runs_the_existing_type_verb():
    assert policy.OPERATION_TO_ACT["SUBMIT"] == "type"
    assert "SUBMIT" in policy.OPERATION_LABELS
    assert policy.ACT_TO_OPERATION["type"] == "TYPE_TEXT", (
        "a `type` step in the history must still count as TYPE_TEXT for stall detection")


# ------------------------------------------------------------------ the goal loop


class _Tab:
    """The parts of `browser.Session` the goal loop touches."""

    def __init__(self, observation: Observation):
        self._observation = observation
        self.history: list = []
        self.last = observation
        self.acts: list[list[dict]] = []

    def reset_progress(self):
        pass

    def page_is_idle(self):
        return True

    def navigate(self, url):
        pass

    def observe(self, **_kw):
        return self._observation

    def evaluate_js(self, _expr):
        raise AssertionError("no JS here")

    def act(self, ops, **_kw):
        from jev_ultrafast_mcp.browser import Step
        self.acts.append(ops)
        step = Step(op=ops[0]["op"], ok=True, ref=ops[0].get("ref"), ms=1)
        self.history.append(step)
        return {"ops": [step.to_dict()], "ok": True, "steps": len(self.acts), "page_changed": True}


def _drive(monkeypatch, tab, decisions, text_for):
    seen_history: list = []

    def fake_choose(_cfg, _obs, _goal, history):
        seen_history.append(list(history))
        return decisions.pop(0) if decisions else {"operation": "DONE", "ref": None, "confidence": 1.0}

    monkeypatch.setattr(server.policy, "available", lambda _cfg: True)
    monkeypatch.setattr(server.policy, "choose", fake_choose)
    monkeypatch.setattr(server.policy, "text_for", text_for)
    monkeypatch.setattr(server, "_session", lambda _name: tab)
    return server.browser_goal("order a pizza", max_steps=10, verbose=True), seen_history


def test_a_field_with_no_value_is_skipped_and_the_goal_goes_on(monkeypatch):
    notes = Element(ref="e2", role="textbox", name="Notes", editable=True)
    order = Element(ref="e3", role="button", name="Order")
    tab = _Tab(_observation([notes, order]))

    def text_for(*_a):
        raise policy.NoValueForField("The goal gives no value for 'Notes'; left it unchanged.")

    decisions = [
        {"operation": "TYPE_TEXT", "ref": "e2", "target": "Notes", "confidence": 0.8},
        {"operation": "CLICK", "ref": "e3", "target": "Order", "confidence": 0.9},
        {"operation": "DONE", "ref": None, "confidence": 0.9},
    ]
    out, seen = _drive(monkeypatch, tab, decisions, text_for)

    assert "status: done" in out, out
    assert "skipped" in out, out
    assert [ops[0]["op"] for ops in tab.acts] == ["click"], "nothing may be typed into Notes"
    assert seen[1][-1].get("error") == policy.NO_VALUE_ERROR, (
        "the next decision must see why that step did nothing")


def test_other_text_helper_failures_still_end_the_goal(monkeypatch):
    name = Element(ref="e1", role="textbox", name="Name", editable=True)
    tab = _Tab(_observation([name]))

    def text_for(*_a):
        raise policy.TurboUnavailable("openrouter: HTTP 429")

    decisions = [{"operation": "TYPE_TEXT", "ref": "e1", "target": "Name", "confidence": 0.8}]
    out, _ = _drive(monkeypatch, tab, decisions, text_for)

    assert "turbo_unavailable" in out or "429" in out, out
    assert tab.acts == [], "nothing may be typed when the helper failed"


def test_submit_retypes_the_field_s_own_value_and_presses_enter(monkeypatch):
    search = Element(ref="e1", role="searchbox", name="Search", editable=True, value="browser-use")
    tab = _Tab(_observation([search]))

    def text_for(*_a):
        raise AssertionError("SUBMIT must not ask the text helper for anything")

    decisions = [
        {"operation": "SUBMIT", "ref": "e1", "target": "Search", "confidence": 0.9},
        {"operation": "DONE", "ref": None, "confidence": 0.9},
    ]
    out, _ = _drive(monkeypatch, tab, decisions, text_for)

    assert tab.acts == [[{"op": "type", "ref": "e1", "text": "browser-use", "submit": True}]], tab.acts
    assert "status: done" in out, out
