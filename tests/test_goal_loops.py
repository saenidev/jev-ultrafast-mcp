"""Multi-target cycles, measured on a live Google Flights calendar, pinned without the site.

(A) The calendar dialog was open and the model clicked three day buttons round and round --
    Oct 15, Oct 22, Oct 29, Oct 29 again, Oct 15 ... -- for fifty steps until `max_steps`. Every
    click was "ok" (the chosen day's name gained ", departure date"), so the no-change detector
    never fired, and the single-target rule (three repeats of one ref in eight steps) never saw
    three of any one ref. A cycle of two to four targets that has gone round twice on the same
    page is now withdrawn, the model is told why, and a third round ends the goal.
(B) The text helper answered 'BKK' for the Departure date field. A field that is plainly a date
    does not take a helper answer with no digit and no month in it.

No network: `_post` is replaced in every test.
"""

from __future__ import annotations

import dataclasses

import pytest

from jev_ultrafast_mcp import policy, server
from jev_ultrafast_mcp.browser import Step
from jev_ultrafast_mcp.config import Config
from jev_ultrafast_mcp.observe import Element, Observation

PAGE = "https://www.google.com/travel/flights/search"


def _observation(elements: list[Element], url: str = PAGE + "?tfs=abc") -> Observation:
    return Observation(
        url=url, title="Flights", text="Flights", elements=elements,
        digest="d", text_digest="t", page_key="k", scroll={"y": 0}, reachable=len(elements),
        omitted=0, overlays=[], cross_frames=0, cross_frame_srcs=[],
    )


def _cfg() -> Config:
    return dataclasses.replace(Config.from_env(), typesafe_key="k", text_model_key="test-key",
                               text_model="test-model")


def _click(ref: str, name: str = "", page: str | None = PAGE, ok: bool = True) -> dict:
    item = {"op": "click", "ref": ref, "target": name or ref, "ok": ok}
    if page is not None:
        item["page"] = page
    return item


def _days() -> list[Element]:
    return [Element(ref="e144", role="button", name="Thursday, October 15, 2026"),
            Element(ref="e151", role="button", name="Thursday, October 22, 2026"),
            Element(ref="e158", role="button", name="Thursday, October 29, 2026"),
            Element(ref="e165", role="button", name="Thursday, November 5, 2026"),
            Element(ref="e200", role="button", name="Done")]


# ------------------------------------------------------------------ detection


@pytest.mark.parametrize("refs", [
    ["e144", "e151", "e158", "e144", "e151", "e158"],               # period 3, exact
    ["e144", "e151", "e144", "e151"],                                # period 2
    ["e1", "e2", "e3", "e4", "e1", "e2", "e3", "e4"],                # period 4
    # The measured shape: the third day clicked twice in a row (its name changed in between).
    ["e144", "e151", "e158", "e158", "e144", "e151", "e158", "e158"],
    # The same three days, not in the same order the second time round.
    ["e144", "e151", "e158", "e151", "e144", "e158"],
])
def test_a_short_cycle_that_went_round_twice_is_found(refs):
    history = [_click(ref) for ref in refs]
    found = policy.loop_targets(history, PAGE)
    assert found == {"CLICK": set(refs)}


def test_every_step_being_ok_does_not_hide_the_cycle():
    history = [_click("e144", "Thursday, October 15, 2026"),
               _click("e151", "Thursday, October 22, 2026"),
               _click("e158", "Thursday, October 29, 2026"),
               _click("e158", "Thursday, October 29, 2026, departure date. ICN"),
               _click("e144", "Thursday, October 15, 2026, departure date. ICN"),
               _click("e151", "Thursday, October 22, 2026"),
               _click("e158", "Thursday, October 29, 2026")]
    assert all(item["ok"] for item in history)
    assert policy.loop_targets(history, PAGE) == {"CLICK": {"e144", "e151", "e158"}}


@pytest.mark.parametrize("refs", [
    ["e1", "e1", "e1", "e2", "e2"],          # steppers: 3 adults, 2 children
    ["e1", "e2", "e3", "e4", "e5"],          # a form filled field by field
    ["e144", "e151", "e158", "e144"],        # one round and a bit
    ["e1", "e2", "e1", "e3", "e1", "e4"],    # returning to one control between new ones
    ["e8", "e8"],                            # a single ref twice: the 3x rule's business
])
def test_ordinary_sequences_are_not_cycles(refs):
    assert policy.loop_targets([_click(ref) for ref in refs], PAGE) == {}


def test_a_cycle_is_only_counted_on_one_page():
    """Refs restart per document: e144 on the results page is not e144 in the calendar."""
    other = "https://www.google.com/travel/flights"
    history = [_click(r, page=other) for r in ("e144", "e151", "e158")]
    history += [_click(r) for r in ("e144", "e151", "e158")]
    assert policy.loop_targets(history, PAGE) == {}
    assert policy.loop_targets(history, other) == {}, "the cycle must be on the page being decided"


def test_a_page_is_its_path_not_its_query():
    """Flights rewrites `?tfs=` as dates change; the model never sees the query either."""
    history = [_click(r) for r in ("e1", "e2", "e1", "e2")]
    assert policy.loop_targets(history, PAGE + "?tfs=xyz#top") == {"CLICK": {"e1", "e2"}}


def test_a_type_cycle_withdraws_typing():
    history = [{"op": "type", "ref": "e5", "target": "Where to?", "ok": True, "page": PAGE},
               _click("e9"),
               {"op": "type", "ref": "e5", "target": "Where to?", "ok": True, "page": PAGE},
               _click("e9")]
    assert policy.loop_targets(history, PAGE) == {"TYPE_TEXT": {"e5"}, "CLICK": {"e9"}}


def test_a_submit_is_keyed_as_submit_not_typing():
    sent = {"op": "type", "ref": "e5", "target": "Search", "ok": True, "page": PAGE,
            "submitted": "kettle"}
    history = [sent, _click("e9"), sent, _click("e9")]
    assert policy.loop_targets(history, PAGE) == {"SUBMIT": {"e5"}, "CLICK": {"e9"}}


def test_a_third_round_is_exhaustion():
    two = [_click(r) for r in ("e144", "e151", "e158")] * 2
    assert not policy.loop_exhausted(two, PAGE)
    assert policy.loop_exhausted(two + [_click(r) for r in ("e144", "e151", "e158")], PAGE)
    assert not policy.loop_exhausted([_click("e8")] * 6, PAGE), "a single ref is not a cycle"


# ------------------------------------------------------------------ what the model is asked


def _sent(monkeypatch, elements, history, url=PAGE + "?tfs=abc"):
    sent: list[dict] = []

    def fake_post(_url, _key, body):
        sent.append(body)
        ids = list(body["questions"]["operation"]["criteria"])
        return {"answers": {"operation": {"choice": "DONE", "confidence": 1.0,
                                          "probabilities": {n: float(n == "DONE") for n in ids}}}}

    monkeypatch.setattr(policy, "_post", fake_post)
    decision = policy.choose(_cfg(), _observation(elements, url), "depart Oct 15", history)
    return sent[0], decision


def test_the_cycle_is_withdrawn_and_the_model_is_told(monkeypatch):
    history = [_click(r) for r in ("e144", "e151", "e158")] * 2
    body, decision = _sent(monkeypatch, _days(), history)
    offered = set(body["questions"]["click_target"]["criteria"])
    assert offered == {"e165", "e200"}
    note = body["state"]["loop_note"]
    assert "e144" in note and "e151" in note and "e158" in note
    assert decision["loop"] == {"CLICK": ["e144", "e151", "e158"]}


def test_withdrawing_every_click_leaves_the_other_operations(monkeypatch):
    """The head may empty: WAIT, DONE and BLOCKED are still there, and one more lap is not."""
    days = _days()[:3]
    field = Element(ref="e93", role="textbox", name="Departure", editable=True)
    history = [_click(r) for r in ("e144", "e151", "e158")] * 2
    body, _decision = _sent(monkeypatch, [*days, field], history)
    operations = set(body["questions"]["operation"]["criteria"])
    assert "CLICK" not in operations and "click_target" not in body["questions"]
    assert {"TYPE_TEXT", "WAIT", "DONE", "BLOCKED"} <= operations


def test_no_note_and_nothing_withdrawn_without_a_cycle(monkeypatch):
    body, decision = _sent(monkeypatch, _days(), [_click("e144"), _click("e151")])
    assert "loop_note" not in body["state"]
    assert len(body["questions"]["click_target"]["criteria"]) == 5
    assert "loop" not in decision


def test_the_page_stamp_never_reaches_the_model(monkeypatch):
    history = [_click("e144", page=PAGE + "?tfs=secret-ish")]
    body, _decision = _sent(monkeypatch, _days(), history)
    for item in body["state"]["recent_actions"]:
        assert "page" not in item
    assert "secret-ish" not in str(body)


def test_the_text_helper_never_sees_the_page_stamp_or_a_query(monkeypatch):
    bodies: list[dict] = []

    def helper(_url, _key, body):
        bodies.append(body)
        return {"choices": [{"message": {"content": '{"text": "Oct 15"}'}}]}

    monkeypatch.setattr(policy, "_post", helper)
    field = Element(ref="e93", role="textbox", name="Departure", editable=True)
    history = [{**_click("e144", page=PAGE + "?tfs=secret-ish"), "where": PAGE + "?q=hidden"}]
    policy.text_for(_cfg(), "depart on Oct 15", field, _observation([field]), history)
    sent = str(bodies[0])
    assert "secret-ish" not in sent and "hidden" not in sent and "'page'" not in sent


def test_the_rules_mention_the_loop_note():
    assert "loop_note" in " ".join(policy.NEXT_ACTION.split())


# ------------------------------------------------------------------ the goal loop


class _Tab:
    """A calendar page whose URL query changes on every click, as Flights' does."""

    def __init__(self, elements):
        self.elements = elements
        self.history: list = []
        self.clicks: list[str] = []
        self.n = 0
        self.last = self.observe()

    def reset_progress(self): pass
    def page_is_idle(self): return True
    def navigate(self, _url): pass

    def observe(self, **_kw):
        self.n += 1
        return _observation(self.elements, url=f"{PAGE}?tfs=v{self.n}")

    def act(self, ops, **_kw):
        op = ops[0]
        self.clicks.append(op.get("ref"))
        step = Step(op=op["op"], ok=True, ref=op.get("ref"), target=op.get("ref"), ms=1)
        self.history.append(step)
        self.last = self.observe()
        return {"ops": [step.to_dict()], "ok": True, "page_changed": True}


def _answer(body, op, target=None):
    ops = list(body["questions"]["operation"]["criteria"])
    answers = {"operation": {"choice": op, "confidence": 0.9,
                             "probabilities": {n: float(n == op) for n in ops}}}
    for qid, q in body["questions"].items():
        if qid.endswith("_target"):
            refs = list(q["criteria"])
            pick = target if target in refs else refs[0]
            answers[qid] = {"choice": pick, "confidence": 1.0,
                            "probabilities": {r: float(r == pick) for r in refs}}
    return {"answers": answers}


def test_the_goal_loop_cuts_a_three_day_cycle(monkeypatch):
    """End to end through `browser_goal`, with a model that cycles among three days for as long as
    they are offered (the measured behaviour) and otherwise confirms with Done."""
    tab = _Tab(_days())
    cycle = ["e144", "e151", "e158"]
    seen_notes: list[str] = []

    def cycling_model(_url, _key, body):
        if "loop_note" in body["state"]:
            seen_notes.append(body["state"]["loop_note"])
        if tab.clicks and tab.clicks[-1] == "e200":
            return _answer(body, "DONE")
        offered = set(body["questions"].get("click_target", {}).get("criteria", {}))
        nxt = cycle[len(tab.clicks) % 3]
        return _answer(body, "CLICK", nxt if nxt in offered else "e200")

    monkeypatch.setattr(server, "CONFIG", _cfg())
    monkeypatch.setattr(policy, "_post", cycling_model)
    monkeypatch.setattr(server, "_session", lambda _name: tab)
    out = server.browser_goal("Set departure to Thursday, October 15, 2026", max_steps=60,
                              verbose=True)

    assert tab.clicks == cycle * 2 + ["e200"], out
    assert "status: done" in out, out
    assert seen_notes, "the model must be told why the days stopped being offered"
    assert "looping among" in out, out


def test_a_cycle_that_survives_withdrawal_ends_the_goal(monkeypatch):
    """A decision that keeps choosing the cycle anyway -- whatever let it through -- stops the goal
    on the third lap instead of spending all of `max_steps`."""
    tab = _Tab(_days())
    order = iter(["e144", "e151", "e158"] * 30)
    asked: list[list[dict]] = []

    def stubborn(_cfg, _observation, _goal, history, **_kw):
        asked.append(history)
        ref = next(order)
        return {"operation": "CLICK", "ref": ref, "target": ref, "confidence": 0.9}

    monkeypatch.setattr(server.policy, "available", lambda _cfg: True)
    monkeypatch.setattr(server.policy, "choose", stubborn)
    monkeypatch.setattr(server, "_session", lambda _name: tab)
    out = server.browser_goal("Set departure", max_steps=60, verbose=True)

    assert "status: stopped: looping" in out, out
    assert len(tab.clicks) == 9, out
    assert all(item.get("page") == PAGE for item in asked[-1]), \
        "every history item carries the page path it was taken on"


def test_the_loop_history_reaches_back_beyond_ten_steps(monkeypatch):
    """A period-4 cycle needs eight steps and a third lap twelve; the window must hold them."""
    tab = _Tab(_days())
    order = iter(["e144", "e151", "e158", "e165"] * 30)

    def stubborn(_cfg, _observation, _goal, _history, **_kw):
        ref = next(order)
        return {"operation": "CLICK", "ref": ref, "target": ref, "confidence": 0.9}

    monkeypatch.setattr(server.policy, "available", lambda _cfg: True)
    monkeypatch.setattr(server.policy, "choose", stubborn)
    monkeypatch.setattr(server, "_session", lambda _name: tab)
    out = server.browser_goal("Set departure", max_steps=60, verbose=True)
    assert "status: stopped: looping" in out and len(tab.clicks) == 12, out


# ------------------------------------------------------------------ (B) date fields


class _Helper:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    def __call__(self, url, key, body):
        self.calls += 1
        return {"choices": [{"message": {"content": self.answer}}]}


def _helper_text(monkeypatch, field: Element, answer: str,
                 goal: str = "Fly Bangkok to Seoul departing October 15") -> str:
    helper = _Helper(answer)
    monkeypatch.setattr(policy, "_post", helper)
    value = policy.text_for(_cfg(), goal, field, _observation([field]), [])
    assert helper.calls == 1
    return value


@pytest.mark.parametrize("name, value", [
    ("Departure", ""), ("Return", ""), ("Date", ""), ("Check-in", ""), ("Check-out", ""),
    ("DOB", ""), ("Date of birth", ""), ("Travel dates", ""), ("Date from", ""), ("Date to", ""),
    ("Leaving", ""),
    ("Notes", "2026-10-15"), ("Leaving on", "Thu, Oct 15"),
])
def test_a_date_field_refuses_a_helper_answer_that_is_no_date(monkeypatch, name, value):
    field = Element(ref="e93", role="textbox", name=name, value=value, editable=True)
    with pytest.raises(policy.NoValueForField) as caught:
        _helper_text(monkeypatch, field, '{"text": "BKK"}')
    assert "date" in str(caught.value)


@pytest.mark.parametrize("answer", ["Thu, Oct 15", "2026-10-15", "15/10/2026", "October 15",
                                    "Oct 15, 2026"])
def test_a_date_field_takes_a_date(monkeypatch, answer):
    field = Element(ref="e93", role="textbox", name="Departure", editable=True)
    assert _helper_text(monkeypatch, field, '{"text": "%s"}' % answer) == answer


@pytest.mark.parametrize("name", ["Departure airport", "Where from?", "Return to city", "Search",
                                  "Return flight number", "Departure city", "Marketing code"])
def test_other_fields_take_a_word(monkeypatch, name):
    field = Element(ref="e93", role="combobox", name=name, editable=True)
    assert _helper_text(monkeypatch, field, '{"text": "BKK"}') == "BKK"
