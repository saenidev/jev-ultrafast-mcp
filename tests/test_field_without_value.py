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
    history = [{"op": "type", "ref": "e2", "target": "Notes", "ok": False,
                "error": policy.NO_VALUE_ERROR, "where": "https://example.test/"}]

    _, heads = policy._operation_heads(_observation([name, notes]))
    policy.withdraw_valueless(heads, history, "https://example.test/")

    assert [e.ref for e in heads["TYPE_TEXT"]] == ["e1"]


def test_a_refusal_on_another_page_withdraws_nothing_here():
    """Refs restart per document: e1 on page A is not e1 on page B (review P2-2)."""
    email = Element(ref="e1", role="textbox", name="Email", editable=True)
    history = [{"op": "type", "ref": "e1", "target": "Notes", "ok": False,
                "error": policy.NO_VALUE_ERROR, "where": "https://example.test/a"}]

    _, heads = policy._operation_heads(_observation([email]))
    policy.withdraw_valueless(heads, history, "https://example.test/")

    assert [e.ref for e in heads["TYPE_TEXT"]] == ["e1"]


def test_a_refusal_from_an_earlier_goal_withdraws_nothing():
    """Only the running goal stamps `where`; a previous goal's refusal carries none (P2-2)."""
    notes = Element(ref="e2", role="textbox", name="Notes", editable=True)
    history = [{"op": "type", "ref": "e2", "target": "Notes", "ok": False,
                "error": policy.NO_VALUE_ERROR}]

    _, heads = policy._operation_heads(_observation([notes]))
    policy.withdraw_valueless(heads, history, "https://example.test/")

    assert [e.ref for e in heads["TYPE_TEXT"]] == ["e2"]


def test_the_same_ref_with_another_name_is_another_field():
    other = Element(ref="e2", role="textbox", name="Company", editable=True)
    history = [{"op": "type", "ref": "e2", "target": "Notes", "ok": False,
                "error": policy.NO_VALUE_ERROR, "where": "https://example.test/"}]

    _, heads = policy._operation_heads(_observation([other]))
    policy.withdraw_valueless(heads, history, "https://example.test/")

    assert [e.ref for e in heads["TYPE_TEXT"]] == ["e2"]


def test_withdrawing_the_last_typeable_field_removes_the_operation():
    notes = Element(ref="e2", role="textbox", name="Notes", editable=True)
    order = Element(ref="e3", role="button", name="Order")
    history = [{"op": "type", "ref": "e2", "target": "Notes", "ok": False,
                "error": policy.NO_VALUE_ERROR, "where": "https://example.test/"}]

    _, heads = policy._operation_heads(_observation([notes, order]))
    policy.withdraw_valueless(heads, history, "https://example.test/")

    assert "TYPE_TEXT" not in heads, "offering it again only buys the same refusal"
    assert "e3" in [e.ref for e in heads["CLICK"]]


def test_submit_is_offered_only_on_a_filled_non_secret_field():
    filled = Element(ref="e1", role="searchbox", name="Search", editable=True, value="browser-use")
    empty = Element(ref="e2", role="textbox", name="Name", editable=True)
    secret = Element(ref="e3", role="textbox", name="Password", editable=True, value="x", secret=True)

    assert "SUBMIT" in filled.target_kinds()
    assert "SUBMIT" not in empty.target_kinds()
    assert "SUBMIT" not in secret.target_kinds()


def test_submit_is_not_offered_where_enter_is_a_newline():
    """A textarea/contenteditable: Enter adds a line, or sends a message nobody composed (P1-2)."""
    box = Element(ref="e1", role="textbox", name="Comment", editable=True, value="hi", multiline=True)
    assert "SUBMIT" not in box.target_kinds()
    assert "TYPE_TEXT" in box.target_kinds()


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

    def fake_choose(_cfg, _obs, _goal, history, **_kw):
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


def test_submit_presses_enter_without_retyping(monkeypatch):
    search = Element(ref="e1", role="searchbox", name="Search", editable=True, value="browser-use")
    tab = _Tab(_observation([search]))

    def text_for(*_a):
        raise AssertionError("SUBMIT must not ask the text helper for anything")

    decisions = [
        {"operation": "SUBMIT", "ref": "e1", "target": "Search", "confidence": 0.9},
        {"operation": "DONE", "ref": None, "confidence": 0.9},
    ]
    out, _ = _drive(monkeypatch, tab, decisions, text_for)

    assert tab.acts == [[{"op": "type", "ref": "e1", "text": "", "clear": False, "submit": True}]], (
        "SUBMIT presses Enter on the field as it is; retyping the observed (truncated) value "
        "would replace the real content")
    assert "status: done" in out, out


def test_the_rules_say_when_to_submit_instead_of_picking_a_suggestion():
    """Measured on GitHub: with only "select the matching suggestion" to go on, the model clicked
    a suggestion *list* that held no match until the stall detector ended the goal."""
    rules = " ".join(policy.NEXT_ACTION.split())
    assert "SUBMIT the filled field" in rules
    assert "autocomplete suggestion selected" in rules, "the combobox rule itself must stay"



def test_choose_does_not_offer_typing_once_every_field_is_valueless(monkeypatch):
    """Through `choose` itself, so dropping its withdraw call or its recompute is caught (M4/M5)."""
    notes = Element(ref="e2", role="textbox", name="Notes", editable=True)
    order = Element(ref="e3", role="button", name="Order")
    sent: list[dict] = []

    def fake_post(url, key, body):
        sent.append(body)
        ids = list(body["questions"]["operation"]["criteria"])
        probabilities = {name: (1.0 if name == "DONE" else 0.0) for name in ids}
        return {"answers": {"operation": {"choice": "DONE", "confidence": 1.0,
                                          "probabilities": probabilities}}}

    monkeypatch.setattr(policy, "_post", fake_post)
    cfg = dataclasses.replace(Config.from_env(), typesafe_key="k")
    history = [{"op": "type", "ref": "e2", "target": "Notes", "ok": False,
                "error": policy.NO_VALUE_ERROR, "where": "https://example.test/"}]
    policy.choose(cfg, _observation([notes, order]), "order a pizza", history)

    questions = sent[0]["questions"]
    assert "TYPE_TEXT" not in questions["operation"]["criteria"]
    assert "type_text_target" not in questions
    assert "CLICK" in questions["operation"]["criteria"]


def test_a_new_goal_does_not_see_the_previous_goals_steps(monkeypatch):
    """The model is told not to repeat satisfied steps; last goal's "Add to cart" is not this one's."""
    add = Element(ref="e3", role="button", name="Add to cart")
    tab = _Tab(_observation([add]))
    from jev_ultrafast_mcp.browser import Step
    tab.history.append(Step(op="click", ok=True, ref="e3", target="Add to cart", ms=1))

    decisions = [{"operation": "DONE", "ref": None, "confidence": 0.9}]
    _out, seen = _drive(monkeypatch, tab, decisions, lambda *_a: "x")

    assert seen[0] == [], f"a fresh goal must start with an empty history, got {seen[0]}"


def test_a_refused_enter_withdraws_submit_and_the_goal_goes_on(monkeypatch):
    search = Element(ref="e1", role="searchbox", name="Search", editable=True, value="help")
    go = Element(ref="e2", role="button", name="Go")
    tab = _Tab(_observation([search, go]))
    from jev_ultrafast_mcp.browser import Step

    def act(ops, **_kw):
        tab.acts.append(ops)
        refused = ops[0].get("submit")
        step = Step(op="type" if refused else ops[0]["op"], ok=not refused, ref=ops[0].get("ref"),
                    target="Search" if refused else "Go", ms=1,
                    error="needs_confirmation" if refused else None,
                    detail="Enter would submit 'Cancel subscription'" if refused else None)
        tab.history.append(step)
        return {"ops": [step.to_dict()], "ok": not refused, "steps": len(tab.acts), "page_changed": False}
    tab.act = act

    decisions = [
        {"operation": "SUBMIT", "ref": "e1", "target": "Search", "confidence": 0.9},
        {"operation": "CLICK", "ref": "e2", "target": "Go", "confidence": 0.9},
        {"operation": "DONE", "ref": None, "confidence": 0.9},
    ]
    out, seen = _drive(monkeypatch, tab, decisions, lambda *_a: "x")

    assert "status: done" in out, out
    assert "Enter withdrawn" in out, out
    refused = seen[1][-1]
    assert refused.get("error") == "needs_confirmation" and refused.get("where"), refused
    heads = policy.withdraw_refused_submit({"SUBMIT": [search]}, seen[1], tab.last.url)
    assert "SUBMIT" not in heads, "the refused field must stop being offered for SUBMIT here"
    elsewhere = policy.withdraw_refused_submit({"SUBMIT": [search]}, seen[1], "https://other.example/")
    assert elsewhere["SUBMIT"] == [search], "and only on the page where it was refused"


def test_pages_left_behind_are_remembered_briefly():
    from jev_ultrafast_mcp.server import PAGE_NOTES, _note_page_left

    def obs(url, text):
        o = _observation([])
        o.url, o.text, o.title = url, text, url
        return o

    notes: list = []
    _note_page_left(notes, obs("https://a/", "see reference ticket T-3317"), obs("https://b/", ""))
    assert notes and "T-3317" in notes[0]["text"]
    _note_page_left(notes, obs("https://b/", "b"), obs("https://b/", "b"))
    assert len(notes) == 1, "an action that stays on the page is not a page left"
    for i in range(PAGE_NOTES + 3):
        _note_page_left(notes, obs(f"https://p{i}/", "x" * 5000), obs("https://z/", ""))
    assert len(notes) == PAGE_NOTES and all(len(n["text"]) <= 700 for n in notes)


def test_earlier_pages_reach_the_decision_request(monkeypatch):
    sent = []

    def fake_post(url, key, body):
        sent.append(body)
        ids = list(body["questions"]["operation"]["criteria"])
        return {"answers": {"operation": {"choice": "DONE", "confidence": 1.0,
                                          "probabilities": {n: float(n == "DONE") for n in ids}}}}

    monkeypatch.setattr(policy, "_post", fake_post)
    cfg = _text_cfg()
    cfg.typesafe_key = "k"
    button = Element(ref="e1", role="button", name="Close ticket")
    notes = [{"url": "https://kb/", "title": "Resetting", "text": "reference ticket T-3317"}]
    policy.choose(cfg, _observation([button]), "close it", [], pages_seen=notes)
    assert sent[0]["state"]["earlier_pages"] == notes
    policy.choose(cfg, _observation([button]), "close it", [])
    assert "earlier_pages" not in sent[1]["state"], "no key at all when nothing was left behind"


def _post_with(probabilities, target_ref="e1"):
    def fake_post(url, key, body):
        ids = list(body["questions"]["operation"]["criteria"])
        probs = {n: probabilities.get(n, 0.0) for n in ids}
        top = max(probs, key=probs.get)
        answers = {"operation": {"choice": top, "confidence": probs[top], "probabilities": probs}}
        for qid, q in body["questions"].items():
            if qid.endswith("_target"):
                refs = list(q["criteria"])
                answers[qid] = {"choice": refs[0], "confidence": 1.0,
                                "probabilities": {r: float(r == refs[0]) for r in refs}}
        return {"answers": answers}
    return fake_post


def _choose_with(monkeypatch, probabilities, second_chance=True):
    monkeypatch.setattr(policy, "_post", _post_with(probabilities))
    cfg = _text_cfg()
    cfg.typesafe_key = "k"
    button = Element(ref="e1", role="button", name="Accept all cookies")
    return policy.choose(cfg, _observation([button]), "buy a kettle", [], second_chance=second_chance)


def test_a_weak_blocked_beside_a_near_tied_action_takes_the_action(monkeypatch):
    d = _choose_with(monkeypatch, {"BLOCKED": 0.37, "CLICK": 0.32, "SCROLL": 0.2, "WAIT": 0.11})
    assert d["operation"] == "CLICK" and d["ref"] == "e1" and d["overrode"] == "BLOCKED", d


def test_a_confident_blocked_stands(monkeypatch):
    d = _choose_with(monkeypatch, {"BLOCKED": 0.62, "CLICK": 0.30, "SCROLL": 0.08})
    assert d["operation"] == "BLOCKED" and "overrode" not in d, d


def test_a_weak_blocked_far_ahead_of_any_action_stands(monkeypatch):
    d = _choose_with(monkeypatch, {"BLOCKED": 0.45, "CLICK": 0.10, "SCROLL": 0.25, "WAIT": 0.20})
    assert d["operation"] == "BLOCKED", "scrolling or waiting is not an action taken over BLOCKED"


def test_no_override_without_a_second_chance(monkeypatch):
    d = _choose_with(monkeypatch, {"BLOCKED": 0.37, "CLICK": 0.32, "SCROLL": 0.2, "WAIT": 0.11},
                     second_chance=False)
    assert d["operation"] == "BLOCKED", d


def test_the_goal_loop_bounds_the_overrides(monkeypatch):
    """Three near-ties become actions; the fourth BLOCKED is final."""
    from jev_ultrafast_mcp.server import WEAK_BLOCKED_OVERRIDES
    button = Element(ref="e1", role="button", name="Accept all cookies")
    tab = _Tab(_observation([button]))
    chances = []

    def fake_choose(_cfg, _obs, _goal, _history, second_chance=False, **_kw):
        chances.append(second_chance)
        if second_chance:
            return {"operation": "CLICK", "ref": "e1", "target": "Accept all cookies",
                    "confidence": 0.32, "overrode": "BLOCKED",
                    "probabilities": {"BLOCKED": 0.37, "CLICK": 0.32}}
        return {"operation": "BLOCKED", "ref": None, "confidence": 0.37}

    monkeypatch.setattr(server.policy, "available", lambda _cfg: True)
    monkeypatch.setattr(server.policy, "choose", fake_choose)
    monkeypatch.setattr(server, "_session", lambda _name: tab)
    out = server.browser_goal("buy a kettle", max_steps=10, verbose=True)
    assert chances[:WEAK_BLOCKED_OVERRIDES] == [True] * WEAK_BLOCKED_OVERRIDES
    assert chances[WEAK_BLOCKED_OVERRIDES] is False, chances
    assert out.count("taking the action") == WEAK_BLOCKED_OVERRIDES, out
    assert "status: blocked" in out, out


def test_the_same_query_is_not_offered_for_submit_again():
    """After a search the results page still shows the query; sending it again changes nothing."""
    box = Element(ref="e3", role="textbox", name="Search products", editable=True, value="Blue Kettle")
    history = [{"op": "type", "ref": "e7", "target": "Search products", "ok": True,
                "submitted": "Blue Kettle"}]
    heads = policy.withdraw_resubmit({"SUBMIT": [box], "CLICK": [box]}, history)
    assert "SUBMIT" not in heads, "the query this goal already sent must not be offered again"
    assert heads["CLICK"] == [box], "only SUBMIT is withdrawn"


def test_a_changed_query_or_another_field_is_still_offered():
    history = [{"op": "type", "ref": "e7", "target": "Search products", "ok": True,
                "submitted": "Blue Kettle"}]
    changed = Element(ref="e3", role="textbox", name="Search products", editable=True, value="Oak Board")
    other = Element(ref="e4", role="textbox", name="Postcode", editable=True, value="Blue Kettle")
    heads = policy.withdraw_resubmit({"SUBMIT": [changed, other]}, history)
    assert heads["SUBMIT"] == [changed, other]
    failed = [{**history[0], "ok": False}]
    same = Element(ref="e3", role="textbox", name="Search products", editable=True, value="Blue Kettle")
    assert policy.withdraw_resubmit({"SUBMIT": [same]}, failed)["SUBMIT"] == [same], (
        "a submit that did not go through may be retried")


def test_the_goal_loop_records_what_each_submit_sent(monkeypatch):
    search = Element(ref="e1", role="searchbox", name="Search", editable=True, value="browser-use")
    tab = _Tab(_observation([search]))
    decisions = [
        {"operation": "SUBMIT", "ref": "e1", "target": "Search", "confidence": 0.9},
        {"operation": "DONE", "ref": None, "confidence": 0.9},
    ]
    _out, seen = _drive(monkeypatch, tab, decisions, lambda *_a: "x")
    assert seen[1][-1].get("submitted") == "browser-use", seen[1]


def test_every_completed_step_reaches_the_decision_with_its_page(monkeypatch):
    """`history` holds ten steps; a checkout is twenty-five, and the first ones say what was bought."""
    add = Element(ref="e3", role="button", name="Add to cart")
    tab = _Tab(dataclasses.replace(_observation([add]), title="Blue Kettle"))
    decisions = [{"operation": "CLICK", "ref": "e3", "target": "Add to cart", "confidence": 0.9}] * 12 + [
        {"operation": "DONE", "ref": None, "confidence": 0.9}]
    seen_done: list = []

    def fake_choose(_cfg, _obs, _goal, history, **kw):
        seen_done.append(kw.get("done_so_far"))
        return decisions.pop(0)

    monkeypatch.setattr(server.policy, "available", lambda _cfg: True)
    monkeypatch.setattr(server.policy, "choose", fake_choose)
    monkeypatch.setattr(server, "_session", lambda _name: tab)
    server.browser_goal("buy things", max_steps=20)

    last = seen_done[-1]
    assert len(last) == 12, f"all twelve completed steps, not the last ten: {len(last)}"
    assert last[0] == "click Add to cart (on: Blue Kettle)", last[0]
    assert seen_done[0] in (None, []), "a fresh goal has done nothing yet"


def test_done_so_far_is_sent_in_the_state(monkeypatch):
    sent = []
    answer = _post_with({"DONE": 1.0})
    monkeypatch.setattr(policy, "_post", lambda _u, _k, body: sent.append(body) or answer(None, None, body))
    cfg = dataclasses.replace(_text_cfg(), typesafe_key="k")
    obs = _observation([Element(ref="e1", role="button", name="Go")])
    policy.choose(cfg, obs, "g", [], done_so_far=["click Add to cart (on: Blue Kettle)"])
    assert sent[0]["state"]["done_so_far"] == ["click Add to cart (on: Blue Kettle)"]
