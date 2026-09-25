"""Four failures measured on a live flight-search form, pinned without the site.

(A) Speed. Every TYPE_TEXT waited 1.5-2.5 s on the text helper even when the goal spelled the value
    out ("Where from: type Bangkok and click the suggestion ..."). An explicit, unambiguous
    `type X` for the field is now read straight from the goal; anything less clear still goes to
    the helper.
(B) Retype loop. After typing into an autocomplete combobox the page showed the suggestions, and
    the model typed into the same field again four or five times instead of clicking one. A field
    just typed into is not offered for typing while its suggestions are showing, and typing the
    same field twice with suggestions on screen counts as stalled.
(C) Wrong-field typing. A field that already holds the value the goal names for it is not typed
    into again.
(D) Unguarded removal. "Remove flight from Bangkok to Seoul on Sun, Oct 11" was clicked with no
    confirmation because its name matched the goal's words.

No network: `_post` is replaced in every test.
"""

from __future__ import annotations

import dataclasses

import pytest

from jev_ultrafast_mcp import policy
from jev_ultrafast_mcp.config import Config
from jev_ultrafast_mcp.observe import Element, Observation
from jev_ultrafast_mcp.safety import confirm_reason


def _observation(elements: list[Element]) -> Observation:
    return Observation(
        url="https://example.test/", title="Flights", text="Flights", elements=elements,
        digest="d", text_digest="t", page_key="k", scroll={"y": 0}, reachable=len(elements),
        omitted=0, overlays=[], cross_frames=0, cross_frame_srcs=[],
    )


def _cfg() -> Config:
    return dataclasses.replace(Config.from_env(), typesafe_key="k", text_model_key="test-key",
                               text_model="test-model")


def _field(name: str, value: str = "", ref: str = "e1", role: str = "combobox", **kw) -> Element:
    return Element(ref=ref, role=role, name=name, value=value, editable=True, **kw)


class _Helper:
    """A text helper that records whether it was asked."""

    def __init__(self, answer: str = '{"text": "FROM-HELPER"}'):
        self.calls = 0
        self.answer = answer

    def __call__(self, url, key, body):
        self.calls += 1
        return {"choices": [{"message": {"content": self.answer}}]}


def _text(monkeypatch, goal: str, field: Element, helper: _Helper | None = None) -> tuple[str, _Helper]:
    helper = helper or _Helper()
    monkeypatch.setattr(policy, "_post", helper)
    return policy.text_for(_cfg(), goal, field, _observation([field]), []), helper


# ------------------------------------------------------------------ (A) literal values

TWO_FIELDS = ("Where from: type Bangkok and click the suggestion Suvarnabhumi Airport (BKK). "
              "Where to: type Seoul and click the suggestion Incheon International Airport (ICN).")


@pytest.mark.parametrize("goal, name, expected", [
    ("Where from: type Bangkok and click the suggestion Suvarnabhumi Airport (BKK)",
     "Where from?", "Bangkok"),
    (TWO_FIELDS, "Where from?", "Bangkok"),
    (TWO_FIELDS, "Where to?", "Seoul"),
    ('Search: type "noise cancelling headphones" and press Enter', "Search", "noise cancelling headphones"),
    ("type 'Seoul' into the Where to field, then pick Incheon", "Where to?", "Seoul"),
    ("Open the page, then type \u201cNew York\u201d into the Where to box", "Where to?", "New York"),
    ("type 'Python' and press Enter", "Search Wikipedia", "Python"),
    ("In the Email field enter jun@example.com and press Continue", "Email", "jun@example.com"),
])
def test_an_explicit_value_for_the_field_skips_the_helper(monkeypatch, goal, name, expected):
    value, helper = _text(monkeypatch, goal, _field(name))
    assert value == expected
    assert helper.calls == 0, "an unambiguous value in the goal must not wait on the text helper"


@pytest.mark.parametrize("goal, name", [
    # Two values for two fields, and this field is named by neither.
    (TWO_FIELDS, "Destination"),
    # The only instruction names another field: that value is not this field's.
    ("Where to: type Tokyo and click the suggestion Haneda", "Where from?"),
    # "type" and "enter" as nouns or keys, not instructions.
    ("Set the trip type to One way", "Trip type"),
    ("Enter the departure date and press Enter", "Departure"),
    ("press Enter in the search box", "Search"),
    # Two different values for the same field: no way to tell which.
    ("Search: type kettle. Search: type toaster.", "Search"),
    # Nothing to read at all.
    ("find flights from Bangkok (BKK) to Seoul (ICN)", "Where from?"),
    # An unquoted value too long to be sure where it ends.
    ("Notes: type please leave the parcel with the neighbour at number twelve", "Notes"),
    # A long clause that happens to contain the field's one word does not name it.
    ("On the results page for kettles and toasters: type kettle", "Kettles"),
    # A field named by a word both instructions share gets neither value.
    (TWO_FIELDS, "Where"),
])
def test_anything_less_clear_still_asks_the_helper(monkeypatch, goal, name):
    value, helper = _text(monkeypatch, goal, _field(name))
    assert helper.calls == 1 and value == "FROM-HELPER"


def test_an_unnamed_value_is_only_taken_where_there_is_one_field_to_take_it(monkeypatch):
    """"type 'Python' and press Enter" names no field; with two fields on the page it could be
    either, so the helper -- which sees the page -- decides."""
    helper = _Helper()
    monkeypatch.setattr(policy, "_post", helper)
    search = _field("Search Wikipedia", role="searchbox")
    language = _field("Language", ref="e2", role="textbox")
    value = policy.text_for(_cfg(), "type 'Python' and press Enter", language,
                            _observation([search, language]), [])
    assert helper.calls == 1 and value == "FROM-HELPER"


def test_the_goals_value_for_another_field_is_not_typed_into_this_one(monkeypatch):
    """(C): the new row auto-filled 'Where from? Osaka KIX'; the destination text went there."""
    helper = _Helper('{"text": "Osaka"}')
    monkeypatch.setattr(policy, "_post", helper)
    origin = _field("Where from?", value="Osaka KIX", ref="e160")
    destination = _field("Where to?", ref="e161")
    with pytest.raises(policy.NoValueForField):
        policy.text_for(_cfg(), "Where to: type Osaka and click the suggestion Kansai (KIX)",
                        origin, _observation([origin, destination]), [])
    assert helper.calls == 1, "a value named for another field is not read literally here"


def test_a_secret_field_never_takes_the_literal_path(monkeypatch):
    for field in (_field("Password", role="textbox", secret=True),
                  _field("One-time code", role="textbox")):
        _value, helper = _text(monkeypatch, 'Password: type "hunter2" and press Sign in', field)
        assert helper.calls == 1, field.name


def test_goal_values_are_read_only_from_imperative_instructions():
    assert policy.literal_values("Where from: type Bangkok and click it") == [("where from", "Bangkok")]
    assert policy.literal_values('type "Bread and Butter" into the search box') == [
        ("search", "Bread and Butter")]
    assert policy.literal_values("type Bread and Butter into the search box") == [("search", None)], (
        "unquoted, there is no telling where the value ends")
    assert policy.literal_values("first type Osaka, then pick KIX") == [("", "Osaka")]
    assert policy.literal_values("type tickets to Paris") == [("", None)]
    assert policy.literal_values("type Bangkok for the origin") == [("", None)]
    assert policy.literal_values("Set the trip type to One way") == []
    assert policy.literal_values("press Enter") == []
    assert policy.literal_values("Enter the departure date") == [("", None)], (
        "an instruction whose value cannot be read is kept, as unreadable, so it still counts")


# ------------------------------------------------------------------ (C) a field already set


def test_a_field_already_holding_the_goals_value_is_not_retyped(monkeypatch):
    """The auto-filled 'Where from? Osaka KIX' row: typing there again is the wrong-field step."""
    helper = _Helper()
    monkeypatch.setattr(policy, "_post", helper)
    field = _field("Where from?", value="Osaka KIX")
    with pytest.raises(policy.NoValueForField) as caught:
        policy.text_for(_cfg(), "Where from: type Osaka and click the suggestion Kansai (KIX)",
                        field, _observation([field]), [])
    assert "already" in str(caught.value)
    assert helper.calls == 0


def test_a_helper_answer_equal_to_the_current_value_is_not_retyped(monkeypatch):
    helper = _Helper('{"text": "Bangkok"}')
    monkeypatch.setattr(policy, "_post", helper)
    field = _field("Where else?", value="bangkok")
    with pytest.raises(policy.NoValueForField):
        policy.text_for(_cfg(), "fly from Bangkok to Seoul", field, _observation([field]), [])


def test_the_text_helper_is_told_to_keep_values_in_their_own_fields():
    rules = " ".join(policy.TEXT_VALUE.split())
    assert "already holds" in rules
    assert "another field" in rules


# ------------------------------------------------------------------ (B) autocomplete


def _suggestions() -> list[Element]:
    return [Element(ref="e140", role="option", name="Suvarnabhumi Airport (BKK)"),
            Element(ref="e141", role="option", name="Bangkok, Thailand")]


def _sent_questions(monkeypatch, elements, history):
    sent: list[dict] = []

    def fake_post(url, key, body):
        sent.append(body)
        ids = list(body["questions"]["operation"]["criteria"])
        return {"answers": {"operation": {"choice": "DONE", "confidence": 1.0,
                                          "probabilities": {n: float(n == "DONE") for n in ids}}}}

    monkeypatch.setattr(policy, "_post", fake_post)
    policy.choose(_cfg(), _observation(elements), "Where from: type Bangkok", history)
    return sent[0]["questions"]


def _typed(ref="e137", name="Where else?", ok=True):
    return {"op": "type", "ref": ref, "target": name, "ok": ok}


def test_a_field_just_typed_into_is_not_retyped_while_its_suggestions_show(monkeypatch):
    where_else = _field("Where else?", value="Bangkok", ref="e137")
    where_to = _field("Where to?", ref="e150")
    questions = _sent_questions(monkeypatch, [where_else, where_to, *_suggestions()], [_typed()])
    assert list(questions["type_text_target"]["criteria"]) == ["e150"]
    assert {"e140", "e141"} <= set(questions["click_target"]["criteria"])


def test_the_same_field_under_a_new_ref_is_still_the_same_field(monkeypatch):
    where_else = _field("Where else?", value="Bangkok", ref="e137")
    where_to = _field("Where to?", ref="e150")
    questions = _sent_questions(monkeypatch, [where_else, where_to, *_suggestions()],
                                [_typed(ref="e136")])
    assert "e137" not in questions["type_text_target"]["criteria"]


def test_when_it_is_the_only_field_typing_is_withdrawn_so_a_suggestion_is_clicked(monkeypatch):
    where_else = _field("Where else?", value="Bangkok", ref="e137")
    questions = _sent_questions(monkeypatch, [where_else, *_suggestions()], [_typed()])
    assert "TYPE_TEXT" not in questions["operation"]["criteria"]
    assert "CLICK" in questions["operation"]["criteria"]
    assert "SUBMIT" in questions["operation"]["criteria"], "Enter stays for a query nothing matches"


def test_without_suggestions_on_screen_typing_is_still_offered(monkeypatch):
    where_else = _field("Where else?", value="Bangkok", ref="e137")
    go = Element(ref="e9", role="button", name="Search")
    questions = _sent_questions(monkeypatch, [where_else, go], [_typed()])
    assert "e137" in questions["type_text_target"]["criteria"]


def test_a_plain_text_box_beside_an_unrelated_list_keeps_typing(monkeypatch):
    """The rule is about autocomplete fields; a notes box next to a static option list is not one."""
    notes = _field("Notes", value="hi", ref="e3", role="textbox")
    questions = _sent_questions(monkeypatch, [notes, *_suggestions()], [_typed(ref="e3", name="Notes")])
    assert "e3" in questions["type_text_target"]["criteria"]


def test_typing_one_autocomplete_field_twice_with_suggestions_showing_is_a_stall(monkeypatch):
    """Threshold 2 here, not 3: the trace retyped e136 four times before the loop gave up."""
    where_else = _field("Where else?", value="Bangkok", ref="e137")
    where_to = _field("Where to?", ref="e150")
    history = [_typed(), {"op": "click", "ref": "e9", "target": "Swap", "ok": True}, _typed()]
    history.append({"op": "click", "ref": "e9", "target": "Swap", "ok": True})
    questions = _sent_questions(monkeypatch, [where_else, where_to, *_suggestions()], history)
    assert "e137" not in questions["type_text_target"]["criteria"]


def test_a_submit_is_not_counted_as_typing(monkeypatch):
    search = _field("Search", value="kettle", ref="e2", role="searchbox")
    other = _field("Postcode", ref="e5", role="textbox")
    history = [{**_typed(ref="e2", name="Search"), "submitted": "kettle"}]
    questions = _sent_questions(monkeypatch, [search, other, *_suggestions()], history)
    assert "e2" in questions["type_text_target"]["criteria"]


def test_other_operations_keep_their_threshold_of_three():
    history = [{"op": "click", "ref": "e8", "ok": True}] * 2
    assert policy.stalled_targets(history) == {}


def test_the_rules_say_to_click_the_suggestion_not_retype():
    rules = " ".join(policy.NEXT_ACTION.split())
    assert "click the suggestion that matches the goal; do not retype" in rules


def test_the_goal_loop_reaches_the_suggestion_instead_of_retyping(monkeypatch):
    """End to end through `browser_goal`, with a model that retypes whenever typing is on offer --
    the measured behaviour. The loop must type once, then click the suggestion."""
    from jev_ultrafast_mcp import server
    from jev_ultrafast_mcp.browser import Step

    where_else = _field("Where else?", ref="e137")
    option = Element(ref="e140", role="option", name="Suvarnabhumi Airport (BKK)")
    observation = _observation([where_else, option])
    typed: list[str] = []

    class Tab:
        history: list = []
        last = observation

        def reset_progress(self): pass
        def page_is_idle(self): return True
        def navigate(self, _url): pass
        def observe(self, **_kw): return observation

        def act(self, ops, **_kw):
            op = ops[0]
            if op["op"] == "type":
                typed.append(op["text"])
                where_else.value = op["text"]
            step = Step(op=op["op"], ok=True, ref=op.get("ref"),
                        target=observation.by_ref[op["ref"]].name, ms=1)
            self.history.append(step)
            return {"ops": [step.to_dict()], "ok": True, "page_changed": True}

    tab = Tab()
    tab.history = []

    def retyping_model(url, key, body):
        ops = list(body["questions"]["operation"]["criteria"])
        if tab.history and tab.history[-1].op == "click":
            choice = "DONE"
        else:
            choice = "TYPE_TEXT" if "TYPE_TEXT" in ops else "CLICK"
        answers = {"operation": {"choice": choice, "confidence": 0.9,
                                 "probabilities": {n: float(n == choice) for n in ops}}}
        for qid, q in body["questions"].items():
            if qid.endswith("_target"):
                refs = list(q["criteria"])
                answers[qid] = {"choice": refs[0], "confidence": 1.0,
                                "probabilities": {r: float(r == refs[0]) for r in refs}}
        return {"answers": answers}

    def no_helper(*_a):
        raise AssertionError("the goal spells the value out; the text helper must not be asked")

    monkeypatch.setattr(server, "CONFIG", _cfg())
    monkeypatch.setattr(policy, "_post", retyping_model)
    monkeypatch.setattr(policy, "_text_from_helper", no_helper)
    monkeypatch.setattr(server, "_session", lambda _name: tab)
    out = server.browser_goal("Where else: type Bangkok and click the suggestion Suvarnabhumi Airport (BKK)",
                              max_steps=6, verbose=True)

    assert typed == ["Bangkok"], out
    assert [s.op for s in tab.history] == ["type", "click"], out
    assert "status: done" in out, out


# ------------------------------------------------------------------ (D) removal


@pytest.mark.parametrize("name", [
    "Remove flight from Bangkok to Seoul on Sun, Oct 11", "Remove", "Remove from cart",
    "Remove passenger John Smith", "Discard draft", "Discard", "Clear all", "Erase all data",
    "Empty cart", "Move to trash", "\uff32emove flight",
])
def test_removal_needs_confirmation(name):
    assert confirm_reason(Config(), name, "button"), name


@pytest.mark.parametrize("name", [
    "Remove adult", "Remove child", "Remove infant in seat", "Remove infant on lap",
    "Clear", "Clear search", "Archive", "Search", "Departure",
])
def test_count_steppers_and_ordinary_controls_do_not(name):
    assert confirm_reason(Config(), name, "button") is None, name


def test_the_override_never_takes_a_removal(monkeypatch):
    """A weak BLOCKED is not turned into a removal (the override never picks a guarded target)."""
    remove = Element(ref="e1", role="button", name="Remove flight from Bangkok to Seoul on Sun, Oct 11")

    def fake_post(url, key, body):
        ids = list(body["questions"]["operation"]["criteria"])
        probs = {n: {"BLOCKED": 0.42, "CLICK": 0.40}.get(n, 0.0) for n in ids}
        spare = 1 - sum(probs.values())
        probs["WAIT"] = probs.get("WAIT", 0) + spare
        answers = {"operation": {"choice": "BLOCKED", "confidence": 0.42, "probabilities": probs}}
        answers["click_target"] = {"choice": "e1", "confidence": 1.0, "probabilities": {"e1": 1.0}}
        return {"answers": answers}

    monkeypatch.setattr(policy, "_post", fake_post)
    decision = policy.choose(_cfg(), _observation([remove]), "set the departure date", [],
                             second_chance=True)
    assert decision["operation"] == "BLOCKED" and decision["ref"] is None


def test_the_target_rules_prefer_the_field_over_a_control_that_names_it():
    rules = " ".join(policy.TARGET_RULES.split())
    assert "merely mentions the goal's words" in rules
