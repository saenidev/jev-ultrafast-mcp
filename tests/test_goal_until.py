"""`browser_goal(until=...)`: a subgoal ends on deterministic checks, not on a paid DONE decision.

Measured on live Google Flights (2026-09-26): with the planner splitting a task into subgoals, about
40% of all Jev requests were the final DONE after each subgoal -- a request whose only job was to
notice that the page already showed what the planner's own checks would test a moment later. With
`until`, the loop runs those checks on the observation it already reads after every action and stops
as soon as they pass, so the DONE request is never made.

Rules pinned here:
  * checks are evaluated once before the first decision (reported, never a reason to stop: the
    goal may still have to do something) and after every executed action;
  * the goal stops the moment they pass after at least one action, with status `done` and the line
    `until: met after N steps`;
  * no extra page reads: the observation the loop already holds is the one checked;
  * `js` checks never run here, whatever JEVMCP_ALLOW_JS says;
  * `until=None` is the old behaviour exactly.

No network, no browser: the session and the decision model are scripted.
"""

from __future__ import annotations

import dataclasses

import pytest

from jev_ultrafast_mcp import server
from jev_ultrafast_mcp.observe import Element, Observation


def _page(text: str, url: str = "https://example.test/") -> Observation:
    return Observation(
        url=url, title="Example", text=text,
        elements=[Element(ref="e1", role="button", name="Next")],
        digest=text, text_digest=text, page_key="k", scroll={"y": 0}, reachable=1,
        omitted=0, overlays=[], cross_frames=0, cross_frame_srcs=[],
    )


class _Session:
    """A session whose page is a script: each action moves to the next observation."""

    def __init__(self, pages: list[Observation]):
        self.pages = list(pages)
        self.last: Observation | None = None
        self.history: list = []
        self.acts: list = []
        self.observes = 0
        self.js = 0

    def navigate(self, url: str) -> None:
        pass

    def reset_progress(self) -> None:
        pass

    def page_is_idle(self) -> bool:
        return True

    def observe(self, **_kwargs) -> Observation:
        self.observes += 1
        self.last = self.pages[0]
        return self.last

    def evaluate_js(self, _expression: str):
        self.js += 1
        return True

    def act(self, ops, **_kwargs) -> dict:
        from jev_ultrafast_mcp.browser import Step
        self.acts.append(ops)
        if len(self.pages) > 1:
            self.pages.pop(0)
        # The real session reads the page after acting and keeps it as `last`.
        self.last = self.pages[0]
        step = Step(op=ops[0]["op"], ok=True, ref=ops[0].get("ref"), ms=1)
        self.history.append(step)
        return {"ops": [step.to_dict()], "ok": True, "steps": len(self.history),
                "page_changed": True}


def _click() -> dict:
    return {"operation": "CLICK", "ref": "e1", "target": "Next", "confidence": 0.9, "latency_ms": 5}


def _run(monkeypatch, session: _Session, decisions: list[dict], **kwargs):
    asked: list[Observation] = []

    def fake_choose(_cfg, observation, _goal, _history, **_kw):
        asked.append(observation)
        if not decisions:
            return {"operation": "DONE", "ref": None, "confidence": 1.0, "latency_ms": 5}
        return decisions.pop(0)

    monkeypatch.setattr(server.policy, "available", lambda _cfg: True)
    monkeypatch.setattr(server.policy, "choose", fake_choose)
    monkeypatch.setattr(server, "_session", lambda _name: session)
    # `_first_read` polls until the page settles; one read is enough for a scripted page.
    monkeypatch.setattr(server, "_first_read", lambda tab: tab.observe())
    out = server.browser_goal("go to the results", max_steps=10, verbose=True, **kwargs)
    return out, asked


RESULTS = [{"type": "text_contains", "text": "Results ready"}]


def test_until_ends_the_goal_without_asking_the_model_for_done(monkeypatch):
    session = _Session([_page("Search form"), _page("Loading"), _page("Results ready")])

    out, asked = _run(monkeypatch, session, [_click(), _click(), _click()], until=RESULTS)

    assert "status: done" in out, out
    assert "until: met after 2 steps" in out, out
    assert len(asked) == 2, f"{len(asked)} decisions; the DONE request should never be made"
    assert len(session.acts) == 2
    assert "turbo: 2 decisions" in out, out


def test_until_already_true_still_lets_the_model_act_once(monkeypatch):
    """A page that already shows the proof may still need the step (a filter to re-apply)."""
    session = _Session([_page("Results ready"), _page("Results ready, sorted")])

    out, asked = _run(monkeypatch, session, [_click(), _click()], until=RESULTS)

    assert len(asked) == 1, "the model is asked for the first step even though until held"
    assert len(session.acts) == 1
    assert "status: done" in out and "until: met after 1 step" in out, out
    assert "true before the first step" in out, out


def test_until_already_true_and_the_model_says_done_is_done(monkeypatch):
    session = _Session([_page("Results ready")])

    out, asked = _run(monkeypatch, session, [], until=RESULTS)

    assert "status: done" in out, out
    assert len(asked) == 1 and not session.acts
    assert "until: met after 0 steps" in out, out


def test_until_never_met_is_reported(monkeypatch):
    session = _Session([_page("Search form"), _page("Still a form")])

    out, _asked = _run(monkeypatch, session, [_click()], until=RESULTS)

    assert "until: not met" in out, out
    assert "text_contains" in out, out


def test_until_reads_no_extra_pages(monkeypatch):
    pages = [_page("Search form"), _page("Loading"), _page("Loading more"), _page("x")]
    plain = _Session(pages)
    _run(monkeypatch, plain, [_click(), _click(), _click()])
    checked = _Session(pages)
    _run(monkeypatch, checked, [_click(), _click(), _click()],
         until=[{"type": "text_contains", "text": "never shown"}])

    assert checked.observes == plain.observes, (checked.observes, plain.observes)


def test_until_never_runs_js(monkeypatch):
    monkeypatch.setattr(server, "CONFIG", dataclasses.replace(server.CONFIG, allow_js=True))
    session = _Session([_page("Search form"), _page("Results ready")])

    out, _asked = _run(monkeypatch, session, [_click(), _click()],
                       until=[{"type": "js", "expr": "true"}])

    assert session.js == 0, "a js check must not be evaluated on the until path"
    assert "until: met" not in out, out


def test_without_until_the_goal_behaves_as_before(monkeypatch):
    session = _Session([_page("Search form"), _page("Results ready")])

    out, asked = _run(monkeypatch, session, [_click()])

    assert "status: done" in out
    assert len(asked) == 2, "without until the model's DONE is what ends the goal"
    assert "until:" not in out, out


@pytest.mark.parametrize("empty", [[], None])
def test_an_empty_until_is_no_until(monkeypatch, empty):
    session = _Session([_page("Search form"), _page("Results ready")])

    out, asked = _run(monkeypatch, session, [_click()], until=empty)

    assert len(asked) == 2 and "until:" not in out, out


def test_until_is_documented_in_the_tool():
    doc = server.browser_goal.__doc__ or ""
    assert "until" in doc and "until: met after" in doc


def _popup_page(text: str, *, options: bool = False, dialog: bool = False) -> Observation:
    page = _page(text)
    elements = list(page.elements)
    if options:
        elements.append(Element(ref="e9", role="option", name="Suvarnabhumi Airport (BKK)"))
    overlays = [{"role": "dialog", "name": "Departure calendar", "modal": True}] if dialog else []
    return dataclasses.replace(page, elements=elements, overlays=overlays)


def test_until_waits_while_the_action_left_suggestions_open(monkeypatch):
    """Measured on Google Flights: `field_shows Where from? BKK` held right after typing BKK,
    with the suggestion list still open, so the subgoal ended before the suggestion was clicked
    and the next subgoal started inside the popup."""
    session = _Session([_page("Form"), _popup_page("Typed BKK", options=True), _page("Typed BKK")])

    out, asked = _run(monkeypatch, session, [_click(), _click(), _click()],
                      until=[{"type": "text_contains", "text": "Typed BKK"}])

    assert "until: met after 2 steps" in out, out
    assert len(session.acts) == 2


def test_until_waits_while_the_action_left_a_new_dialog_open(monkeypatch):
    """A date typed into a calendar dialog is not set until the dialog is closed (Done)."""
    session = _Session([_page("Form"), _popup_page("Oct 15", dialog=True), _page("Oct 15")])

    out, asked = _run(monkeypatch, session, [_click(), _click(), _click()],
                      until=[{"type": "text_contains", "text": "Oct 15"}])

    assert "until: met after 2 steps" in out, out


def test_until_accepts_a_dialog_that_was_already_open_when_the_goal_began(monkeypatch):
    """A subgoal that works inside an already open dialog may end with it still open."""
    session = _Session([_popup_page("Form", dialog=True), _popup_page("Oct 15", dialog=True)])

    out, asked = _run(monkeypatch, session, [_click(), _click()],
                      until=[{"type": "text_contains", "text": "Oct 15"}])

    assert "until: met after 1 step" in out, out
