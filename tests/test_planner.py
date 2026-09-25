"""browser_task: a planner above Jev, pinned against a real page with both models faked.

The planner (`planner._http`) and the decision model (`policy._post`) are replaced; the browser, the
observer, `browser_goal`'s loop, the confirmation rail and the assertions are all real. So these
check what the task loop actually does to a page, not a fake tab's record of it.

(a) checks are enforced: a subgoal Jev calls DONE whose checks fail is a failure, and the replan
    request carries the failed check.
(b) three failed subgoals in a row stop the task.
(c) a malformed planner answer is retried once, then reported as an error.
(d) the planner has no way around the confirmation rail.
(e) the time split adds up.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from urllib.parse import quote

import pytest

from jev_ultrafast_mcp import assertions, planner, policy, server
from jev_ultrafast_mcp.browser import BrowserManager
from jev_ultrafast_mcp.config import Config, find_chrome
from jev_ultrafast_mcp.observe import Element, Observation


def _chrome_available() -> bool:
    try:
        find_chrome(None)
        return True
    except RuntimeError:
        return False


needs_chrome = pytest.mark.skipif(not _chrome_available(), reason="needs Chrome")

WIZARD = """<!doctype html><title>Signup</title>
<div id=one><h1>Step 1 of 2</h1><label>Full name <input id=fullname></label>
<button onclick="one.hidden=true;two.hidden=false">Next</button></div>
<div id=two hidden><h1>Step 2 of 2</h1><label>City <input id=city></label>
<button onclick="two.hidden=true;done.hidden=false;
  done.textContent='Thanks '+fullname.value+' from '+city.value">Finish</button></div>
<p id=done hidden></p>"""

DELETE = """<!doctype html><title>Account</title>
<h1>Your account</h1><button onclick="document.title='DELETED'">Delete account</button>
<button>Save</button>"""


# ------------------------------------------------------------------ fakes

class FakeJev:
    """A decision model that follows a script: goal text -> [(operation, target words), ...].

    Step i of a goal (counted from the goal's own recent_actions) runs the i-th scripted action;
    past the end it answers DONE. Targets are found by name among the offered criteria.
    """

    def __init__(self, script: dict[str, list[tuple[str, str]]], delay: float = 0.0):
        self.script = script
        self.calls = 0
        self.delay = delay
        self.goals: list[str] = []

    def __call__(self, url, key, body):
        if "operation" not in body.get("questions", {}):
            raise AssertionError(f"unexpected question {list(body['questions'])}")
        self.calls += 1
        time.sleep(self.delay)
        goal = body["questions"]["operation"]["instructions"]["goal"]
        self.goals.append(goal)
        done = len(body["state"].get("recent_actions") or [])
        actions = next((v for k, v in self.script.items() if k in goal), [])
        ops = list(body["questions"]["operation"]["criteria"])
        choice, target = ("DONE", None) if done >= len(actions) else actions[done]
        answers = {"operation": {"choice": choice, "confidence": 0.9,
                                 "probabilities": {n: float(n == choice) for n in ops}}}
        for qid, question in body["questions"].items():
            if not qid.endswith("_target"):
                continue
            refs = list(question["criteria"])
            pick = refs[0]
            if target and qid == f"{choice.lower()}_target":
                pick = next((r for r in refs if target.lower() in question["criteria"][r]["element"].lower()),
                            refs[0])
            answers[qid] = {"choice": pick, "confidence": 1.0,
                            "probabilities": {r: float(r == pick) for r in refs}}
        return {"answers": answers}


def plan(*subgoals, done=False, ans="", note=""):
    return {"done": done, "ans": ans, "note": note,
            "sg": [{"g": g, "c": c, "n": n} for g, c, n in subgoals]}


class FakePlanner:
    """Answers with the next queued plan; records every prompt it was sent."""

    def __init__(self, answers, delay: float = 0.0):
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.bodies: list[dict] = []
        self.delay = delay

    def __call__(self, cfg, body):
        self.bodies.append(body)
        self.prompts.append(body["messages"][0]["content"])
        time.sleep(self.delay)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, str):
            return {"content": [{"type": "text", "text": answer}], "stop_reason": "end_turn"}
        return {"content": [{"type": "text", "text": json.dumps(answer)}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 50}}


@pytest.fixture(scope="module")
def manager():
    m = BrowserManager(Config(headless=True))
    try:
        yield m
    finally:
        m.shutdown()


@pytest.fixture
def wired(monkeypatch, manager):
    cfg = dataclasses.replace(Config.from_env(), typesafe_key="test-key", planner_key="planner-key",
                              planner_base_url="http://planner.invalid", settle_timeout=0.3)
    monkeypatch.setattr(server, "CONFIG", cfg)
    monkeypatch.setattr(server, "MANAGER", manager)
    monkeypatch.setattr(planner, "CHECK_SETTLE_S", 0.2)

    def no_helper(*_a, **_k):
        raise AssertionError("planner goals spell values out; the text helper must not be asked")
    monkeypatch.setattr(policy, "_text_from_helper", no_helper)

    def install(jev: FakeJev, fake_planner: FakePlanner):
        monkeypatch.setattr(policy, "_post", jev)
        monkeypatch.setattr(planner, "_http", fake_planner)
    return install


def _page(html: str) -> str:
    return "data:text/html," + quote(html)


# ------------------------------------------------------------------ (a) two subgoals, checks enforced

NAME_GOAL = 'Full name: type "Ada Lovelace", then click the "Next" button'
CITY_GOAL = 'City: type "Paris", then click the "Finish" button'
JEV_WIZARD = {
    "Full name:": [("TYPE_TEXT", "Full name"), ("CLICK", "Next")],
    "City:": [("TYPE_TEXT", "City"), ("CLICK", "Finish")],
    "Pretend": [],   # Jev claims DONE without doing anything
}


@needs_chrome
def test_a_two_subgoal_task_completes_with_its_checks(wired):
    jev = FakeJev(JEV_WIZARD)
    fake = FakePlanner([
        plan((NAME_GOAL, [{"t": "text_contains", "s": "Step 2 of 2"}], 4),
             (CITY_GOAL, [{"t": "text_contains", "s": "Thanks Ada Lovelace from Paris"}], 4)),
        plan(done=True, ans="Signed up as Ada Lovelace from Paris"),
    ])
    wired(jev, fake)

    out = server.browser_task("Sign up as Ada Lovelace from Paris", url=_page(WIZARD), session="t-a", verbose=True)

    assert "status: done" in out, out
    assert "answer: Signed up as Ada Lovelace from Paris" in out
    assert out.count("[ok]") == 2, out
    # The second subgoal ran straight from the plan: one planning call, one closing review.
    assert len(fake.prompts) == 2, out
    assert "Thanks Ada Lovelace from Paris" in out


@needs_chrome
def test_a_subgoal_jev_calls_done_but_whose_check_fails_is_replanned_with_the_failure(wired):
    jev = FakeJev(JEV_WIZARD)
    fake = FakePlanner([
        plan(("Pretend the name is filled in", [{"t": "text_contains", "s": "Step 2 of 2"}], 3)),
        plan((NAME_GOAL, [{"t": "text_contains", "s": "Step 2 of 2"}], 4)),
        plan(done=True, ans="reached step 2"),
    ])
    wired(jev, fake)

    out = server.browser_task("Get to step 2 as Ada Lovelace", url=_page(WIZARD), session="t-a2")

    assert "status: done" in out, out
    assert "[FAIL] Pretend the name is filled in" in out
    assert "jev: done" in out, "Jev's claim is reported, and overruled by the check"
    replan = fake.prompts[1]
    assert "result: FAILED" in replan and "jev status: done" in replan, replan
    assert "X  text_contains" in replan and "Step 2 of 2" in replan, replan


# ------------------------------------------------------------------ (b) giving up

@needs_chrome
def test_the_planner_gives_up_after_three_failed_subgoals(wired):
    jev = FakeJev(JEV_WIZARD)
    fake = FakePlanner([plan(("Pretend it is done", [{"t": "text_contains", "s": "Never on page"}], 3))])
    wired(jev, fake)

    out = server.browser_task("Do the impossible", url=_page(WIZARD), session="t-b")

    assert "status: stopped: planner gave up" in out, out
    assert out.count("[FAIL]") == 3, out
    assert len(fake.prompts) == 3, "a plan, then a replan after each of the first two failures"


# ------------------------------------------------------------------ (c) parse failures

@needs_chrome
def test_a_malformed_plan_is_retried_once_then_reported(wired):
    jev = FakeJev(JEV_WIZARD)
    fake = FakePlanner(["not json at all", '{"done": "maybe"}'])
    wired(jev, fake)

    out = server.browser_task("Sign up", url=_page(WIZARD), session="t-c")

    assert out.splitlines()[1].startswith("status: error:"), out
    assert "after 2 attempts" in out
    assert len(fake.prompts) == 2
    assert jev.calls == 0, "nothing may run on a plan that was never read"


@needs_chrome
def test_one_malformed_plan_is_recovered_by_the_retry(wired):
    jev = FakeJev(JEV_WIZARD)
    fake = FakePlanner(["```json\n{broken", plan(done=True, ans="nothing to do")])
    wired(jev, fake)

    out = server.browser_task("Say hello", url=_page(WIZARD), session="t-c2")

    assert "status: done" in out, out
    assert "timing: planner 1 call " in out, out


def test_parse_plan_drops_unknown_and_js_checks():
    raw = plan(("Click the \"Next\" button",
                [{"t": "js", "s": "document.cookie"}, {"t": "text_contains", "s": ""},
                 {"t": "value_named", "s": "City"}, {"t": "url_contains", "s": "/done"}], 99))
    parsed = planner.parse_plan(raw, max_steps_cap=15)
    assert parsed.subgoals[0].checks == [{"type": "url_contains", "text": "/done"}]
    assert parsed.subgoals[0].max_steps == 15
    assert parsed.dropped_checks == 3


def test_the_request_asks_for_strict_json_at_the_configured_effort(monkeypatch):
    seen = {}

    def fake(cfg, body):
        seen.update(body)
        return {"content": [{"type": "text", "text": json.dumps(plan(done=True))}]}
    monkeypatch.setattr(planner, "_http", fake)
    cfg = dataclasses.replace(Config.from_env(), planner_key="k", planner_model="claude-opus-5-5",
                              planner_effort="low")
    planner.ask(cfg, "TASK: x", 10)
    assert seen["model"] == "claude-opus-5-5"
    assert seen["output_config"]["effort"] == "low"
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert "untrusted" in seen["system"] and "confirmation" in seen["system"]


# ------------------------------------------------------------------ (d) the rail

@needs_chrome
def test_the_planner_cannot_get_past_the_confirmation_rail(wired):
    jev = FakeJev({"Delete": [("CLICK", "Delete account")]})
    fake = FakePlanner([
        plan(('Click the "Delete account" button', [{"t": "title_matches", "s": "DELETED"}], 3)),
        plan(('Click the "Delete account" button', [], 3)),
    ])
    wired(jev, fake)

    out = server.browser_task("Delete my account", url=_page(DELETE), session="t-d")

    assert "status: blocked: needs confirmation" in out, out
    session = server.MANAGER.session("t-d")
    assert session.observe().title == "Account", "the delete must not have run"
    assert len(fake.prompts) == 1, "a rail stop is the user's call, never handed back for a retry"


def test_the_planner_output_has_no_field_that_can_confirm():
    text = json.dumps(planner.PLAN_SCHEMA)
    assert "confirm" not in text and "ref" not in text
    assert "js" not in planner.PLANNER_CHECKS


# ------------------------------------------------------------------ (e) latency accounting

@needs_chrome
def test_the_time_split_is_measured_and_reported(wired):
    jev = FakeJev(JEV_WIZARD, delay=0.06)
    fake = FakePlanner([
        plan((NAME_GOAL, [{"t": "value_named", "s": "City", "v": "", "r": "textbox"},
                          {"t": "text_contains", "s": "Step 2 of 2"}], 4)),
        plan(done=True, ans="ok"),
    ], delay=0.05)
    wired(jev, fake)

    out = server.browser_task("Reach step 2", url=_page(WIZARD), session="t-e")

    timing = next(line for line in out.splitlines() if line.startswith("timing: "))
    match = re.match(r"timing: planner 2 calls ([\d.]+)s \(([\d./]+)s\) · jev (\d+) decisions ([\d.]+)s"
                     r" · page ([\d.]+)s · ([\d.]+)s wall · 300 planner tokens", timing)
    assert match, timing
    planner_s, per_call, decisions, jev_s, page_s, wall_s = match.groups()
    assert float(planner_s) >= 0.1 and len(per_call.split("/")) == 2
    # Two actions plus the DONE that closed the subgoal.
    assert int(decisions) == jev.calls == 3
    assert float(jev_s) >= 0.1
    assert float(wall_s) >= float(planner_s) + float(jev_s)


# ------------------------------------------------------------------ what the planner reads

def _obs(elements, text="", url="https://shop.test/cart?token=abc#x"):
    return Observation(url=url, title="Cart", text=text, elements=elements, digest="d", text_digest="t",
                       page_key="k", scroll={}, reachable=len(elements), omitted=0,
                       overlays=[{"name": "Cookie consent", "modal": True}], cross_frames=0,
                       cross_frame_srcs=[])


def test_the_page_brief_is_masked_like_the_decision_models_view():
    elements = [Element(ref="e1", role="textbox", name="Password", value="hunter2", editable=True,
                        secret=True),
                Element(ref="e2", role="textbox", name="Card holder", value="Ada", editable=True),
                Element(ref="e3", role="button", name="Place order")]
    brief = planner.page_brief(_obs(elements, text="card 4111 1111 1111 1111 otp: 998877"), "buy it")
    assert "hunter2" not in brief and "\u00abhidden\u00bb" in brief
    assert "4111" not in brief and "998877" not in brief
    assert "token=abc" not in brief and "url: https://shop.test/cart" in brief
    assert 'e2 textbox "Card holder" ="Ada"' in brief
    assert "dialog open: Cookie consent" in brief


def test_the_brief_caps_the_element_list():
    elements = [Element(ref=f"e{i}", role="link", name=f"Link {i}") for i in range(200)]
    brief = planner.page_brief(_obs(elements), "anything")
    assert f"elements ({planner.PAGE_ELEMENTS} of 200)" in brief


# ------------------------------------------------------------------ planner goals and literal values

@pytest.mark.parametrize("goal, field, value", [
    ('Where from?: type "JFK", then click the suggestion "John F. Kennedy International Airport (JFK)"',
     "Where from?", "JFK"),
    ('Where to?: type "London Heathrow" and click the suggestion "Heathrow Airport (LHR)"',
     "Where to?", "London Heathrow"),
    ('Full name: type "Ada Lovelace", then click the "Next" button', "Full name", "Ada Lovelace"),
    ('Search: type "noise cancelling headphones" and press Enter', "Search", "noise cancelling headphones"),
    ('Email: type "ada@example.com"', "Email", "ada@example.com"),
    ('Departure: type "Fri, Oct 3", then press Enter', "Departure", "Fri, Oct 3"),
])
def test_the_goal_format_the_planner_is_told_to_use_is_read_literally(goal, field, value):
    assert value in {v for _f, v in policy.literal_values(goal)}
    element = Element(ref="e1", role="combobox", name=field, editable=True)
    cfg = dataclasses.replace(Config.from_env(), typesafe_key="k")
    assert policy.literal_for(cfg, goal, element, _obs([element])) == value


def test_the_system_prompt_names_that_format():
    assert '<field label>: type "<value>"' in planner.SYSTEM


# ------------------------------------------------------------------ the parsed Jev reply

def test_the_goal_reply_is_read_for_status_steps_time_and_trace():
    text = "\n".join([
        "goal: x", "status: done", "steps: 2",
        "turbo: 3 decisions · 1,234 tokens · 1.2s model + 0.4s page · 1.9s wall",
        "trace:",
        "  1. TYPE_TEXT e3 City → ok (400ms model / 120ms browser)",
        "  -   re-observing (stale)",
        "  2. CLICK e4 Next → ok (400ms model / 280ms browser)",
        "  3. DONE (conf 0.90)",
        "", "https://x — 4 elements",
    ])
    run = planner.parse_goal_output(text)
    assert (run.status, run.steps, run.decisions, run.jev_ms, run.page_ms) == ("done", 2, 3, 1200, 400)
    assert len(run.trace) == 4


def test_an_error_reply_is_an_error_status():
    assert planner.parse_goal_output("browser_error: gone").status.startswith("error: browser_error")


def test_without_a_planner_key_the_tool_says_so(monkeypatch):
    monkeypatch.setattr(server, "CONFIG", dataclasses.replace(Config.from_env(), typesafe_key="k",
                                                               planner_key=None))
    assert server.browser_task("anything").startswith("planner_unavailable:")


def test_the_planner_key_falls_back_to_the_anthropic_key(monkeypatch):
    monkeypatch.delenv("PLANNER_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    monkeypatch.setenv("PLANNER_BASE_URL", "http://127.0.0.1:18802")
    cfg = Config.from_env()
    assert cfg.planner_key == "anthropic-test"
    assert cfg.planner_base_url == "http://127.0.0.1:18802"
    assert (cfg.planner_model, cfg.planner_effort) == ("claude-opus-5-5", "low")
    monkeypatch.setenv("PLANNER_API_KEY", "planner-test")
    assert Config.from_env().planner_key == "planner-test"
