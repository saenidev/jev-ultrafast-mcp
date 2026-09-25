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
import inspect
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
def test_a_malformed_plan_is_retried_once_then_jev_runs_the_task_alone(wired):
    jev = FakeJev(JEV_WIZARD)
    fake = FakePlanner(["not json at all", '{"done": "maybe"}'])
    wired(jev, fake)

    out = server.browser_task("Sign up", url=_page(WIZARD), session="t-c")

    assert "after 2 attempts" in out, out
    assert "planner unavailable" in out and "ran Jev alone" in out, out
    assert len(fake.prompts) == 2
    assert "REFUSED" in fake.prompts[1], "the retry says why the first answer was refused"
    assert jev.goals and jev.goals[0] == "Sign up", "the fallback is one Jev run of the task itself"


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
    assert parsed.subgoals[0].pick is None
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
    # Two actions, plus the DONE that closed the subgoal when browser_goal has no `until`.
    assert int(decisions) == jev.calls
    assert jev.calls == (2 if "until" in inspect.signature(server.browser_goal).parameters else 3)
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


def test_without_any_key_the_tool_says_so(monkeypatch):
    monkeypatch.setattr(server, "CONFIG", dataclasses.replace(Config.from_env(), typesafe_key=None,
                                                               planner_key=None))
    assert server.browser_task("anything").startswith("turbo_unavailable:")


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


# ------------------------------------------------------------------ plan once, replan on failure
#
# These drive `planner.run` directly: the page is a mutable fake Observation, `execute` is a fake
# browser_goal that edits it and answers in browser_goal's text format, and `check` is the real
# `assertions.run`. So they pin what the loop decides and how many calls it spends.

def _el(ref, role, name, value="", **kw):
    return Element(ref=ref, role=role, name=name, value=value, **kw)


class World:
    """A page the fake goals change. `script` maps a goal substring to (edit, status)."""

    def __init__(self, elements, text="", url="https://www.google.com/travel/flights"):
        self.obs = _obs(list(elements), text=text, url=url)
        self.obs.title = "Google Flights"
        self.script: dict[str, list] = {}
        self.calls: list[tuple[str, int, object]] = []
        self.picks: list[dict] = []
        self.pick_reply = "ok\nclicked"
        self.pick_edit = None
        self.until_supported = True

    def observe(self):
        self.obs.by_ref = {e.ref: e for e in self.obs.elements}
        return self.obs

    def element(self, name):
        return next(e for e in self.obs.elements if e.name.startswith(name))

    def execute(self, goal, steps, until=None):
        self.calls.append((goal, steps, until))
        todo = next((v for k, v in self.script.items() if k in goal), [(None, "done")])
        edit, status = todo.pop(0) if len(todo) > 1 else todo[0]
        if edit:
            edit(self)
        lines = [f"goal: {goal}", f"status: {status}", "steps: 2"]
        decisions = 2
        if until and self.until_supported:
            met = assertions.run(until, self.observe())["pass"]
            if met and status == "done":
                lines.append("until: met after 2 steps")
            elif not met:
                lines.append("until: not met (x)")
                decisions = 3
        else:
            decisions = 3
        lines.append(f"turbo: {decisions} decisions · 100 tokens · 1.0s model + 0.5s page · 2.0s wall")
        lines += ["trace:", "  1. CLICK e1 x → ok (1ms model / 1ms browser)", "", "view"]
        return "\n".join(lines)

    def pick(self, spec):
        self.picks.append(spec)
        if self.pick_edit:
            self.pick_edit(self)
        return self.pick_reply


def _cfg(**kw):
    base = {"typesafe_key": "k", "planner_key": "pk", "planner_base_url": "http://planner.invalid"}
    return dataclasses.replace(Config.from_env(), **{**base, **kw})


def _check(checks, observation):
    return assertions.run(checks, observation, allow_js=False)


def _run(world, fake, *, plan_=None, pick=True, cfg=None, task="Book nothing, just search"):
    return planner.run(cfg or _cfg(), task, observe=world.observe, execute=world.execute,
                       check=_check, pick=world.pick if pick else None, plan=plan_,
                       max_subgoals=12, max_steps_per_subgoal=15, sleep=lambda _s: None)


def _set(name, value):
    def edit(world):
        world.element(name).name = f"{name} {value}"
    return edit


def full_plan(*subgoals, done=False, ans="", note="", fin=True):
    body = plan(*subgoals, done=done, ans=ans, note=note)
    body["fin"] = fin
    return body


def flights_world():
    return World([_el("e1", "combobox", "Where from?", "Bangkok", editable=True),
                  _el("e2", "combobox", "Where to?", "", editable=True),
                  _el("e3", "button", "Search")])


def test_a_whole_plan_runs_on_one_planner_call(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(_set("Where to?", "Seoul ICN"), "done")],
                    "Search": [(lambda w: setattr(w.obs, "url", w.obs.url + "/search?tfs=1"), "done")]}
    fake = FakePlanner([full_plan(
        ('Where to?: type "ICN", then click the suggestion "Incheon International Airport"',
         [{"t": "field_shows", "s": "Where to?", "v": "ICN"}], 6),
        ('Click the "Search" button', [{"t": "url_contains", "s": "/search"}], 3))])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert report["status"] == "done", report
    assert report["planner_calls"] == 1, "no review after a new page, no closing call when fin"
    assert report["until_hits"] == 2
    assert [c[2] for c in world.calls] == [[{"type": "field_shows", "name": "Where to?", "value": "ICN"}],
                                           [{"type": "url_contains", "text": "/search"}]]


def test_a_plan_that_needs_the_final_page_asks_once_for_the_answer(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(_set("Where to?", "Seoul ICN"), "done")]}
    fake = FakePlanner([
        full_plan(('Where to?: type "ICN"', [{"t": "field_shows", "s": "Where to?", "v": "ICN"}], 4),
                  fin=False),
        full_plan(done=True, ans="cheapest is 198 USD")])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert (report["status"], report["answer"], report["planner_calls"]) == ("done", "cheapest is 198 USD", 2)
    assert "Every planned subgoal ran and passed" in fake.prompts[1]
    assert report["replans"] == 0


def test_a_failed_subgoal_is_the_only_thing_that_replans(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(None, "stopped: looping"), (_set("Where to?", "Seoul ICN"), "done")]}
    fake = FakePlanner([
        full_plan(('Where to?: type "ICN"', [{"t": "field_shows", "s": "Where to?", "v": "ICN"}], 4),
                  ('Click the "Search" button', [], 3)),
        full_plan(('Where to?: type "ICN", then press Enter',
                   [{"t": "field_shows", "s": "Where to?", "v": "ICN"}], 4),
                  ('Click the "Search" button', [], 3))])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert report["status"] == "done", report
    assert (report["planner_calls"], report["replans"]) == (2, 1)
    assert "LAST SUBGOAL FAILED" in fake.prompts[1] and "stopped: looping" in fake.prompts[1]
    assert "THE REST OF THE PLAN" in fake.prompts[1] and 'Click the "Search" button' in fake.prompts[1]


def test_checks_that_already_hold_are_dropped_and_reported(monkeypatch):
    world = flights_world()
    fake = FakePlanner([full_plan(
        ('Where from?: type "BKK"', [{"t": "field_shows", "s": "Where from?", "v": "Bangkok"},
                                     {"t": "field_shows", "s": "Where to?", "v": "ICN"}], 4))])
    world.script = {"Where from?:": [(_set("Where to?", "ICN"), "done")]}
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert world.calls[0][2] == [{"type": "field_shows", "name": "Where to?", "value": "ICN"}]
    assert report["dropped_checks"] == 1
    assert "1 check held before it ran, dropped" in report["subgoals"][0].note
    assert "1 dropped checks" in planner.render("t", report)


def test_when_every_check_already_holds_jev_done_decides(monkeypatch):
    world = flights_world()
    world.script = {"Where from?:": [(None, "blocked")]}
    fake = FakePlanner([
        full_plan(('Where from?: type "BKK"', [{"t": "field_shows", "s": "Where from?", "v": "Bangkok"}], 4)),
        full_plan(done=True, ans="gave up")])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert world.calls[0][2] is None, "no `until` left to pass"
    assert not report["subgoals"][0].ok, "a held check is not evidence the subgoal did anything"
    assert "Jev's DONE decided" in report["subgoals"][0].note


def test_without_until_support_checks_are_run_after_the_goal(monkeypatch):
    world = flights_world()
    world.until_supported = False
    world.script = {"Where to?:": [(_set("Where to?", "Seoul ICN"), "done")]}
    fake = FakePlanner([full_plan(('Where to?: type "ICN"',
                                   [{"t": "field_shows", "s": "Where to?", "v": "icn"}], 4))])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert report["status"] == "done" and report["until_hits"] == 0
    assert report["subgoals"][0].checks[0]["ok"]


def test_a_jev_http_error_retries_the_subgoal_once_without_replanning(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(None, "turbo_unavailable: Decision model returned HTTP 400"),
                                   (_set("Where to?", "Seoul ICN"), "done")]}
    fake = FakePlanner([full_plan(('Where to?: type "ICN"',
                                   [{"t": "field_shows", "s": "Where to?", "v": "ICN"}], 4))])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert report["status"] == "done", report
    assert (report["planner_calls"], report["jev_retries"], len(world.calls)) == (1, 1, 2)


def test_a_jev_error_twice_ends_the_task_as_an_error(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(None, "turbo_unavailable: HTTP 400")]}
    fake = FakePlanner([full_plan(('Where to?: type "ICN"', [], 4))])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert report["status"].startswith("error: turbo_unavailable"), report
    assert len(world.calls) == 2 and report["planner_calls"] == 1


# ------------------------------------------------------------------ fallbacks

def test_a_planner_that_fails_first_falls_back_to_one_jev_run(monkeypatch):
    world = flights_world()

    def down(cfg, body):
        raise planner.PlannerError("planner HTTP 529: overloaded")
    monkeypatch.setattr(planner, "_http", down)
    report = planner.run(_cfg(), "Search BKK to ICN", observe=world.observe, execute=world.execute,
                         check=_check, max_steps_per_subgoal=10, sleep=lambda _s: None)
    assert report["status"] == "done", report
    assert report["plan_source"] == "jev-alone"
    assert world.calls == [("Search BKK to ICN", 30, None)]
    out = planner.render("Search BKK to ICN", report)
    assert "planner unavailable, ran Jev alone" in out and "timing: planner 1 call" in out


def test_no_planner_key_runs_jev_alone(monkeypatch):
    world = flights_world()
    monkeypatch.setattr(planner, "_http", lambda *_a: pytest.fail("no planner call without a key"))
    report = planner.run(_cfg(planner_key=None), "Search", observe=world.observe,
                         execute=world.execute, check=_check, sleep=lambda _s: None)
    assert report["plan_source"] == "jev-alone" and report["planner_calls"] == 0
    assert report["status"] == "done"


# ------------------------------------------------------------------ host-supplied plans

HOST_PLAN = [
    {"goal": 'Where to?: type "ICN", then click the suggestion "Incheon International Airport"',
     "checks": [{"type": "field_shows", "name": "Where to?", "value": "ICN"}], "max_steps": 6},
    {"goal": 'Click the "Search" button', "checks": [], "max_steps": 3},
]


def test_a_host_plan_runs_with_zero_planner_calls(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(_set("Where to?", "Seoul ICN"), "done")]}
    monkeypatch.setattr(planner, "_http", lambda *_a: pytest.fail("the host already planned"))
    report = planner.run(_cfg(), "search", observe=world.observe, execute=world.execute,
                         check=_check, plan=HOST_PLAN, sleep=lambda _s: None)
    assert report["status"] == "done" and report["planner_calls"] == 0
    assert report["plan_source"] == "host"
    assert world.calls[0][1] == 6


def test_a_failing_host_plan_calls_the_planner(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(None, "stopped: hit max_steps=6")]}
    fake = FakePlanner([full_plan(done=True, ans="could not set the destination")])
    monkeypatch.setattr(planner, "_http", fake)
    report = planner.run(_cfg(), "search", observe=world.observe, execute=world.execute,
                         check=_check, plan=HOST_PLAN, sleep=lambda _s: None)
    assert report["planner_calls"] == 1 and "LAST SUBGOAL FAILED" in fake.prompts[0]
    assert report["answer"] == "could not set the destination"


def test_a_host_plan_follows_the_same_safety_rules(monkeypatch):
    world = flights_world()
    monkeypatch.setattr(planner, "_http", lambda *_a: pytest.fail("refused before anything runs"))
    bad = [{"goal": "Pick the first chip", "pick": {"role": "button", "name_regex": "^Remove",
                                                    "key": "min_number"}}]
    report = planner.run(_cfg(), "search", observe=world.observe, execute=world.execute,
                         check=_check, plan=bad, sleep=lambda _s: None)
    assert report["status"].startswith("error: the supplied plan was refused"), report
    assert "consequential" in report["status"] and not world.calls


# ------------------------------------------------------------------ pick subgoals

RESULTS = [_el("e10", "link", "From 312 US dollars. Nonstop flight with Korean Air."),
           _el("e11", "link", "From 198 US dollars. 1 stop flight with Sun PhuQuoc Airways."),
           _el("e12", "button", "View more flights")]


def test_a_pick_subgoal_runs_without_a_model(monkeypatch):
    world = World(RESULTS, text="Top departing flights")
    world.pick_edit = lambda w: setattr(w.obs, "text", "Booking options")
    fake = FakePlanner([full_plan(
        ("Pick the cheapest flight", [{"t": "text_contains", "s": "Booking options"}], 1))])
    fake.answers[0]["sg"][0]["p"] = {"r": "link", "re": "^From [0-9,]+ US dollars", "k": "min",
                                     "nr": "From ([0-9,]+)"}
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert report["status"] == "done", report
    assert world.picks == [{"role": "link", "name_regex": "^From [0-9,]+ US dollars",
                            "key": "min_number", "number_regex": "From ([0-9,]+)"}]
    assert not world.calls and report["jev_decisions"] == 0 and report["picks"] == 1
    assert report["subgoals"][0].kind == "pick"


def test_without_run_click_best_a_pick_fails_clearly_and_replans(monkeypatch):
    world = World(RESULTS)
    first = full_plan(("Pick the cheapest flight", [], 1))
    first["sg"][0]["p"] = {"r": "link", "re": "^From", "k": "min"}
    fake = FakePlanner([first, full_plan(done=True, ans="n/a")])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake, pick=False)
    assert "pick unavailable" in report["subgoals"][0].status
    assert "pick (p) is not available" in fake.prompts[0]
    assert report["planner_calls"] == 2


def test_a_pick_that_meets_the_rail_blocks_the_task(monkeypatch):
    world = World(RESULTS)
    world.pick_reply = "needs_confirmation: matches confirmation rule"
    first = full_plan(("Pick the cheapest", [], 1))
    first["sg"][0]["p"] = {"r": "link", "re": "^From", "k": "min"}
    fake = FakePlanner([first])
    monkeypatch.setattr(planner, "_http", fake)
    report = _run(world, fake)
    assert report["status"].startswith("blocked: needs confirmation"), report
    assert report["planner_calls"] == 1


@pytest.mark.parametrize("pick, why", [
    ({"r": "button", "re": "Remove", "k": "min"}, "consequential"),
    ({"r": "button", "re": "^(Pay|Book) now", "k": "min"}, "consequential"),
    ({"r": "link", "re": ".*", "k": "min"}, "names nothing literal"),
    ({"r": "link", "re": "From (", "k": "min"}, "does not compile"),
    ({"r": "textbox", "re": "From", "k": "min"}, "role"),
    ({"r": "link", "re": "From", "k": "median"}, "min/max"),
])
def test_bad_picks_are_refused_when_the_plan_is_read(pick, why):
    raw = full_plan(("Pick it", [], 1))
    raw["sg"][0]["p"] = pick
    with pytest.raises(planner.PlannerError, match=re.escape(why)):
        planner.parse_plan(raw, 15, task="find the cheapest flight")


def test_a_subgoal_that_removes_what_the_task_never_asked_to_remove_is_refused():
    raw = full_plan(('Click the "Remove flight from Bangkok to Seoul" button', [], 2))
    with pytest.raises(planner.PlannerError, match="edit fields instead"):
        planner.parse_plan(raw, 15, task="Search a multi-city trip BKK to ICN then ICN to NRT")
    # A task that asks for it may plan it (the rail still stops it for the user).
    planner.parse_plan(raw, 15, task="Remove the flight from Bangkok to Seoul")
    # A typed value is not a control.
    ok = full_plan(('Search: type "delete key replacement"', [], 2))
    planner.parse_plan(ok, 15, task="Find a delete key replacement")
    planner.parse_plan(full_plan(('Note: type "please delete nothing"', [], 2)), 15, task="fill the note")


def test_the_schema_offers_pick_compactly():
    items = planner.PLAN_SCHEMA["properties"]["sg"]["items"]["properties"]
    assert set(items["p"]["properties"]) == {"r", "re", "k", "nr"}
    assert "field_shows" in planner.PLANNER_CHECKS


# ------------------------------------------------------------------ field_shows

def test_field_shows_reads_a_combobox_name_as_well_as_its_value():
    obs = _obs([_el("e1", "combobox", "Where from? New York JFK", "New York", editable=True),
                _el("e2", "combobox", "Cabin class", "", current="Business")])
    obs.by_ref = {e.ref: e for e in obs.elements}
    run = lambda c: assertions.run(c, obs)["pass"]  # noqa: E731
    assert run([{"type": "field_shows", "name": "Where from?", "value": "JFK"}])
    assert run([{"type": "field_shows", "name": "where from", "value": "jfk"}])
    assert run([{"type": "field_shows", "name": "Where from?", "value": "new york"}])
    assert run([{"type": "field_shows", "name": "Cabin", "value": "business"}])
    assert not run([{"type": "field_shows", "name": "Where from?", "value": "LHR"}])
    # The field's own label is not what it shows.
    assert not run([{"type": "field_shows", "name": "Where from?", "value": "from"}])
    # value_named keeps its old meaning unless asked; the planner's value_named asks.
    assert not run([{"type": "value_named", "name": "Where from?", "value": "JFK"}])
    assert run([{"type": "value_named", "name": "Where from?", "value": "JFK", "in_name": True}])
    assert planner.PLANNER_CHECKS["value_named"]({"s": "a", "v": "b"})["in_name"] is True


def test_field_shows_never_reads_a_secret():
    obs = _obs([_el("e1", "textbox", "Password hunter2", "hunter2", secret=True, editable=True)])
    obs.by_ref = {e.ref: e for e in obs.elements}
    result = assertions.run([{"type": "field_shows", "name": "Password", "value": "hunter2"}], obs)
    assert not result["pass"] and "hunter2" not in json.dumps(result)


# ------------------------------------------------------------------ what the planner is told

def test_the_prompt_plans_whole_tasks_in_few_subgoals_and_never_removes_rows():
    for phrase in ("Plan the WHOLE task", "as FEW subgoals", "field_shows", "never for a field's value",
                   "never after a", "Never remove a row or chip", "Remove/Delete/Clear", "p (pick)",
                   "View more flights", '"k":"min"'):
        assert phrase in planner.SYSTEM, phrase


def test_the_report_counts_what_the_run_cost(monkeypatch):
    world = flights_world()
    world.script = {"Where to?:": [(_set("Where to?", "Seoul ICN"), "done")]}
    fake = FakePlanner([full_plan(('Where to?: type "ICN"',
                                   [{"t": "field_shows", "s": "Where to?", "v": "ICN"}], 4))])
    monkeypatch.setattr(planner, "_http", fake)
    out = planner.render("t", _run(world, fake))
    assert re.search(r"^timing: planner 1 call .* · jev 2 decisions .*s wall", out, re.M), out
    assert "plan: planner · 0 replans · 0 picks · 1 until-hits · 0 dropped checks" in out
    assert "until met" in out


# ------------------------------------------------------------------ server wiring (real page)

@needs_chrome
def test_browser_task_runs_a_host_plan_on_a_real_page_with_no_planner(wired):
    jev = FakeJev(JEV_WIZARD)
    fake = FakePlanner([plan(done=True)])
    wired(jev, fake)
    host = [{"goal": NAME_GOAL, "checks": [{"type": "text_contains", "text": "Step 2 of 2"}], "max_steps": 4},
            {"goal": CITY_GOAL, "checks": [{"type": "text_contains", "text": "Thanks Ada"}], "max_steps": 4}]

    out = server.browser_task("Sign up as Ada Lovelace from Paris", url=_page(WIZARD), session="t-h",
                              plan=host)

    assert "status: done" in out, out
    assert "timing: planner 0 calls" in out and not fake.prompts
    assert "plan: host" in out


@needs_chrome
def test_browser_task_uses_run_click_best_when_the_server_has_it(wired, monkeypatch):
    jev = FakeJev({})
    seen = []

    def click_best(session, spec):
        seen.append((session, spec))
        return "ok\nclicked e3"
    monkeypatch.setattr(server, "run_click_best", click_best, raising=False)
    host = [{"goal": "Pick the dearest", "pick": {"role": "button", "name_regex": "Save",
                                                  "key": "max_number"}}]
    wired(jev, FakePlanner([plan(done=True)]))

    out = server.browser_task("save it", url=_page(DELETE), session="t-p", plan=host)

    assert "status: done" in out, out
    assert seen == [("t-p", {"role": "button", "name_regex": "Save", "key": "max_number"})]
    assert jev.calls == 0 and "1 picks" in out


@needs_chrome
def test_browser_task_passes_checks_as_until_when_browser_goal_takes_it(wired, monkeypatch):
    wired(FakeJev({}), FakePlanner([plan(done=True)]))
    seen = []

    def goal_with_until(goal, url="", session="default", max_steps=20, verify=None, verbose=False,
                        until=None):
        seen.append(until)
        return (f"goal: {goal}\nstatus: done\nsteps: 1\nuntil: met after 1 step\n"
                "turbo: 1 decision · 10 tokens · 0.3s model + 0.1s page · 0.5s wall\ntrace:\n")
    monkeypatch.setattr(server, "browser_goal", goal_with_until)
    host = [{"goal": NAME_GOAL, "checks": [{"type": "text_contains", "text": "Never on this page"}],
             "max_steps": 4}]

    out = server.browser_task("Sign up", url=_page(WIZARD), session="t-u", plan=host)

    assert seen == [[{"type": "text_contains", "text": "Never on this page"}]]
    assert "status: done" in out and "1 until-hits" in out, out
    assert "jev 1 decisions" in out


@pytest.mark.parametrize("reply, status", [
    ("ok\nclicked", "ok"),
    ('click_best: ok ref=e11 number=198 candidates=2 covered=0 target="From 198"\n[delta]', "ok ref=e11"),
    ("error: no candidates", "error: no candidates"),
    ('click_best: failed error=no_candidates detail="none matched"', "error: failed error=no_candidates"),
    ('click_best: failed error=needs_confirmation detail="matches rule"', "needs_confirmation: "),
    ("needs_confirmation", "needs_confirmation"),
    ("", "error: empty reply"),
])
def test_both_run_click_best_reply_shapes_are_read(reply, status):
    assert planner._pick_status(reply).startswith(status)


def test_a_prefixed_rail_refusal_from_a_pick_blocks_the_task(monkeypatch):
    world = World(RESULTS)
    world.pick_reply = 'click_best: failed error=needs_confirmation detail="x"'
    first = full_plan(("Pick the cheapest", [], 1))
    first["sg"][0]["p"] = {"r": "link", "re": "^From", "k": "min"}
    fake = FakePlanner([first])
    monkeypatch.setattr(planner, "_http", fake)
    assert _run(world, fake)["status"].startswith("blocked: needs confirmation")
