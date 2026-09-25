"""The third independent review's rail bypasses, against a real page.

Each consequential page below sets `document.title` to PAID/DELETED/UNSUB/ORDERED if the guarded
control ran. A correct rail answers `needs_confirmation` and leaves the title alone; a benign page
must still work. Real Chrome, real observer, real `Session._run_op`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_ultrafast_mcp.browser import BrowserManager  # noqa: E402
from jev_ultrafast_mcp.config import Config, find_chrome  # noqa: E402
from jev_ultrafast_mcp.safety import confirm_reason  # noqa: E402


def _chrome_available() -> bool:
    try:
        find_chrome(None)
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _chrome_available(), reason="needs Chrome")

FIRED = ("PAID", "DELETED", "UNSUB", "ORDERED")

TAB_TO_DELETE = """<!doctype html><title>settings</title>
<form onsubmit="return false"><label>Search <input name=q value=x></label></form>
<button onclick="document.title='DELETED'">Delete repository</button>"""

BLOCK = {
    "keys [Tab, Space] in one op onto Delete": (TAB_TO_DELETE, "Search", {"op": "keys", "keys": ["Tab", "Space"]}),
    "keys [Tab, Enter] in one op onto Delete": (TAB_TO_DELETE, "Search", {"op": "keys", "keys": ["Tab", "Enter"]}),
    "keys [Tab, Enter] from nothing focused": ("""<!doctype html><title>settings</title>
<button onclick="document.title='DELETED'">Delete repository</button>""", None,
        {"op": "keys", "keys": ["Tab", "Enter"]}),
    "Enter on a focused Unsubscribe link": ("""<!doctype html><title>mail</title>
<a id=u href="#" onclick="document.title='UNSUB';return false">Unsubscribe</a>
<script>u.focus()</script>""", None, {"op": "keys", "keys": "Enter"}),
    "Enter on a focused menuitem": ("""<!doctype html><title>acct</title>
<div role=menu><div role=menuitem tabindex=0 id=m onclick="document.title='DELETED'"
 onkeydown="if(event.key==='Enter')this.click()">Delete account</div></div><script>m.focus()</script>""",
        None, {"op": "keys", "keys": "Enter"}),
    "no form: benign Clear inside, Delete one wrapper up": ("""<!doctype html><title>settings</title>
<div class=modal>
 <div class=row><label>Repository name <input id=n value=myrepo></label><button type=button>Clear</button></div>
 <div class=footer><button id=d onclick="document.title='DELETED'">Delete repository</button></div>
</div>
<script>n.addEventListener('keydown',e=>{if(e.key==='Enter')d.click()})</script>""", "Repository",
        {"op": "type", "text": "", "clear": False, "submit": True}),
    "form whose handler presses a guarded button outside it": ("""<!doctype html><title>settings</title>
<section><form id=f onsubmit="return false"><label>Repository name <input id=n value=myrepo></label></form>
<div class=danger><button id=d onclick="document.title='DELETED'">Delete repository</button></div></section>
<script>n.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();d.click()}})</script>""",
        "Repository", {"op": "type", "text": "", "clear": False, "submit": True}),
    "Enter; button labelled only by CSS ::before": ("""<!doctype html><title>checkout</title>
<style>.pay::before{content:'Pay now'}</style>
<form onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<button class=pay></button></form>""", "Amount", {"op": "type", "text": "", "clear": False, "submit": True}),
    "type op aimed at a Pay now button": ("""<!doctype html><title>checkout</title>
<form onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<button>Pay now</button></form>""", "Pay now", {"op": "type", "text": ""}),
}

CLICK_BLOCK = {
    "decoy role=button nested inside Pay now": ("""<!doctype html><title>checkout</title>
<form onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<button style="padding:20px">Pay now <span role=button aria-label=Continue
 style="display:inline-block;padding:10px;background:#eee">&rarr;</span></button></form>""", "Continue"),
    "button labelled only by CSS ::before": ("""<!doctype html><title>checkout</title>
<style>.pay::before{content:'Pay now'}</style>
<form onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<button class=pay></button></form>""", ""),
    "zero-width space inside Pay now": ("""<!doctype html><title>checkout</title>
<form onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<button>Pay\u200b now</button></form>""", "Pay"),
    "Place your order": ("""<!doctype html><title>review</title>
<button onclick="document.title='ORDERED'">Place your order</button>""", "Place"),
}

ALLOW = {
    "search in a whole-page form whose other submit is Buy now": ("""<!doctype html><title>shop</title>
<form onsubmit="document.title='SEARCHED';return false">
<header><label>Search <input name=q value="usb cable"></label><input type=submit value=Go></header>
<main>USB cable <input type=submit name=buy value="Buy now"></main></form>""", "Search"),
    "script search in a header whose sibling nav has Unsubscribe": ("""<!doctype html><title>site</title>
<header><div class=search><input id=q aria-label=Search value="browser use"></div>
<nav><button>Menu</button><button>Unsubscribe</button></nav></header>
<script>q.addEventListener('keydown',e=>{if(e.key==='Enter')document.title='SEARCHED'})</script>""", "Search"),
    "plain search form": ("""<!doctype html><title>search</title>
<form onsubmit="document.title='SEARCHED';return false">
<label>Search <input name=q value="browser use"></label><button>Search</button></form>""", "Search"),
}


@pytest.fixture(scope="module")
def manager():
    m = BrowserManager(Config(headless=True, allow_js=True))
    try:
        yield m
    finally:
        m.shutdown()


def _open(manager, name, page):
    session = manager.start(name, "data:text/html," + quote(page))
    return session, session.observe()


def _ref(observation, prefix):
    for element in observation.elements:
        if element.name.startswith(prefix):
            return element.ref
    raise AssertionError(f"no {prefix!r} in {[(e.ref, e.role, e.name) for e in observation.elements]}")


def _act(session, op):
    return session.act([op], stop_on_error=False)["ops"][0]


@pytest.mark.parametrize("case", list(BLOCK))
def test_the_rail_stops_it(manager, case):
    page, field, op = BLOCK[case]
    session, observation = _open(manager, "block-" + str(abs(hash(case))), page)
    if field is not None:
        op = {**op, "ref": _ref(observation, field)}
        if op["op"] == "keys":
            _act(session, {"op": "click", "ref": op.pop("ref")})
    step = _act(session, op)
    title = session.evaluate_js("document.title")
    assert not title.startswith(FIRED), f"{case}: the guarded control ran ({title})"
    assert step.get("error") in {"needs_confirmation", "invalid_request"}, step


@pytest.mark.parametrize("case", list(CLICK_BLOCK))
def test_the_click_rail_stops_it(manager, case):
    page, name = CLICK_BLOCK[case]
    session, observation = _open(manager, "click-" + str(abs(hash(case))), page)
    ref = (_ref(observation, name) if name
           else next(e.ref for e in observation.elements if e.role == "button"))
    step = _act(session, {"op": "click", "ref": ref})
    title = session.evaluate_js("document.title")
    assert not title.startswith(FIRED), f"{case}: the guarded control ran ({title})"
    assert step.get("error") == "needs_confirmation", step


@pytest.mark.parametrize("case", list(ALLOW))
def test_an_ordinary_search_still_submits(manager, case):
    page, field = ALLOW[case]
    session, observation = _open(manager, "allow-" + str(abs(hash(case))), page)
    step = _act(session, {"op": "type", "ref": _ref(observation, field), "text": "", "clear": False,
                          "submit": True})
    assert step.get("ok"), step
    assert session.evaluate_js("document.title") == "SEARCHED"


def test_confirm_still_lets_a_deliberate_action_through(manager):
    page, field, op = BLOCK["form whose handler presses a guarded button outside it"]
    session, observation = _open(manager, "confirmed", page)
    step = _act(session, {**op, "ref": _ref(observation, field), "confirm": True})
    assert step.get("ok"), step
    assert session.evaluate_js("document.title") == "DELETED"


@pytest.mark.parametrize("name", ["Place your order", "Pay", "Delete", "Buy", "Confirm and pay",
                                  "Cancel my subscription", "Pay\u200b now", "Ｐａｙ now", "Remove"])
def test_common_consequential_labels_need_confirmation(name):
    assert confirm_reason(Config(), name, "button"), name


# "Remove" moved to the list above: a bare Remove deletes whatever row it sits in, and a
# goal naming that row made the model click one (tests/test_goal_field_policy.py).
@pytest.mark.parametrize("name", ["Add to cart", "Archive", "Close ticket", "Continue to review",
                                  "Save and continue", "Payment methods", "PayPal",
                                  "Submit order", "Search", "Update", "Remove adult"])
def test_ordinary_labels_do_not(name):
    assert confirm_reason(Config(), name, "button") is None, name
