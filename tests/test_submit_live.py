"""SUBMIT (type + submit) against a real page: the attacks from the independent review.

Real Chrome, real observer, real `Session._run_op`, so these check what actually happens to the page
rather than a fake tab's record of what was sent.

- Enter in a field whose form holds a confirmation-rule button ("Pay now") must need `confirm`, as a
  click on that button does -- whether the page submits on its own keydown handler or by HTML's
  implicit submission.
- Enter must actually submit a plain HTML form (it used to fire keydown/keyup only, no keypress).
- SUBMIT must not retype (and so truncate) the field's content.
- A page overriding the `.type` getter must not un-mask a password field.
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


def _chrome_available() -> bool:
    try:
        find_chrome(None)
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _chrome_available(), reason="needs Chrome")

PAY_BY_SCRIPT = """<!doctype html><title>checkout</title>
<form id=f onsubmit="document.title='PAID:'+this.amount.value;return false">
<label>Amount <input name=amount value=250></label><button>Pay now</button></form>
<script>document.querySelector('[name=amount]').addEventListener('keydown',
  e => { if (e.key === 'Enter') document.getElementById('f').requestSubmit() })</script>"""

PAY_BY_HTML = """<!doctype html><title>checkout</title>
<form onsubmit="document.title='PAID:'+this.amount.value;return false">
<label>Amount <input name=amount value=250></label><button>Pay now</button></form>"""

# `window.sent` rather than the title: `document.title` collapses runs of whitespace itself.
SEARCH = """<!doctype html><title>search</title>
<form onsubmit="window.sent=this.q.value;document.title='SENT:'+this.q.value;return false">
<label>Search <input name=q></label><button>Search</button></form>"""

SPOOFED_PASSWORD = """<!doctype html><title>login</title>
<label>Token <input type=password id=p value=hunter3></label>
<script>Object.defineProperty(document.getElementById('p'), 'type', {get: () => 'text'})</script>"""


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
    return next(e for e in observation.elements if e.name.startswith(prefix)).ref


def _submit(session, ref, **extra):
    op = {"op": "type", "ref": ref, "text": "", "clear": False, "submit": True, **extra}
    return session.act([op], stop_on_error=False)["ops"][0]


@pytest.mark.parametrize("page", [PAY_BY_SCRIPT, PAY_BY_HTML], ids=["page-keydown", "html-implicit"])
def test_enter_next_to_a_confirmation_button_needs_confirming(manager, page):
    session, observation = _open(manager, "pay", page)
    step = _submit(session, _ref(observation, "Amount"))

    assert step["ok"] is False and step["error"] == "needs_confirmation", step
    assert "Pay now" in step["detail"]
    assert session.evaluate_js("document.title") == "checkout", "nothing may have been paid"


def test_confirmed_enter_does_submit(manager):
    session, observation = _open(manager, "pay-confirmed", PAY_BY_HTML)
    step = _submit(session, _ref(observation, "Amount"), confirm=True)

    assert step["ok"] is True, step
    assert session.evaluate_js("document.title") == "PAID:250"


def test_enter_submits_a_plain_html_form(manager):
    session, observation = _open(manager, "search", SEARCH)
    ref = _ref(observation, "Search")
    session.act([{"op": "type", "ref": ref, "text": "browser use"}], stop_on_error=False)
    step = _submit(session, ref)

    assert step["ok"] is True, step
    assert session.evaluate_js("document.title") == "SENT:browser use"


def test_submit_does_not_rewrite_a_long_value(manager):
    long_value = "q" * 171 + "  two  spaces"
    session, observation = _open(manager, "long", SEARCH)
    ref = _ref(observation, "Search")
    session.act([{"op": "type", "ref": ref, "text": long_value}], stop_on_error=False)
    _submit(session, ref)

    assert session.evaluate_js("window.sent") == long_value


def test_a_spoofed_type_getter_does_not_unmask_a_password(manager):
    _, observation = _open(manager, "spoof", SPOOFED_PASSWORD)
    field = next(e for e in observation.elements if e.name.startswith("Token"))

    assert field.secret is True
    assert field.value == ""


def test_enter_is_refused_when_the_page_will_not_say_what_it_submits(manager):
    """A helper that cannot answer (ref gone, helper replaced by the page) is refused, not trusted."""
    session, observation = _open(manager, "unknown", PAY_BY_HTML)
    ref = _ref(observation, "Amount")
    session.evaluate_js("window.__jevMcp.submitters = () => 'no'")
    step = _submit(session, ref)

    assert step["ok"] is False and step["error"] == "needs_confirmation", step
    assert "cannot tell" in step["detail"]
    assert session.evaluate_js("document.title") == "checkout"
