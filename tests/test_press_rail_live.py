"""The Enter / Space rail, against real pages: the layouts the second independent review found.

Each page submits (or presses a guarded button) only if a key press gets past the rail, and the
page's title says so. Bypass pages must come back `needs_confirmation` with the title untouched;
ordinary pages must submit. Real Chrome, real observer, real `Session._run_op`.
"""

from __future__ import annotations

import json
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

# ---- layouts where Enter in the field must need `confirm` (a guarded button would be pressed)

BYPASS = {
    # Bootstrap/React modal: the form holds the field, the footer holds the button, outside it.
    "dialog footer outside the form": ("Repository name", """<!doctype html><title>settings</title>
<div class=modal role=dialog aria-label="Delete">
 <form id=x onsubmit="document.title='DELETED';return false"><label>Repository name <input name=n value=myrepo></label></form>
 <div class=footer><button onclick="document.getElementById('x').requestSubmit()">Delete repository</button></div>
</div>"""),
    # No form, no dialog: a wrapper whose page handler presses the button beside the field.
    "no form or dialog, page handler": ("Repository name", """<!doctype html><title>settings</title>
<div class=modal>
 <label>Repository name <input id=n value=myrepo></label>
 <button id=d onclick="document.title='DELETED'">Delete repository</button>
</div>
<script>n.addEventListener('keydown',e=>{if(e.key==='Enter')d.click()})</script>"""),
    # A web-component field inside a light-DOM form.
    "shadow-DOM field in a form": ("Amount", """<!doctype html><title>checkout</title>
<form id=f onsubmit="document.title='PAID';return false"><x-in></x-in><button>Pay now</button></form>
<script>customElements.define('x-in',class extends HTMLElement{constructor(){super();
const r=this.attachShadow({mode:'open'});r.innerHTML='<label>Amount <input value=250></label>';
r.querySelector('input').addEventListener('keydown',e=>{if(e.key==='Enter')document.getElementById('f').requestSubmit()})}})</script>"""),
    # A design-system button: the real <button> is in the custom element's shadow root.
    "custom-element submit button": ("Amount", """<!doctype html><title>checkout</title>
<form id=f onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<x-btn>Pay now</x-btn></form>
<script>customElements.define('x-btn',class extends HTMLElement{constructor(){super();
const r=this.attachShadow({mode:'open'});r.innerHTML='<button><slot></slot></button>';
r.querySelector('button').onclick=()=>this.closest('form').requestSubmit()}})</script>
<script>document.querySelector('[name=amount]').addEventListener('keydown',
  e => { if (e.key === 'Enter') document.querySelector('x-btn').shadowRoot.querySelector('button').click() })</script>"""),
    # Says "Pay now" on screen, "Continue" to assistive tech.
    "submit value differs from aria-label": ("Amount", """<!doctype html><title>checkout</title>
<form onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<input type=submit value="Pay now" aria-label="Continue"></form>"""),
}

# ---- layouts where Enter must go through: nothing guarded is what Enter would press

ORDINARY = {
    # ASP.NET WebForms: one form wraps the whole page, with unrelated type=button actions.
    "whole-page form with unrelated type=button actions": ("Search", """<!doctype html><title>shop</title>
<form id=aspnetForm onsubmit="document.title='SEARCHED';return false">
<header><label>Search <input name=q value="usb cable"></label><button>Go</button></header>
<main><article>USB cable <button type=button onclick="document.title='BOUGHT'">Buy now</button></article></main>
<footer>Newsletter <button type=button>Unsubscribe</button></footer></form>"""),
    # A hidden control cannot be pressed by Enter.
    "hidden guarded button": ("Search", """<!doctype html><title>s</title>
<form onsubmit="document.title='SEARCHED';return false"><label>Search <input name=q value="x"></label>
<button>Search</button><button type=button style="display:none">Withdraw</button></form>"""),
    # A disabled one neither.
    "disabled guarded button": ("Search", """<!doctype html><title>s</title>
<form onsubmit="document.title='SEARCHED';return false"><label>Search <input name=q value="x"></label>
<button>Search</button><button disabled>Pay now</button></form>"""),
    # A guarded *link* is not something Enter in a field presses.
    "guarded link beside the form": ("Search", """<!doctype html><title>shop</title>
<form onsubmit="document.title='SEARCHED';return false"><label>Search <input name=q value="usb"></label>
<button>Go</button><a href="#u">Unsubscribe</a></form>"""),
}


@pytest.fixture(scope="module")
def manager():
    mgr = BrowserManager(Config(headless=True, allow_js=True))
    yield mgr
    mgr.shutdown()


def _field(session, prefix):
    observation = session.observe()
    element = next((e for e in observation.elements if e.name.startswith(prefix)), None)
    assert element is not None, [e.name for e in observation.elements]
    return element


def _submit(session, ref):
    return session.act([{"op": "type", "ref": ref, "text": "", "clear": False, "submit": True}],
                       stop_on_error=False)["ops"][0]


@pytest.mark.parametrize("case", sorted(BYPASS))
def test_enter_that_would_press_a_guarded_button_needs_confirm(manager, case):
    prefix, page = BYPASS[case]
    session = manager.start("bypass-" + case.replace(" ", "-")[:30], "data:text/html," + quote(page))
    title = session.evaluate_js("document.title")
    step = _submit(session, _field(session, prefix).ref)
    assert step["error"] == "needs_confirmation", step
    assert session.evaluate_js("document.title") == title, "the page must not have been submitted"


@pytest.mark.parametrize("case", sorted(ORDINARY))
def test_enter_that_presses_nothing_guarded_goes_through(manager, case):
    prefix, page = ORDINARY[case]
    session = manager.start("ordinary-" + case.replace(" ", "-")[:30], "data:text/html," + quote(page))
    step = _submit(session, _field(session, prefix).ref)
    assert step["ok"], step
    assert session.evaluate_js("document.title") == "SEARCHED"


PAY = """<!doctype html><title>checkout</title>
<form onsubmit="document.title='PAID';return false"><label>Amount <input name=amount value=250></label>
<button>Pay now</button></form>"""


def test_keys_enter_in_a_field_answers_to_the_rail(manager):
    session = manager.start("keys-enter", "data:text/html," + quote(PAY))
    session.act([{"op": "click", "ref": _field(session, "Amount").ref}])
    step = session.act([{"op": "keys", "keys": "Enter"}], stop_on_error=False)["ops"][0]
    assert step["error"] == "needs_confirmation", step
    assert session.evaluate_js("document.title") == "checkout"


def test_keys_tab_then_space_onto_a_guarded_button_answers_to_the_rail(manager):
    session = manager.start("keys-space", "data:text/html," + quote(PAY))
    session.act([{"op": "click", "ref": _field(session, "Amount").ref}])
    session.act([{"op": "keys", "keys": "Tab"}])
    step = session.act([{"op": "keys", "keys": "Space"}], stop_on_error=False)["ops"][0]
    assert step["error"] == "needs_confirmation", step
    assert session.evaluate_js("document.title") == "checkout"


def test_keys_enter_confirmed_goes_through(manager):
    session = manager.start("keys-confirmed", "data:text/html," + quote(PAY))
    session.act([{"op": "click", "ref": _field(session, "Amount").ref}])
    step = session.act([{"op": "keys", "keys": "Enter", "confirm": True}], stop_on_error=False)["ops"][0]
    assert step["ok"], step
    assert session.evaluate_js("document.title") == "PAID"


def test_slow_typing_a_line_break_is_an_enter(manager):
    session = manager.start("slow-cr", "data:text/html," + quote(PAY))
    ref = _field(session, "Amount").ref
    step = session.act([{"op": "type", "ref": ref, "text": "1\r", "slow": True}],
                       stop_on_error=False)["ops"][0]
    assert step["error"] == "needs_confirmation", step
    assert session.evaluate_js("document.title") == "checkout"


def test_click_rail_reads_what_the_button_shows(manager):
    page = BYPASS["submit value differs from aria-label"][1]
    session = manager.start("click-value", "data:text/html," + quote(page))
    button = next(e for e in session.observe().elements if e.role == "button")
    step = session.act([{"op": "click", "ref": button.ref}], stop_on_error=False)["ops"][0]
    assert step["error"] == "needs_confirmation", step
    assert session.evaluate_js("document.title") == "checkout"


SPOOF_BOTH = """<!doctype html><title>login</title>
<label>Token <input type=password id=p value=hunter3></label>
<script>const p = document.getElementById('p');
Object.defineProperty(p, 'type', {get: () => 'text'});
p.getAttribute = n => n === 'type' ? 'text' : Element.prototype.getAttribute.call(p, n);</script>"""


def test_overriding_type_and_getattribute_still_masks(manager):
    session = manager.start("spoof-both", "data:text/html," + quote(SPOOF_BOTH))
    field = _field(session, "Token")
    assert field.secret, "a password field must stay secret"
    assert field.value != "hunter3", "its value must be masked"
