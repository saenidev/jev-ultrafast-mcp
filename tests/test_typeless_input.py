"""Text fields the page does not label with a `type` must still be typeable.

HTML's default input type is `text`: `<input name="custname">` *is* a text box, and so is
`<input type="">` or any type the browser does not recognise (the spec falls back to the text
state). The observer used to read the raw `type` attribute and only accept an explicit list,
so a type-less input got no role, was not editable, and was never offered to TYPE_TEXT. Measured
on httpbin's pizza form (`<label>Customer name: <input name="custname"></label>`): the customer
name box was missing from the table, the model typed the name into Telephone instead, and the
goal stalled.

These run the real observer in a real browser, so they check what the page reports rather than
a copy of the rule.
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


PAGE = """<!doctype html><title>t</title>
<label>Customer name: <input name="custname"></label>
<label>Empty type: <input type="" name="blank"></label>
<label>Unknown type: <input type="bogus" name="odd"></label>
<label>Upper case: <input type="EMAIL" name="upper"></label>
<label>Telephone: <input type="tel" name="tel"></label>
<label>Secret: <input type="password" name="pw"></label>
<label>Hidden: <input type="hidden" name="h" value="x"></label>
<label><input type="checkbox" name="c"> Tick</label>
<input type="submit" value="Go">
"""


@pytest.fixture(scope="module")
def observation():
    manager = BrowserManager(Config(headless=True))
    try:
        session = manager.start("typeless", "data:text/html," + quote(PAGE))
        yield session.observe()
    finally:
        manager.shutdown()


def _by_name(observation, name):
    return next((e for e in observation.elements if e.name.startswith(name)), None)


pytestmark = pytest.mark.skipif(not _chrome_available(), reason="no Chromium-family browser found")


@pytest.mark.parametrize("name", ["Customer name", "Empty type", "Unknown type", "Upper case", "Telephone"])
def test_text_state_inputs_are_editable_textboxes(observation, name):
    element = _by_name(observation, name)
    assert element is not None, f"{name!r} missing from the table: {[e.name for e in observation.elements]}"
    assert element.role == "textbox"
    assert element.editable
    assert "TYPE_TEXT" in element.target_kinds()


def test_a_password_stays_a_secret_textbox(observation):
    element = _by_name(observation, "Secret")
    assert element is not None and element.role == "textbox" and element.secret


def test_non_text_inputs_keep_their_own_roles(observation):
    assert _by_name(observation, "Tick").role == "checkbox"
    assert _by_name(observation, "Go").role == "button"
    assert _by_name(observation, "Hidden") is None
