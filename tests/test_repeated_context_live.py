"""Repeated controls have to be told apart by their `context`, including inputs that have no text.

Measured on Google Flights' multi-city form (2026-09-25): three or four rows, each

    <div row>
      <div><div><input role=combobox aria-label="Where from?" value="Bangkok BKK"></div></div>
      <button aria-label="Swap origin and destination."></button>
      <div><div><input role=combobox aria-label="Where to?" value="Seoul ICN"></div></div>
      <div><div><input aria-label="Departure" value="Sun, Oct 11"></div></div>
      <button aria-label="Remove flight from Bangkok to Seoul on Sun, Oct 11"></button>
    </div>

Context used to be the text of the row/card/form the control sits in, else its parent. None of the
rows is a `tr`/`li`, the parent is the input's own wrapper, and an input's value is not part of any
`innerText`, so all three "Departure" boxes read context '' -- the model could not tell which date
belonged to which flight, set one, and clicked calendar days forever.

The rule now: when that text is empty or shared with a same-named control, walk up (at most a few
levels) to the largest ancestor that holds this control and no other of the same name, and describe
it by its fields' values (secret ones never) and its text. When nothing distinguishes a control,
its position does: "(2 of 3)". A row scope that already tells controls apart (a mail list's `li`)
is kept exactly as it was. Real Chrome, real observer.
"""

from __future__ import annotations

import sys
import time
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

LEGS = [("Bangkok BKK", "Seoul ICN", "Sun, Oct 11"),
        ("Seoul ICN", "Osaka KIX", "Thu, Oct 15"),
        ("Osaka KIX", "Bangkok BKK", "Tue, Oct 20")]


def _leg(src: str, dst: str, date: str) -> str:
    wrap = '<div class=cell><div class=wrap>{}</div></div>'
    return ('<div class=row style="display:flex;gap:4px;margin:6px 0">'
            + wrap.format(f'<input role=combobox aria-label="Where from?" value="{src}">')
            + '<button aria-label="Swap origin and destination."><svg width=10 height=10></svg></button>'
            + wrap.format(f'<input role=combobox aria-label="Where to?" value="{dst}">')
            + wrap.format(f'<input aria-label="Departure" value="{date}">')
            + f'<button aria-label="Remove flight from {src.split()[0]} to {dst.split()[0]} on {date}">'
            + '<svg width=10 height=10></svg></button></div>')


def _flights(legs) -> str:
    return ("<!doctype html><title>multi-city</title><form onsubmit='return false'>"
            + "".join(_leg(*leg) for leg in legs)
            + "<button type=button>Add flight</button> <button type=button>Search</button></form>")


FLIGHTS = _flights(LEGS)

# Nothing tells these apart but where they are.
BLANK = ("<!doctype html><title>notes</title>"
         + "".join('<div class=row><div class=wrap><input aria-label="Note"></div></div>' for _ in range(3)))

# A mail list: each row's Star/Archive buttons in their own tiny form. The `li` is what separates
# them, and that context must not change.
MAIL = ("<!doctype html><title>inbox</title><ul>" + "".join(
    f"<li><span>{who}</span> <span>{subject}</span><form><button>Star</button>"
    f"<button>Archive</button></form></li>"
    for who, subject in [("Ann", "Invoice 12"), ("Bob", "Lunch Friday"), ("Cy", "Build failed")])
    + "</ul>")

# Rows whose only distinguishing values are secrets: those must never reach the context.
SECRETS = ("<!doctype html><title>accounts</title>" + "".join(
    f'<div class=row><div class=w><input aria-label="Account" value="acct"></div>'
    f'<div class=w><input type=password aria-label="Key" value="hunter{i}"></div>'
    f'<div class=w><input aria-label="PIN" value="98{i}7"></div>'
    f'<div class=w><div contenteditable=true aria-label="Card number">41111111{i}</div></div></div>'
    for i in range(3)))


@pytest.fixture(scope="module")
def manager():
    m = BrowserManager(Config(headless=True, allow_js=True))
    try:
        yield m
    finally:
        m.shutdown()


def _raw(manager, name, html):
    session = manager.start(name, "data:text/html," + quote(html))
    session.observe()  # injects the helper
    return session, session._read_state(include_text=False)


def _contexts(raw, role, name):
    return [a.get("context") for a in raw["actions"] if a["role"] == role and a["name"] == name]


def test_departure_boxes_are_told_apart_by_their_rows_values(manager):
    _, raw = _raw(manager, "ctx-flights", FLIGHTS)
    departures = _contexts(raw, "textbox", "Departure")
    assert departures == [
        "Where from? Bangkok BKK | Where to? Seoul ICN",
        "Where from? Seoul ICN | Where to? Osaka KIX",
        "Where from? Osaka KIX | Where to? Bangkok BKK",
    ], departures
    origins = _contexts(raw, "combobox", "Where from?")
    assert origins == [
        "Where to? Seoul ICN | Departure Sun, Oct 11",
        "Where to? Osaka KIX | Departure Thu, Oct 15",
        "Where to? Bangkok BKK | Departure Tue, Oct 20",
    ], origins
    swaps = _contexts(raw, "button", "Swap origin and destination.")
    assert len(set(swaps)) == 3 and all(len(c) <= 120 for c in swaps), swaps
    assert "Osaka KIX" in swaps[1] and "Thu, Oct 15" in swaps[1], swaps


def test_the_rendered_table_shows_that_context(manager):
    session = manager.start("ctx-flights-table", "data:text/html," + quote(FLIGHTS))
    departures = [e for e in session.observe().elements if e.name == "Departure"]
    assert [e.context for e in departures][1] == "Where from? Seoul ICN | Where to? Osaka KIX"


def test_a_group_with_nothing_to_tell_it_apart_gets_its_position(manager):
    _, raw = _raw(manager, "ctx-blank", BLANK)
    assert _contexts(raw, "textbox", "Note") == ["(1 of 3)", "(2 of 3)", "(3 of 3)"]


def test_same_row_text_everywhere_keeps_the_text_and_adds_the_position(manager):
    html = ("<!doctype html><title>x</title>"
            + "".join("<div class=r><span>Qty</span><button>+</button></div>" for _ in range(2)))
    _, raw = _raw(manager, "ctx-same", html)
    # The widened row, less the control's own name (which the table already shows).
    assert _contexts(raw, "button", "+") == ["Qty (1 of 2)", "Qty (2 of 2)"]


def test_siblings_under_one_parent_get_their_position(manager):
    # Nothing holds one without the other, so there is no row to widen to.
    html = "<!doctype html><title>x</title><div><span>Size</span><button>Pick</button><button>Pick</button></div>"
    _, raw = _raw(manager, "ctx-siblings", html)
    assert _contexts(raw, "button", "Pick") == ["SizePickPick (1 of 2)", "SizePickPick (2 of 2)"]


def test_a_one_letter_name_is_cut_as_a_word_not_out_of_words(manager):
    html = ("<!doctype html><title>x</title>"
            + "".join(f"<div class=r><span>{w} a</span> <span class=w><button>a</button></span></div>"
                      for w in ["banana", "papaya"]))
    _, raw = _raw(manager, "ctx-word", html)
    assert _contexts(raw, "button", "a") == ["banana a", "papaya a"]


def test_a_mail_rows_context_is_its_row_as_before(manager):
    _, raw = _raw(manager, "ctx-mail", MAIL)
    assert _contexts(raw, "button", "Star") == [
        "Ann Invoice 12 StarArchive", "Bob Lunch Friday StarArchive", "Cy Build failed StarArchive"]


def test_secret_values_never_reach_the_context(manager):
    _, raw = _raw(manager, "ctx-secret", SECRETS)
    accounts = _contexts(raw, "textbox", "Account")
    assert len(accounts) == 3, raw["actions"]
    # Every repeated control's context -- the secret fields' own included.
    blob = " ".join(a.get("context") or "" for a in raw["actions"])
    assert "context" in raw["actions"][0]
    for i in range(3):
        assert f"hunter{i}" not in blob and f"98{i}7" not in blob and f"41111111{i}" not in blob, blob
    # Nothing left to tell them apart but position.
    assert accounts == ["(1 of 3)", "(2 of 3)", "(3 of 3)"], accounts


def test_the_walk_stops_below_an_ancestor_shared_with_a_same_named_control(manager):
    # Two rows in one card: the card's other fields belong to both, so they must not be used.
    html = ("<!doctype html><title>x</title><div class=card><input aria-label='Trip' value='Summer'>"
            + "".join(f"<div class=r><div class=w><input aria-label='City' value='{c}'></div>"
                      f"<div class=w><input aria-label='Date' value='{d}'></div></div>"
                      for c, d in [("Rome", "May 1"), ("Oslo", "May 9")]) + "</div>")
    _, raw = _raw(manager, "ctx-shared", html)
    assert _contexts(raw, "textbox", "Date") == ["City Rome", "City Oslo"]


def test_four_hundred_repeated_rows_read_in_bounded_time(manager):
    legs = [(f"City{i} AAA", f"Town{i} BBB", f"Day {i}") for i in range(400)]
    session, raw = _raw(manager, "ctx-perf", _flights(legs))
    kept = _contexts(raw, "textbox", "Departure")
    assert kept and len(set(kept)) == len(kept), kept[:5]
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        session._read_state(include_text=False)
        timings.append(time.perf_counter() - start)
    print(f"400-row read: {[round(t * 1000) for t in timings]} ms")
    assert min(timings) < 2.0, timings
