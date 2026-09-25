"""A link drawn as a `pointer-events: none` overlay over its own row is reachable, not covered.

Measured on Google Flights' results (2026-09-25): each row is

    <div class=gQ6yfe>
      <div role=link aria-label="From 1015 US dollars total. Nonstop ..." style="position:absolute;
           pointer-events:none"></div>          <- the accessible name, no children
      <div jsaction="click:..." >...visible row...</div>   <- the real click target, its sibling
    </div>

`elementFromPoint` at the link's centre returns the sibling's content, so the observer flagged every
result `occluded` and the policy never offered one: the model could not pick a flight. A real mouse
click at that point runs the page's handler. The rule: an element that cannot itself be hit
(`pointer-events: none`) is reachable when the hit lies inside its own parent -- its own row -- and
still covered when anything else (a modal, a banner) is on top. The click rails still see what the
click lands on. Real Chrome, real observer, real `Session.act`.
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


def _row(i: int, name: str, sibling: str) -> str:
    return (f'<li><div class=cell style="position:relative;height:60px;margin:4px 0">'
            f'<div role=link tabindex=0 aria-label="{name}" style="position:absolute;inset:0;'
            f'pointer-events:none"></div>{sibling}</div></li>')


def _flight(i: int) -> str:
    sibling = (f'<div class=body onclick="document.title=\'PICKED {i}\'" style="height:60px;'
               f'pointer-events:auto;background:#eef"><span>Korean Air</span> <b>${1000 + i}</b></div>')
    return _row(i, f"From {1000 + i} US dollars total. Nonstop flight with Korean Air.", sibling)


RESULTS = ("<!doctype html><title>results</title><style>body{margin:0}ul{list-style:none;margin:0;"
           "padding:0}</style><ul>" + "".join(_flight(i) for i in range(4)) + "</ul>")

# The same rows under a genuine modal: the hit is the modal, which is not inside the row.
COVERED = RESULTS.replace("</ul>", "</ul><div role=dialog style=\"position:fixed;inset:0;"
                          "background:rgba(0,0,0,.5)\"><button>Accept cookies</button></div>")

# A pass-through overlay named "Continue" over a sibling button named "Delete account".
RAIL = ("<!doctype html><title>settings</title><style>body{margin:0}</style>" + _row(
    0, "Continue",
    '<button onclick="document.title=\'DELETED\'" style="display:block;width:100%;height:60px">'
    "Delete account</button>") + "")


@pytest.fixture(scope="module")
def manager():
    m = BrowserManager(Config(headless=True, allow_js=True))
    try:
        yield m
    finally:
        m.shutdown()


def _links(observation):
    return [e for e in observation.elements if e.role == "link"]


def test_overlay_links_over_their_own_rows_are_reachable_and_clickable(manager):
    session = manager.start("passthrough", "data:text/html," + quote(RESULTS))
    observation = session.observe()
    links = _links(observation)
    assert len(links) == 4, [(e.ref, e.role, e.name) for e in observation.elements]
    assert not any(e.occluded for e in links), [(e.name, e.occluded) for e in links]

    step = session.act([{"op": "click", "ref": links[2].ref}], stop_on_error=False)["ops"][0]
    assert step.get("ok"), step
    assert session.evaluate_js("document.title") == "PICKED 2"


def test_a_real_modal_over_such_a_row_still_covers_it(manager):
    session = manager.start("passthrough-covered", "data:text/html," + quote(COVERED))
    observation = session.observe()
    links = _links(observation)
    assert links and all(e.occluded for e in links), [(e.name, e.occluded) for e in links]

    step = session.act([{"op": "click", "ref": links[0].ref}], stop_on_error=False)["ops"][0]
    assert not step.get("ok"), step
    assert "occluded" in str(step.get("error", "")) + str(step.get("detail", "")), step
    assert session.evaluate_js("document.title") == "results"


# A pass-through overlay whose parent is the page itself: "inside its parent" would be "anywhere",
# so it gets no pass -- here a modal covers it, and it must read as covered.
BODY_CHILD = ("<!doctype html><title>bare</title><style>body{margin:0}</style>"
              '<div role=link tabindex=0 aria-label="Open offer" style="position:absolute;top:0;left:0;'
              'width:300px;height:60px;pointer-events:none"></div>'
              '<div role=dialog style="position:fixed;inset:0;background:rgba(0,0,0,.5)">'
              "<button>Accept cookies</button></div>")


def test_an_overlay_whose_parent_is_the_page_gets_no_pass(manager):
    session = manager.start("passthrough-body", "data:text/html," + quote(BODY_CHILD))
    link = next(e for e in session.observe().elements if e.name == "Open offer")
    assert link.occluded


# An ordinary link (it takes clicks itself) under a sibling popover in the same row: covered.
# The pass is for elements that cannot be hit, not for anything with a busy row.
SIBLING_COVER = ("<!doctype html><title>row</title><style>body{margin:0}</style>"
                 '<div class=row style="position:relative;height:60px"><a href="#x" style="display:block;'
                 'height:60px">Details</a><div class=pop style="position:absolute;inset:0;'
                 'background:#fff">Popover</div></div>')


def test_a_clickable_link_covered_inside_its_own_row_is_still_covered(manager):
    session = manager.start("passthrough-sibling", "data:text/html," + quote(SIBLING_COVER))
    link = next(e for e in session.observe().elements if e.name == "Details")
    assert link.occluded


def test_an_overlay_named_continue_over_delete_account_needs_confirmation(manager):
    session = manager.start("passthrough-rail", "data:text/html," + quote(RAIL))
    observation = session.observe()
    link = next(e for e in observation.elements if e.name == "Continue")
    assert not link.occluded

    step = session.act([{"op": "click", "ref": link.ref}], stop_on_error=False)["ops"][0]
    assert step.get("error") == "needs_confirmation", step
    # Refused before anything was dispatched (the rail read what lies under the overlay), not only
    # caught by the tripwire afterwards: a dry run must say the same.
    assert "tried to press" not in step.get("detail", ""), step
    dry = session.act([{"op": "click", "ref": link.ref}], dry_run=True, stop_on_error=False)["ops"][0]
    assert dry.get("error") == "needs_confirmation", dry
    assert session.evaluate_js("document.title") == "settings"


def test_the_tripwire_alone_stops_the_click_the_overlay_passes_through(manager):
    """Without the pre-dispatch check, the click at the overlay's centre lands on "Delete account"
    and the page-level tripwire cancels it. Two layers, each tested on its own."""
    session = manager.start("passthrough-wire", "data:text/html," + quote(RAIL))
    observation = session.observe()
    link = next(e for e in observation.elements if e.name == "Continue")
    with session._armed(False) as wire:
        session._do_click(link.ref)
    assert wire.hits and "Delete account" in wire.hits[0], wire.hits
    assert session.evaluate_js("document.title") == "settings"


def test_a_confirmed_click_through_the_overlay_reaches_the_button(manager):
    session = manager.start("passthrough-confirmed", "data:text/html," + quote(RAIL))
    observation = session.observe()
    link = next(e for e in observation.elements if e.name == "Continue")
    step = session.act([{"op": "click", "ref": link.ref, "confirm": True}], stop_on_error=False)["ops"][0]
    assert step.get("ok"), step
    assert session.evaluate_js("document.title") == "DELETED"
