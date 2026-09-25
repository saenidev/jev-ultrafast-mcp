"""`click_best`: choose among matching elements by a number in their names, with no model at all.

Measured on live Google Flights (2026-09-26): the planner's "open the cheapest flight" subgoal
failed twice. Jev was asked in words to find the lowest price among 250 elements -- a comparison
that a decision model makes badly and code makes exactly. `click_best` parses a number out of each
matching element's name and clicks the minimum (or maximum), through the same click path and the
same rails as `click`: a covered element is never a candidate, and a name matching a confirmation
rule still needs `"confirm": true`. Real Chrome, real observer, real `Session.act`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_ultrafast_mcp import server  # noqa: E402
from jev_ultrafast_mcp.browser import BrowserManager  # noqa: E402
from jev_ultrafast_mcp.config import Config, find_chrome  # noqa: E402


def _chrome_available() -> bool:
    try:
        find_chrome(None)
        return True
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _chrome_available(), reason="needs Chrome")

PRICE = r"From ([\d,]+) US dollars"


def _flight(i: int, price: str, lead: str = "From") -> str:
    return (f'<a href="#" style="display:block;height:40px;line-height:40px" '
            f'onclick="document.title=\'PICKED {i}\';return false" '
            f'aria-label="{lead} {price} US dollars. 1 stop flight with Korean Air. Leaves '
            f'Suvarnabhumi Airport at 10:{i}0 PM and arrives at Incheon at 9:25 AM.">'
            f"Korean Air {price}</a>")


def _page(flights: str, extra: str = "") -> str:
    return ("<!doctype html><title>results</title><style>body{margin:0}</style>"
            '<button type=button onclick="document.title=\'SORTED\'">Sort by price</button>'
            + flights + extra)


# Six flights. The cheapest (row 2, $150) sits under a price-alert popover that covers exactly its
# row, so it is not a candidate; the cheapest reachable one is row 4 at $198, tied with row 5 (the
# first in document order wins). Prices use thousands separators.
PRICES = ["1,215", "480", "150", "905", "198", "198"]
COVERED = _page("".join(_flight(i, p) for i, p in enumerate(PRICES)),
                '<div role=dialog style="position:absolute;left:0;right:0;top:101px;height:40px;'
                'background:#fff">Track prices for this route</div>')

# The cheapest flight's control is named like a purchase: the click rail stops it.
BUY = _page(_flight(0, "480") + '<a href="#" style="display:block;height:40px" '
            "onclick=\"document.title='BOUGHT';return false\" "
            'aria-label="Buy now: From 99 US dollars. Nonstop">Buy $99</a>' + _flight(2, "305"))


@pytest.fixture(scope="module")
def manager():
    m = BrowserManager(Config(headless=True, allow_js=True))
    try:
        yield m
    finally:
        m.shutdown()


def _best(session, **spec) -> dict:
    op = {"op": "click_best", "role": "link", "number_regex": PRICE, **spec}
    return session.act([op], stop_on_error=False)["ops"][0]


def test_the_cheapest_reachable_flight_is_clicked(manager):
    session = manager.start("best-min", "data:text/html," + quote(COVERED))
    observation = session.observe()
    covered = [e for e in observation.elements if e.role == "link" and e.occluded]
    assert len(covered) == 1 and "150" in covered[0].name, [(e.name, e.occluded)
                                                            for e in observation.elements]

    step = _best(session, name_regex="US dollars", key="min_number")

    assert step["ok"], step
    assert step["op"] == "click_best"
    assert session.evaluate_js("document.title") == "PICKED 4", "the first of the two $198 flights"
    assert "198" in step["detail"] and "5 candidates" in step["detail"], step
    assert "1 covered" in step["detail"], step
    assert "From 198 US dollars" in step["target"], step


def test_max_number_parses_thousands_separators(manager):
    session = manager.start("best-max", "data:text/html," + quote(COVERED))
    session.observe()

    step = _best(session, name_regex="Korean Air", key="max_number")

    assert step["ok"], step
    assert session.evaluate_js("document.title") == "PICKED 0"
    assert "1215" in step["detail"].replace(",", ""), step


def test_a_consequential_name_still_needs_confirmation(manager):
    session = manager.start("best-buy", "data:text/html," + quote(BUY))
    session.observe()

    step = _best(session, name_regex="US dollars")

    assert not step["ok"] and step["error"] == "needs_confirmation", step
    assert "99" in step["detail"], step
    assert session.evaluate_js("document.title") == "results", "nothing was clicked"

    session.observe()
    confirmed = _best(session, name_regex="US dollars", confirm=True)
    assert confirmed["ok"], confirmed
    assert session.evaluate_js("document.title") == "BOUGHT"


def test_no_candidate_is_a_refusal_not_a_click(manager):
    session = manager.start("best-none", "data:text/html," + quote(COVERED))
    session.observe()

    step = _best(session, name_regex="Lufthansa")

    assert not step["ok"] and step["error"] == "no_candidates", step
    assert session.evaluate_js("document.title") == "results"


@pytest.mark.parametrize("spec", [
    {"number_regex": "("},                      # not a regex
    {"key": "median"},                          # not a key
    {"number_regex": ""},                       # nothing to parse
])
def test_a_malformed_spec_is_an_invalid_request(manager, spec):
    session = manager.start("best-bad", "data:text/html," + quote(COVERED))
    session.observe()

    step = _best(session, **{"name_regex": "US dollars", **spec})

    assert not step["ok"] and step["error"] == "invalid_request", step
    assert session.evaluate_js("document.title") == "results"


def test_dry_run_reports_the_choice_and_clicks_nothing(manager):
    session = manager.start("best-dry", "data:text/html," + quote(COVERED))
    session.observe()

    payload = session.act([{"op": "click_best", "role": "link", "name_regex": "US dollars",
                            "number_regex": PRICE}], dry_run=True)

    step = payload["ops"][0]
    assert step["ok"] and "198" in step["detail"], step
    assert session.evaluate_js("document.title") == "results"


def test_run_click_best_is_the_planner_entry_point(manager, monkeypatch):
    session = manager.start("best-planner", "data:text/html," + quote(COVERED))
    monkeypatch.setattr(server, "_session", lambda _name: session)

    out = server.run_click_best("best-planner", {"role": "link", "name_regex": "US dollars",
                                                 "key": "min_number", "number_regex": PRICE})

    first = out.splitlines()[0]
    assert first.startswith("click_best: ok ref=e"), out
    assert "number=198" in first and "candidates=5" in first, first
    assert session.evaluate_js("document.title") == "PICKED 4"


def test_run_click_best_reports_a_refusal_on_its_first_line(manager, monkeypatch):
    session = manager.start("best-planner-buy", "data:text/html," + quote(BUY))
    monkeypatch.setattr(server, "_session", lambda _name: session)

    out = server.run_click_best("best-planner-buy", {"role": "link", "name_regex": "US dollars",
                                                     "number_regex": PRICE})

    assert out.splitlines()[0].startswith("click_best: failed error=needs_confirmation"), out
    assert session.evaluate_js("document.title") == "results"


def test_click_best_is_documented_in_browser_act():
    doc = server.browser_act.__doc__ or ""
    assert "click_best" in doc and "min_number" in doc and "number_regex" in doc


def test_candidates_beyond_the_table_cap_are_still_compared(manager):
    """Measured on Google Flights' full results: 243 'Flight details' buttons outranked the result
    links for the 250-element table, every link was omitted, and the pick found no candidates."""
    buttons = "".join(f'<button type=button>Flight details {i}</button>' for i in range(300))
    flights = "".join(_flight(i, p) for i, p in enumerate(["540", "377", "1,015", "198", "260"]))
    session = manager.start("best-cap", "data:text/html," + quote(_page(buttons + flights)))
    observation = session.observe()
    assert observation.omitted, "the page must overflow the table for this test to mean anything"
    assert not any("US dollars" in e.name for e in observation.elements), "links must be omitted"

    step = _best(session, name_regex="^From [0-9,]+ US dollars", key="min_number")

    assert step["ok"], step
    assert session.evaluate_js("document.title") == "PICKED 3", step
    assert "5 candidates" in step["detail"], step


def test_the_pick_waits_for_results_that_are_still_loading(manager):
    """Measured on Google Flights multi-city: the pick ran the instant Search (or the previous
    leg's pick) was clicked, found 0 candidates while results rendered, and the planner spent two
    extra subgoals (and two extra Opus calls) waiting. The pick waits for its candidates."""
    later = ("<script>setTimeout(() => { document.body.insertAdjacentHTML('beforeend', "
             + repr("".join(_flight(i, p) for i, p in enumerate(["540", "198", "260"])))
             + "); }, 1500);</script>")
    session = manager.start("best-late", "data:text/html," + quote(_page("", later)))
    session.observe()

    step = _best(session, name_regex="^From [0-9,]+ US dollars", key="min_number")

    assert step["ok"], step
    assert session.evaluate_js("document.title") == "PICKED 1", step
    assert "3 candidates" in step["detail"], step


def test_a_pick_with_nothing_to_wait_for_gives_up_within_its_wait(manager):
    import time
    session = manager.start("best-none-wait", "data:text/html," + quote(COVERED))
    session.observe()
    started = time.monotonic()

    step = _best(session, name_regex="Lufthansa", wait_s=1)

    assert not step["ok"] and step["error"] == "no_candidates", step
    assert time.monotonic() - started < 4, "a no-candidate pick must not hang"
