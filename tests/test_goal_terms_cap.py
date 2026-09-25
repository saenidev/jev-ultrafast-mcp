"""A goal's words keep the controls it names inside the observer's cap.

Measured on Wikipedia's Mount Everest article (2,683 controls, cap 250): the "Tenzing Norgay" link the
goal names ranked with every other off-screen link and was cut, so the model answered BLOCKED three runs
out of three. With the goal's words the observer keeps it; without a goal the read is what it was.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_ultrafast_mcp import policy  # noqa: E402
from jev_ultrafast_mcp.browser import BrowserManager  # noqa: E402
from jev_ultrafast_mcp.config import Config, find_chrome  # noqa: E402


def _chrome_available() -> bool:
    try:
        find_chrome(None)
        return True
    except RuntimeError:
        return False


# A form at the top, then 400 links; the one the goal names is near the end, below the fold.
LINKS = "".join(f'<p><a href="#a{i}">Summit route {i}</a></p>' for i in range(400))
PAGE = ("<!doctype html><title>article</title><form><label>Search <input name=q></label>"
        "<button>Go</button></form>" + LINKS[: len(LINKS) // 2]
        + '<p><a href="#tn">Tenzing Norgay</a></p>' + LINKS[len(LINKS) // 2:])


def test_goal_terms_drop_filler_and_keep_names():
    terms = policy.goal_terms("Open the Tenzing Norgay article from this page, then open Edmund Hillary.")
    assert terms == ("tenzing", "norgay", "edmund", "hillary")


@pytest.mark.skipif(not _chrome_available(), reason="needs Chrome")
def test_a_named_control_survives_the_cap_only_while_a_goal_names_it():
    manager = BrowserManager(Config(headless=True, max_actions=60))
    try:
        session = manager.start("prefer", "data:text/html," + quote(PAGE))
        plain = session.observe()
        assert plain.omitted > 0
        assert not any(e.name == "Tenzing Norgay" for e in plain.elements)

        session.goal_terms = policy.goal_terms("Open the Tenzing Norgay article")
        steered = session.observe()
        assert any(e.name == "Tenzing Norgay" for e in steered.elements)
        # The form is still there: goal words add a control, they do not displace the primary ones.
        assert any(e.role == "button" and e.name == "Go" for e in steered.elements)

        session.goal_terms = ()
        assert not any(e.name == "Tenzing Norgay" for e in session.observe().elements)
    finally:
        manager.shutdown()


def test_the_decision_question_keeps_goal_named_links_inside_its_own_cut():
    """The second cut: `reachable_first` put 120 of the observer's 250 to the model and dropped the
    goal-named link again among the off-screen ones."""
    from jev_ultrafast_mcp.observe import Element
    links = [Element(ref=f"e{i}", role="link", name=f"Summit route {i}", in_viewport=i < 40) for i in range(200)]
    named = Element(ref="e999", role="link", name="Tenzing Norgay", in_viewport=False)
    pool = links[:150] + [named] + links[150:]
    assert named not in policy.reachable_first(pool)
    kept = policy.reachable_first(pool, prefer=policy.goal_terms("Open the Tenzing Norgay article"))
    assert named in kept and len(kept) == 120
    assert kept == [e for e in pool if e in kept], "document order is kept"
