"""A page with content and no controls is waited on only while it could still be arriving.

`observe` waits (up to `settle_timeout`) when a page has content but nothing actionable, because
that is what a client-rendered page looks like before its JavaScript paints. A finished page
("Setup complete", "Order placed") looks the same and used to pay the whole budget every time.
These run against real pages in headless Chrome: a finished page ends the wait early, a
timer-painted page and a page still fetching its content are still waited for.
"""
from __future__ import annotations

import http.server
import threading
import time

import pytest

from jev_ultrafast_mcp import browser as jb
from jev_ultrafast_mcp.config import Config

FINISHED = b"<!doctype html><title>Done</title><body><h1>Setup complete</h1><p>You can close this tab.</p></body>"
LATE_TIMER = b"""<!doctype html><title>Late</title><body><div id=app><p>loading</p></div><script>
setTimeout(() => { document.getElementById('app').innerHTML = '<button>Go</button>'; }, 1200);
</script></body>"""
LATE_FETCH = b"""<!doctype html><title>Fetch</title><body><div id=app><p>loading</p></div><script>
fetch('/slow').then(r => r.text()).then(t => { document.getElementById('app').innerHTML = t; });
</script></body>"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_a):
        pass

    def do_GET(self):  # noqa: N802
        body = {"/finished": FINISHED, "/late-timer": LATE_TIMER, "/late-fetch": LATE_FETCH}.get(self.path)
        if self.path == "/slow":
            time.sleep(2.5)
            body = b"<a href='/finished'>Continue</a>"
        if body is None:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def site():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="module")
def manager():
    m = jb.BrowserManager(Config(headless=True, allow_js=True))
    try:
        yield m
    finally:
        m.shutdown()


def _observe_after_click(manager, site, path):
    """Reach the page by a script navigation (the way a clicked link does), then time `observe`."""
    tab = manager.start("probe", site + "/finished")
    tab.observe()
    tab.cdp.events.clear()
    tab._safe_eval(f"location.href = {site + path!r}")
    time.sleep(0.3)
    started = time.monotonic()
    observation = tab.observe()
    return observation, time.monotonic() - started


def test_a_finished_page_does_not_pay_the_whole_budget(manager, site):
    observation, took = _observe_after_click(manager, site, "/finished")
    assert observation.elements == []
    assert took < manager.cfg.settle_timeout - 1.5, f"waited {took:.1f}s on a page with nothing coming"


def test_a_timer_painted_page_is_still_waited_for(manager, site):
    observation, _ = _observe_after_click(manager, site, "/late-timer")
    assert [e.name for e in observation.elements] == ["Go"]


def test_a_page_still_fetching_its_content_is_still_waited_for(manager, site):
    observation, _ = _observe_after_click(manager, site, "/late-fetch")
    assert [e.name for e in observation.elements] == ["Continue"]
