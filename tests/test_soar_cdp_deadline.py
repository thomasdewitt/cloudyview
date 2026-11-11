"""soar_cdp's goto and eval cannot hang a tool forever.

After the waitFor fixes (a throw fails at once; a dead socket and a silent
evaluate end the wait), two routes were left (review of
d81df79): goto waited for Page.loadEventFired with no deadline (and a
same-document navigation never fires one), and plain tab.eval had no
deadline, while pinned_still runs its long
async renders through it. Now tab.eval takes a generous deadline
(EVAL_DEADLINE_MS unless the caller says otherwise) and goto is
navigateAndLoad: a network failure fails at once with Chrome's reason, a
same-document navigation returns at once, and the load is awaited under a
deadline and against the socket closing.

This drives withDeadline and navigateAndLoad with stand-ins for CDP's
send and its events. Needs node; no Chrome, no GPU.
"""

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SOAR_CDP = REPO / "tools" / "soar_cdp.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")

SCRIPT = textwrap.dedent("""
    import { withDeadline, navigateAndLoad, EVAL_DEADLINE_MS,
             GOTO_DEADLINE_MS } from "%s";
    const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
    const never = () => new Promise(() => {});

    async function timed(name, f) {
      const t0 = Date.now();
      try {
        const v = await f();
        return { name, ok: true, value: v ?? null, ms: Date.now() - t0 };
      } catch (err) {
        return { name, ok: false, error: err.message, deadline: !!err.deadline,
                 closed: !!err.closed, ms: Date.now() - t0 };
      }
    }

    // A tab's plumbing: `reply` is what Page.navigate answers; `loadAfter`
    // fires Page.loadEventFired that many ms after it (null: never).
    function tab({ reply, loadAfter = null, closeAfter = null }) {
      const listeners = [];
      let rejectOnClose;
      const whenClosed = new Promise((_, bad) => { rejectOnClose = bad; });
      whenClosed.catch(() => {});
      const send = async (method) => {
        if (method !== "Page.navigate") throw new Error(method);
        if (loadAfter !== null) {
          setTimeout(() => { for (const f of [...listeners]) f({ method: "Page.loadEventFired" }); },
                     loadAfter);
        }
        if (closeAfter !== null) {
          setTimeout(() => rejectOnClose(Object.assign(
            new Error("the DevTools connection closed (code 1006)"), { closed: true })),
            closeAfter);
        }
        return typeof reply === "function" ? reply() : reply;
      };
      const onEvent = (f) => {
        listeners.push(f);
        return () => listeners.splice(listeners.indexOf(f), 1);
      };
      return { plumbing: { send, onEvent, whenClosed }, listeners };
    }

    const out = [];
    out.push(await timed("deadline holds", () => withDeadline(never(), 60, "evaluate x")));
    out.push(await timed("deadline passes a value", () => withDeadline(sleep(10).then(() => 7), 1000, "x")));
    out.push(await timed("deadline passes a rejection", () => withDeadline(Promise.reject(new Error("boom")), 1000, "x")));
    {
      const t = tab({ reply: { frameId: "F", loaderId: "L" }, loadAfter: 30 });
      const r = await timed("loads", () => navigateAndLoad(t.plumbing, "http://x/", 5000));
      r.listenersLeft = t.listeners.length;
      out.push(r);
    }
    {
      const t = tab({ reply: { frameId: "F" } });          // no loaderId, no load
      out.push(await timed("same document", () => navigateAndLoad(t.plumbing, "about:blank#x", 5000)));
    }
    {
      const t = tab({ reply: { frameId: "F", loaderId: "L",
                               errorText: "net::ERR_CONNECTION_REFUSED" }, loadAfter: 20 });
      out.push(await timed("network failure", () => navigateAndLoad(t.plumbing, "http://127.0.0.1:1/", 5000)));
    }
    {
      const t = tab({ reply: { frameId: "F", loaderId: "L" } });   // no load, ever
      const r = await timed("no load", () => navigateAndLoad(t.plumbing, "http://x/", 80));
      r.listenersLeft = t.listeners.length;
      out.push(r);
    }
    {
      const t = tab({ reply: { frameId: "F", loaderId: "L" }, closeAfter: 30 });
      out.push(await timed("closed mid-load", () => navigateAndLoad(t.plumbing, "http://x/", 5000)));
    }
    {
      const t = tab({ reply: never });                     // navigate never answers
      out.push(await timed("navigate silent", () => navigateAndLoad(t.plumbing, "http://x/", 80)));
    }
    console.log(JSON.stringify({ out, EVAL_DEADLINE_MS, GOTO_DEADLINE_MS }));
""") % SOAR_CDP.as_uri()


@pytest.fixture(scope="module")
def runs():
    out = subprocess.run(["node", "--input-type=module", "-e", SCRIPT],
                         capture_output=True, text=True, check=True,
                         timeout=60)
    got = json.loads(out.stdout)
    return {r["name"]: r for r in got["out"]}, got


def test_the_deadlines_are_generous(runs):
    _, got = runs
    # Minutes, not seconds: the long renders run through tab.eval.
    assert got["EVAL_DEADLINE_MS"] >= 5 * 60 * 1000
    assert got["GOTO_DEADLINE_MS"] >= 60 * 1000


def test_with_deadline(runs):
    r, _ = runs
    held = r["deadline holds"]
    assert not held["ok"] and held["deadline"], held
    assert held["error"] == "evaluate x: no answer within 60 ms", held
    assert 50 <= held["ms"] < 2000, held
    assert r["deadline passes a value"]["ok"] and r["deadline passes a value"]["value"] == 7
    passed = r["deadline passes a rejection"]
    assert not passed["ok"] and passed["error"] == "boom" and not passed["deadline"]


def test_a_goto_waits_for_the_load_and_leaves_no_listener(runs):
    r, _ = runs
    loads = r["loads"]
    assert loads["ok"] and loads["ms"] >= 25, loads
    assert loads["listenersLeft"] == 0, loads


def test_a_same_document_goto_returns_at_once(runs):
    r, _ = runs
    same = r["same document"]
    assert same["ok"] and same["ms"] < 1000, same


def test_a_network_failure_fails_at_once_with_chromes_reason(runs):
    r, _ = runs
    fail = r["network failure"]
    assert not fail["ok"], fail
    assert fail["error"] == "goto http://127.0.0.1:1/: net::ERR_CONNECTION_REFUSED", fail


def test_a_goto_with_no_load_ends_at_its_deadline(runs):
    r, _ = runs
    none = r["no load"]
    assert not none["ok"] and none["deadline"], none
    assert none["error"] == "goto http://x/: the load event: no answer within 80 ms"
    assert none["listenersLeft"] == 0, none
    silent = r["navigate silent"]
    assert not silent["ok"] and silent["deadline"], silent
    assert silent["error"] == "goto http://x/: no answer within 80 ms", silent


def test_a_goto_fails_when_the_socket_closes(runs):
    r, _ = runs
    closed = r["closed mid-load"]
    assert not closed["ok"] and closed["closed"], closed
    assert closed["ms"] < 2000, closed
