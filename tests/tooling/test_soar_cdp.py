"""tools/soar_cdp.mjs, the DevTools client the browser tools share, under node.

Three groups, each under the docstring of the file it was merged from: the
software-adapter refusal, the goto and eval deadlines, and waitFor's
handling of a throwing expression. Needs node; no Chrome, no GPU.
"""

import json
import shutil
import subprocess
import textwrap

import pytest

from tests.conftest import REPO

SOAR_CDP = REPO / "tools" / "soar_cdp.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="needs node")


# =============================================================================
# Merged from test_soar_cdp_adapter
#
# soar_cdp refuses a Chrome that renders WebGPU in software.
#
# Chrome falls back to SwiftShader when its Vulkan flags are missing or wrong,
# and Mesa's llvmpipe/lavapipe answer as a GPU too; every still hashed and
# pass timed on one is fiction against the hardware ones. tint_check and
# pinned_still call requireHardwareAdapter, whose verdict is softwareAdapter's,
# pinned here on the adapter infos those rasterizers report. Needs node.
# =============================================================================


CASES = {
    "rtx5080": ({"vendor": "nvidia", "architecture": "blackwell"}, False),
    "swiftshader": ({"vendor": "google", "architecture": "swiftshader",
                     "description": "SwiftShader Device (Subzero)"}, True),
    "fallback": ({"vendor": "", "architecture": "",
                  "isFallbackAdapter": True}, True),
    "llvmpipe": ({"vendor": "mesa", "architecture": "",
                  "description": "llvmpipe (LLVM 19.1.0, 256 bits)"}, True),
    "lavapipe": ({"vendor": "mesa", "architecture": "lavapipe"}, True),
    "none": (None, True),
}


def test_software_adapters_are_refused():
    if shutil.which("node") is None:
        pytest.skip("needs node")
    js = textwrap.dedent("""
        import { softwareAdapter } from "%s";
        const cases = JSON.parse(process.argv[1]);
        const out = {};
        for (const [k, [info]] of Object.entries(cases)) {
          out[k] = softwareAdapter(info);
        }
        console.log(JSON.stringify(out));
    """) % SOAR_CDP.as_uri()
    got = json.loads(subprocess.run(
        ["node", "--input-type=module", "-e", js, json.dumps(CASES)],
        capture_output=True, text=True, check=True).stdout)
    for name, (_, software) in CASES.items():
        assert (got[name] is not None) == software, (name, got[name])


# =============================================================================
# Merged from test_soar_cdp_deadline
#
# soar_cdp's goto and eval cannot hang a tool forever.
#
# After the waitFor fixes (a throw fails at once; a dead socket and a silent
# evaluate end the wait), two routes were left (review of
# d81df79): goto waited for Page.loadEventFired with no deadline (and a
# same-document navigation never fires one), and plain tab.eval had no
# deadline, while pinned_still runs its long
# async renders through it. Now tab.eval takes a generous deadline
# (EVAL_DEADLINE_MS unless the caller says otherwise) and goto is
# navigateAndLoad: a network failure fails at once with Chrome's reason, a
# same-document navigation returns at once, and the load is awaited under a
# deadline and against the socket closing.
#
# This drives withDeadline and navigateAndLoad with stand-ins for CDP's
# send and its events. Needs node; no Chrome, no GPU.
# =============================================================================


DEADLINE_SCRIPT = textwrap.dedent("""
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
def deadline_runs():
    out = subprocess.run(["node", "--input-type=module", "-e", DEADLINE_SCRIPT],
                         capture_output=True, text=True, check=True,
                         timeout=60)
    got = json.loads(out.stdout)
    return {r["name"]: r for r in got["out"]}, got


def test_the_deadlines_are_generous(deadline_runs):
    _, got = deadline_runs
    # Minutes, not seconds: the long renders run through tab.eval.
    assert got["EVAL_DEADLINE_MS"] >= 5 * 60 * 1000
    assert got["GOTO_DEADLINE_MS"] >= 60 * 1000


def test_with_deadline(deadline_runs):
    r, _ = deadline_runs
    held = r["deadline holds"]
    assert not held["ok"] and held["deadline"], held
    assert held["error"] == "evaluate x: no answer within 60 ms", held
    assert 50 <= held["ms"] < 2000, held
    assert r["deadline passes a value"]["ok"] and r["deadline passes a value"]["value"] == 7
    passed = r["deadline passes a rejection"]
    assert not passed["ok"] and passed["error"] == "boom" and not passed["deadline"]


def test_a_goto_waits_for_the_load_and_leaves_no_listener(deadline_runs):
    r, _ = deadline_runs
    loads = r["loads"]
    assert loads["ok"] and loads["ms"] >= 25, loads
    assert loads["listenersLeft"] == 0, loads


def test_a_same_document_goto_returns_at_once(deadline_runs):
    r, _ = deadline_runs
    same = r["same document"]
    assert same["ok"] and same["ms"] < 1000, same


def test_a_network_failure_fails_at_once_with_chromes_reason(deadline_runs):
    r, _ = deadline_runs
    fail = r["network failure"]
    assert not fail["ok"], fail
    assert fail["error"] == "goto http://127.0.0.1:1/: net::ERR_CONNECTION_REFUSED", fail


def test_a_goto_with_no_load_ends_at_its_deadline(deadline_runs):
    r, _ = deadline_runs
    none = r["no load"]
    assert not none["ok"] and none["deadline"], none
    assert none["error"] == "goto http://x/: the load event: no answer within 80 ms"
    assert none["listenersLeft"] == 0, none
    silent = r["navigate silent"]
    assert not silent["ok"] and silent["deadline"], silent
    assert silent["error"] == "goto http://x/: no answer within 80 ms", silent


def test_a_goto_fails_when_the_socket_closes(deadline_runs):
    r, _ = deadline_runs
    closed = r["closed mid-load"]
    assert not closed["ok"] and closed["closed"], closed
    assert closed["ms"] < 2000, closed


# =============================================================================
# Merged from test_soar_cdp_waitfor
#
# soar_cdp's waitFor fails at once when the waited expression throws.
#
# The tools wait on expressions that carry their own failure checks: the
# loading wait in pinned_still throws the page's failure panel text when the
# scene fails to load. A waitFor that polled
# `tab.eval(expression).catch(() => undefined)` would swallow that throw with
# everything else: the page would show its failure panel and pinned_still time
# out minutes later with the reason lost.
#
# waitFor is now pollUntil over the tab's Runtime.evaluate. This drives
# pollUntil with stand-in evaluate functions that answer as CDP does: a result,
# a result carrying exceptionDetails (the expression threw), or a rejected call
# (a protocol error: no execution context while the page navigates). Needs
# node; no Chrome, no GPU.
# =============================================================================


WAITFOR_SCRIPT = textwrap.dedent("""
    import { pollUntil } from "%s";
    const value = (v) => ({ result: { type: typeof v, value: v } });
    // What Runtime.evaluate answers for `throw new Error("Scene failed: X")`.
    const threw = (msg) => ({
      result: { type: "object", subtype: "error" },
      exceptionDetails: { text: "Uncaught", exception: {
        type: "object", subtype: "error",
        description: `Error: ${msg}\\n    at <anonymous>:1:60` } } });
    // What send() rejects with while the page is between documents.
    const destroyed = () => Promise.reject(new Error(JSON.stringify(
      { code: -32000, message: "Execution context was destroyed." })));

    async function run(name, answers, timeoutMs) {
      let calls = 0;
      const evaluate = async () => {
        const a = answers[Math.min(calls++, answers.length - 1)];
        return typeof a === "function" ? a() : a;
      };
      const t0 = Date.now();
      try {
        const v = await pollUntil(evaluate, "probe()", timeoutMs, 10);
        return { name, ok: true, value: v, calls, ms: Date.now() - t0 };
      } catch (err) {
        return { name, ok: false, error: err.message, calls, ms: Date.now() - t0 };
      }
    }

    // What send() rejects with once the DevTools socket has closed.
    const closed = () => Promise.reject(Object.assign(
      new Error("the DevTools connection closed (code 1006)"), { closed: true }));
    // A primitive thrown: no description, only the value.
    const threwValue = (v) => ({
      result: { type: typeof v, value: v },
      exceptionDetails: { text: "Uncaught",
                          exception: { type: typeof v, value: v } } });

    const out = [
      await run("throws", [value(false), threw("Scene failed: ?surfacecs=2: expected 1 (or 512), 256, or 0")], 60000),
      await run("throws a string", [threwValue("scene failed: no adapter")], 60000),
      await run("closed", [value(false), closed], 60000),
      await run("no answer", [() => new Promise(() => {})], 150),
      // Abandoned at the deadline, then rejected as the socket closes: the
      // race's own handler covers it, so no unhandled rejection ends this
      // script (a guard: checked by a control without the race, 2026-11-09).
      await run("late rejection", [() => new Promise((_, bad) => setTimeout(
        () => bad(Object.assign(new Error("closed"), { closed: true })), 200))], 50),
      await run("waits", [value(false), value(0), value(null), value(""), value("ready")], 60000),
      await run("navigates", [value(false), destroyed, destroyed, value(true)], 60000),
      await run("times out", [destroyed], 80),
      await run("never true", [value(false)], 80),
    ];
    await new Promise((ok) => setTimeout(ok, 400));   // let it reject
    console.log(JSON.stringify(out));
""") % SOAR_CDP.as_uri()


@pytest.fixture(scope="module")
def waitfor_runs():
    out = subprocess.run(["node", "--input-type=module", "-e", WAITFOR_SCRIPT],
                         capture_output=True, text=True, check=True,
                         timeout=60)
    return {r["name"]: r for r in json.loads(out.stdout)}


def test_a_thrown_expression_fails_at_once_with_the_pages_message(waitfor_runs):
    r = waitfor_runs["throws"]
    assert not r["ok"]
    assert "?surfacecs=2: expected 1 (or 512), 256, or 0" in r["error"]
    assert "the page threw while waiting for probe()" in r["error"]
    # On the poll that threw, not at the 60 s timeout.
    assert r["calls"] == 2 and r["ms"] < 5000, r


def test_a_not_yet_true_expression_still_waits(waitfor_runs):
    r = waitfor_runs["waits"]
    assert r["ok"] and r["value"] == "ready" and r["calls"] == 5, r


def test_a_navigation_mid_poll_does_not_throw(waitfor_runs):
    r = waitfor_runs["navigates"]
    assert r["ok"] and r["value"] is True and r["calls"] == 4, r


def test_a_timeout_names_the_last_protocol_error(waitfor_runs):
    r = waitfor_runs["times out"]
    assert not r["ok"] and r["error"].startswith("timed out waiting for probe()")
    assert "Execution context was destroyed." in r["error"], r
    r = waitfor_runs["never true"]
    assert not r["ok"] and r["error"] == "timed out waiting for probe()", r


def test_a_thrown_value_that_is_not_an_error_keeps_its_value(waitfor_runs):
    r = waitfor_runs["throws a string"]
    assert not r["ok"] and r["error"].endswith(": scene failed: no adapter"), r


def test_a_closed_connection_fails_at_once(waitfor_runs):
    r = waitfor_runs["closed"]
    assert not r["ok"] and "DevTools connection closed" in r["error"], r
    assert r["calls"] == 2 and r["ms"] < 5000, r


def test_an_evaluate_that_never_answers_ends_at_the_timeout(waitfor_runs):
    r = waitfor_runs["no answer"]
    assert not r["ok"], r
    assert "the page did not answer within 150 ms" in r["error"], r
    assert 140 <= r["ms"] < 2000, r


def test_an_abandoned_evaluate_rejecting_later_is_no_crash(waitfor_runs):
    # The fixture's script waited past the late rejection before printing:
    # had it been unhandled, node would have exited non-zero with no output.
    r = waitfor_runs["late rejection"]
    assert not r["ok"] and "did not answer within 50 ms" in r["error"], r
