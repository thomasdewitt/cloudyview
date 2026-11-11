"""soar_cdp's waitFor fails at once when the waited expression throws.

The tools wait on expressions that carry their own failure checks: the
loading wait in pinned_still throws the page's failure panel text when the
scene fails to load. A waitFor that polled
`tab.eval(expression).catch(() => undefined)` would swallow that throw with
everything else: the page would show its failure panel and pinned_still time
out minutes later with the reason lost.

waitFor is now pollUntil over the tab's Runtime.evaluate. This drives
pollUntil with stand-in evaluate functions that answer as CDP does: a result,
a result carrying exceptionDetails (the expression threw), or a rejected call
(a protocol error: no execution context while the page navigates). Needs
node; no Chrome, no GPU.
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
def runs():
    out = subprocess.run(["node", "--input-type=module", "-e", SCRIPT],
                         capture_output=True, text=True, check=True,
                         timeout=60)
    return {r["name"]: r for r in json.loads(out.stdout)}


def test_a_thrown_expression_fails_at_once_with_the_pages_message(runs):
    r = runs["throws"]
    assert not r["ok"]
    assert "?surfacecs=2: expected 1 (or 512), 256, or 0" in r["error"]
    assert "the page threw while waiting for probe()" in r["error"]
    # On the poll that threw, not at the 60 s timeout.
    assert r["calls"] == 2 and r["ms"] < 5000, r


def test_a_not_yet_true_expression_still_waits(runs):
    r = runs["waits"]
    assert r["ok"] and r["value"] == "ready" and r["calls"] == 5, r


def test_a_navigation_mid_poll_does_not_throw(runs):
    r = runs["navigates"]
    assert r["ok"] and r["value"] is True and r["calls"] == 4, r


def test_a_timeout_names_the_last_protocol_error(runs):
    r = runs["times out"]
    assert not r["ok"] and r["error"].startswith("timed out waiting for probe()")
    assert "Execution context was destroyed." in r["error"], r
    r = runs["never true"]
    assert not r["ok"] and r["error"] == "timed out waiting for probe()", r


def test_a_thrown_value_that_is_not_an_error_keeps_its_value(runs):
    r = runs["throws a string"]
    assert not r["ok"] and r["error"].endswith(": scene failed: no adapter"), r


def test_a_closed_connection_fails_at_once(runs):
    r = runs["closed"]
    assert not r["ok"] and "DevTools connection closed" in r["error"], r
    assert r["calls"] == 2 and r["ms"] < 5000, r


def test_an_evaluate_that_never_answers_ends_at_the_timeout(runs):
    r = runs["no answer"]
    assert not r["ok"], r
    assert "the page did not answer within 150 ms" in r["error"], r
    assert 140 <= r["ms"] < 2000, r


def test_an_abandoned_evaluate_rejecting_later_is_no_crash(runs):
    # The fixture's script waited past the late rejection before printing:
    # had it been unhandled, node would have exited non-zero with no output.
    r = runs["late rejection"]
    assert not r["ok"] and "did not answer within 50 ms" in r["error"], r
