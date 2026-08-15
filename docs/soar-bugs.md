# Soar — bug log

**Live work only.** A closed entry is deleted from this file rather than kept
here: `git log -p -- docs/soar-bugs.md` has every one of them in full, and
that is the archive (Thomas, 2026-08-15 — "if you want historical context,
the move is to commit and then remove the file. Not keep around endlessly
accumulating files").

Numbers are never reused, so a number that is not here is closed. Several
code comments cite the closed ones — bugs 1, 2, 4, 5, 6, 8 and 12 are named
in `viewer.js`, `ui.js`, `main.js`, `raymarch.wgsl`, `soar_shader_ab.py` and
`tests/test_io_overrides.py`. Those comments carry the reasoning themselves;
the entry behind them is one `git log` away if the full account is wanted.

Entries 1–10, 12, 16, 18 and 19 were removed on 2026-08-15, all closed. 11
and 13 survive as the one open item each of them still holds.

---

## 11. (coda) Back from the units panel returns to the file picker

**Status:** open, deliberately parked. The bug itself — cancelling the
group/units question reported a load failure — was fixed 2026-08-11
(`d6a4d17`); this is what was left behind.

Where "Back" goes from the *units* panel when the file also asked a group
question. Back to the group choice is the honest answer, not back to the file
picker — but that means the ask sequence needs to be re-enterable rather than
a straight-line `await` chain.

And `_ask`'s `done()` restores the loading overlay where `cancel()` does not:

```js
    const done = (value) => {
      this.ui.close();
      this.setLoadingVisible?.(true);
      resolve(value);
    };
```

That is fine while cancel ends in a failure panel. Once cancel returns to the
picker instead, the overlay state has to be put back deliberately or the next
attempt starts with stale chrome.

---

## 13. (item 3) Editable camera/view boxes in the terminal-render panel

**Status:** open, and worth scoping separately. The rest of 13 landed
2026-08-11 (`522f04f`) — the panel exists, offers witness and behold, and
prints the reproduction command.

Editable camera numbers that drive the live view means the panel becomes an
input surface rather than a printed command: parse, validate, clamp to the
domain, push into `Viewer.camera`, reset accumulation, and re-render the
command string on every change. Consider whether it wants to be its own panel
rather than more rows on this one.

One thing to resolve before starting, from the original report: *"the flight
video doesn't really make sense to be shown here."* There is no flight-video
control in the terminal panel today — video lives in `_panel_capture` and
`_panel_track` — so this may already be answered.

---

## 14. Firefox crashes while the tab is backgrounded

**Status:** open, and the one long-running thread here. Absorbs bugs 15, 17
and 18, which were four crash reports and their competing readings; the
reading below is what survived them.

**The trigger is occlusion plus a live GPU device**, stated forwards by
Thomas on 2026-08-14 rather than reconstructed from stacks:

> "still crashing, possibly less pften? Only when i am in another app."

That settles it against the teardown-ordering hypothesis the first reports
suggested: three of the four crashes had no teardown running at all, and the
"actions" recorded on the other two were simply the last thing before tabbing
away. The 2026-08-11 mitigations (`35453d3` — render targets pooled per rung
size, the probe verdict cached, `_releaseField`'s drain guarded) narrowed the
window teardown ordering opens, which is not the window that matters.

**"Possibly less often" is an impression, not a count**, and must not be
promoted into a measurement of that work. It is equally consistent with less
flying since the deploy.

**The fix to aim at: release the device when the page is hidden** —
`visibilitychange` → unconfigure the canvas, drop the device, rebuild on
return. soar then is not holding the object that dies. **Still not
implemented:** `grep -rn visibilitychange web/soar/*.js` finds one listener,
in `capture.js`, waiting for a save to finish.

**Caveat, binding:** re-uploading a multi-gigabyte volume on un-hide is not
acceptable — a device drop that costs a reload is worse than the crash. Keep
the field in host memory and rebuild only GPU resources. This composes with
the parked-loop work: a hidden tab is the limiting case of a view that is not
changing.

**The decisive experiment, outstanding since 2026-08-11 and still the
cheapest thing in this file:** open any unrelated WebGPU page — a browser
sample — background the window, and go do something else for half an hour. If
Firefox dies, the whole thing is upstream with four crash ids and two stack
variants, and soar's remaining job is only to stop holding the object that
dies.

---

## 20. After a GPU out-of-memory, the next load misbehaves until a reload

**Status:** open, reported 2026-08-15 on macOS against the deployed
seven-demo build. Thomas:

> "on mac, if I attempt to load a big scene, i get GPU OOM error (fair
> enough, and shown well), but then i try another and i get weird errors and
> have to reload. can reproduce on request."

**So the OOM itself is working as intended** — `guardAllocation` catches it,
names the field and its size, and the failure panel says what to do. The bug
is the recovery: the session does not come back clean, and only a reload
fixes it.

**Next step: get the error text.** Thomas can reproduce on request and has
offered the actual message, so this entry does not carry a list of candidate
causes — theorising from a symptom when the fact is one question away is how
a log fills up with readings that later have to be argued down. Reproduce,
capture the console verbatim, then diagnose.

**One thing ruled out already, because it was free:** our own teardown
tripping the device-lost reporter. `watchDevice` returns early on
`info.reason === "destroyed"`, so the `device.destroy()` on the failure path
does not itself raise a second "GPU device was lost" panel.

**Not to be fixed by making the OOM quieter.** The panel is right; it is the
state left behind it that is wrong.
