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

One adjacent facet was fixed 2026-08-18 (found by codex review): the landing
page's persistent file input was never reset, so after a Back from the
group/units question, re-picking the *same* file fired no `change` event and
silently did nothing. `main.js` now clears `fileInput.value` on every
selection. The Back-destination and overlay questions above remain open.

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

## 21. Haze distance past ~70 km blacks the screen

**Status:** open, reported 2026-08-18 on macOS (M1, 8 GB) against the
quality panel's haze slider. Thomas: *"there is also a bug where beyond
about 70 km for the haze, the screen goes black."*

Suggestive fact: 70 km is where the aerosol coordinate crosses ZERO —
`hazeFromEFoldingKm(70) ≈ -0.038`, and the slider deliberately runs past
haze 0 toward the 200 km Rayleigh end (`HAZE_MIN` in spectral.js is
negative). So the first suspects are consumers of haze that assume it is
non-negative: `haze * Math.sqrt(Math.abs(haze))` in aerialBetaPerKm is
sign-preserving on purpose, but anything downstream taking `pow(haze, x)`,
`log(haze)`, or a division by a beta that has crossed zero would produce a
NaN, and a NaN anywhere in the tone map is a black screen.

**Candidate fix landed 2026-08-18, same day:** `circumsolar_amplitude` takes
`pow(haze / HAZE_ANCHOR, 1.4)`, which is NaN in WGSL for negative haze.
sky_radiance now clamps its haze read at zero (raymarch.wgsl) — extinction
keeps the sign-preserving negative range, the sky cosmetics hold their
aerosol-free look below zero. Matters doubly because the high/max tiers now
DEFAULT to 70 km (aerosol -0.038), on the wrong side of the cliff. Needs
verification on the Mac against the slider's whole clear end; verified here
only that the shader compiles and the golden views are unchanged.
