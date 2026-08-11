# Soar — bug log

Running list of bugs found while flying the deployed build. Not being fixed
right now; this file is the queue for a later session. Newest at the bottom.

**Bugs 1–6 carry line numbers against commit `e150a47` ("soar ships as one
folder"), which is exactly what was deployed when they were found.** Line
numbers drift; the quoted snippets do not. If a number no longer lands, grep
the snippet.

**Bug 7 onward has no line numbers on purpose** — they were logged during an
optimization pass that is moving code around. Everything is anchored by
symbol name and by verbatim snippet instead.

---

## 1. F12 opens the capture dialog but does not release the pointer

**Status:** open
**Found:** 2026-08-10, deployed web build

**Repro:** fly (pointer captured), press F12. The capture dialog appears, but
mouse movement still turns the camera behind it — the pointer is never
released, so the dialog cannot be used with the mouse.

**Where:** [viewer.js:427](web/soar/viewer.js:427)

```js
      case "F12": e.preventDefault(); this.ui.open("capture"); return;
```

Every other path that opens UI drops the lock first. `pause()`
([viewer.js:450](web/soar/viewer.js:450)):

```js
  pause() {
    this.paused = true;
    this.camera.keys.clear();
    if (this.captured) document.exitPointerLock();
    this.ui.open("main");
    this._syncChrome();
```

and Tab ([viewer.js:413](web/soar/viewer.js:413)):

```js
      case "Tab":
        e.preventDefault();
        if (this.captured) {
          this._tabRelease = true;
          document.exitPointerLock();
```

The F12 case opens the panel without that step.

**Likely fix:** release the pointer before opening, and decide whether F12
should behave like Escape (resume back into flight, no re-grab) or like Tab
(re-grab on close). Watch the interaction with the post-Escape cooldown
documented above `_requestCapture` ("Chromium returns a promise that rejects
inside the post-Escape cooldown"), and with `_syncChrome()`, which needs to
run so the chrome matches the released state.

---

## 2. After a screenshot, Escape releases the cursor but does not open the menu

**Status:** open, cause not confirmed — needs a live repro
**Found:** 2026-08-10, deployed web build

**Repro:** F12 → take a still → fly around some more → press Escape. The
cursor is released, but no menu appears. The only way back in is the "menu"
button in the top-right corner.

**Two things wrong here, and the second is the one that matters.**

### 2a. The Escape-to-pause path gets eaten

Pausing on Escape does not hang off the keydown. Chrome consumes the Escape
that breaks a pointer lock and never delivers the keydown, so the real trigger
is losing the lock, at [viewer.js:352](web/soar/viewer.js:352):

```js
      if (wasCaptured && !this._tabRelease && !this.paused) this.pause();
      this._tabRelease = false;
```

Since the user was flying, `paused` was false — `_frame` gates movement on it
([viewer.js:1020](web/soar/viewer.js:1020)):

```js
    if (!this.paused) this.camera.move(dt);
```

and `wasCaptured` was true. That leaves `_tabRelease` stuck true as the
suspect. It is a one-shot flag set in two places — Tab
([viewer.js:416](web/soar/viewer.js:416)) and fullscreen minimap
([viewer.js:526](web/soar/viewer.js:526)):

```js
      if (this.captured) {
        this._tabRelease = true;
        document.exitPointerLock();
      }
      this.ui.say("Minimap fullscreen — click to travel, M to dismiss.", 4);
```

and cleared in exactly one place, the `pointerlockchange` handler quoted
above. Any path that sets it without a matching lock-change event leaves it
armed, and the next Escape is silently swallowed with no way to recover.

Worth checking alongside it: `saveScreenshot` closes the menu
([viewer.js:815](web/soar/viewer.js:815)) with no following `_syncChrome()`:

```js
    const size = this.captureDimensions();
    this.ui.close();
    this.ui.showProgress(
```

so the chrome classes can end up describing a state the app is no longer in.

Regardless of cause, the deeper problem is that a swallowed Escape is
unrecoverable — pressing it again does nothing, because the lock is already
gone and the keydown route is not what pauses.

### 2b. The top-right "menu" button should not exist

It only appears as a rescue for the state above — mouse free, menu closed
([viewer.css:103](web/soar/viewer.css:103)):

```css
/* Only while the mouse is free AND the menu is closed — see _syncChrome. */
#viewer.captured #toolbar,
#viewer.menu-open #toolbar { display: none; }
```

driven by `_syncChrome` ([viewer.js:444](web/soar/viewer.js:444)):

```js
    viewer.classList.toggle("captured", this.captured);
    viewer.classList.toggle("menu-open", Boolean(this.ui?.isOpen));
```

If Escape always paused correctly there would be nothing for it to do. Delete
it rather than keep it as a net, and make the keyboard route reliable enough
not to need one.

---

## 3. The GPU stays hot on a view that is not changing

**Status:** FIXED in the 2026-08-11 optimization pass ("soar: sleep on a
converged view, and measure the machine before trusting it") — the loop stops
marching at 64 accumulated samples on the hold ladder's top rung, wakes
through a single `_wake(reason)` funnel covering the enumeration below, and
runs a presentation-only loop (blit + overlays, no march) while the bird
animates. The stats readout says `parked`.
**Found:** 2026-08-10, deployed web build

The frame loop is unconditional. `_frame` ends by scheduling the next one
([viewer.js:1103](web/soar/viewer.js:1103)) no matter what happened in it:

```js
    this._raf = requestAnimationFrame(() => this._frame(generation));
```

so a paused menu, or a pilot who simply stopped moving, keeps re-marching the
volume at full cost forever. It should render until the picture is settled,
present that, and then stop until something actually changes.

**Where the convergence signal already is.** `_accumulationPlan`
([renderer.js:604](web/soar/renderer.js:604)) blends each new sample at
`1/nextCount` ([renderer.js:610](web/soar/renderer.js:610)):

```js
    const prevCount = this._accumCount;
    let nextCount = prevCount + 1;
    let prevWeight = prevCount === 0 ? 0.0 : prevCount / nextCount;
    let sampleWeight = prevCount === 0 ? 1.0 : 1.0 / nextCount;
```

`_accumCount` is unbounded, so past a few hundred frames every new frame costs
a full march to change nothing visible. `_accumCount` past a threshold (with
`_accumKey` unchanged and `_accumMotion` false) is the "settled" test, and
needs no new state.

**What has to wake the loop again.** This is the whole difficulty, and the
reason to enumerate it before writing any code:

- camera input — keydown/keyup, mouse move under lock, wheel speed change
- sun, quality tier, tone-map gamma, periodic toggle, nest add/remove
- canvas resize and devicePixelRatio change (see below)
- overlays with their own motion: the bird when enabled and unpaused, the
  minimap on mode change
- track recording and video capture, which must hold the loop awake outright
- the stats/FPS readout, which becomes meaningless while parked and should
  probably say so rather than report a stale number

The resize case needs its own observer, because the size check currently
lives *inside* the frame ([viewer.js:1034](web/soar/viewer.js:1034)) and would
stop running the moment the loop slept:

```js
    if (this.canvas.width !== outW || this.canvas.height !== outH) {
```

**Note the interaction with bugs 1 and 2:** pausing must be reliable before
sleeping on pause is safe. A loop asleep in a state the app thinks is
"flying" would look like a hang.

---

## 4. Parallel banding along the nest seam, seen from inside the nest

> **Before fixing this, ask Thomas to reproduce it.** The description below is
> second-hand and the viewing geometry matters — get the actual view before
> theorising. Do not start from the guess in this note.

**Status:** open, cause unknown
**Found:** 2026-08-10, deployed web build

**Repro (approximate):** stand inside the nest, look up and out at a cloud
that is outside the nest. A set of parallel lines appears, running parallel
to the nest seam.

**Where to start looking.** The march changes step size at the nest boundary
— `sample_level_at` ([raymarch.wgsl:556](web/soar/raymarch.wgsl:556)):

```wgsl
fn sample_level_at(p: vec3<f32>) -> LevelSample {
    let q = wrap_to_domain(p);
    let nested_here = in_nest_box(q);
    var s: LevelSample;
    s.in_nest = nested_here;
    s.sigma = sample_sigma_pinned(q, nested_here);
    if (NESTED && nested_here) {
        s.dt_view = u.nest_bmin.w;
        s.dt_light = u.nest_bmax.w;
    } else {
        s.dt_view = u.bmin.w;
        s.dt_light = u.bmax.w;
    }
```

switched by the hard boolean `in_nest_box`
([raymarch.wgsl:510](web/soar/raymarch.wgsl:510)):

```wgsl
    return all(q >= u.nest_bmin.xyz) && all(q <= u.nest_bmax.xyz);
```

A ray leaving the nest therefore changes step length discontinuously at a
plane — and a plane is exactly the shape of the reported artifact.

Banding of this family is usually a quadrature-phase problem rather than a
step-size one: the file already documents step-shell banding and the jitter
that exists to kill it (grep `coherent step-shell banding this exists to
kill` and `A randomized *lattice* rather than per-step stratification`).
Worth checking whether the march's entry phase
([raymarch.wgsl:1689](web/soar/raymarch.wgsl:1689)):

```wgsl
    let entry_dt = u.bmin.w;
    var t = t_near + jitter_on * jitter * jitter_scale * entry_dt;
```

is re-derived rather than carried across the seam crossing — note it is
seeded off `u.bmin.w`, the *outer* step, once per ray. That would re-align
every ray's samples to the boundary plane and make the shells coherent across
the screen instead of decorrelated.

That is a hypothesis from reading, not a diagnosis. Get the repro first.

---

## 5. A horizontally rectangular nest is not offered as a nest

**Status:** open, strong candidate cause found by reading
**Found:** 2026-08-10, deployed web build

**Symptom:** a nest that should qualify is not detected, when the nest is
horizontally rectangular.

**Cause (likely): the "is it finer?" gate collapses the grid to one scalar.**
`domainExtent` ([field.js:179](web/soar/field.js:179)) reduces all three axes
to a single `dx` — the minimum spacing over x, y **and** z together:

```js
  for (const c of [x, y, z]) {
    ...
    for (let i = 1; i < c.length; i++) {
      dx = Math.min(dx, Math.abs(c[i] - c[i - 1]));
    }
  }
```

`nestablePairs` then gates on that one number
([field.js:149](web/soar/field.js:149)):

```js
      if (inner === outer || !(inner.dx < outer.dx)) continue;
```

In an atmospheric field `dz` is usually far smaller than `dx`/`dy`, so this
scalar is the *vertical* spacing for both levels. A nest that refines
horizontally while sharing the parent's vertical levels — the ordinary case —
therefore ties, fails the strict `<`, and is dropped before containment is
ever tested. A rectangular footprint is exactly the shape that exposes this:
refining one horizontal axis more than the other is a fineness relation a
single scalar cannot express.

The failure is silent. The pair simply never appears in the group picker
([ingest/index.js:170](web/soar/ingest/index.js:170)):

```js
    const pairs = nestablePairs(groups.map((g) => {
```

which reads as "it did not detect the nest".

**The containment math itself is fine.** `nestOverhang`
([field.js:66](web/soar/field.js:66)) is already per-axis:

```js
    overhang.push(Math.max(
      outerMin[i] - nestMin[i], nestMax[i] - outerMax[i], 0.0));
```

as is the `covers` test. Only the fineness gate is at fault.

**This is not a browser/desktop divergence — both sides have it.** Python does
the same collapse in `group_domain_extent`
([io.py:227](cloudyview/io.py:227)):

```python
        spacing.append(float(abs(np.diff(values)).min()))
    return np.array(bmin), np.array(bmax), min(spacing)
```

and the same gate in `find_nestable_group_pairs`
([io.py:290](cloudyview/io.py:290)):

```python
            if inner == outer or inner_dx >= outer_dx:
                continue
```

Fix both together or they stop agreeing, which the header of `field.js` says
outright must not happen ("The arithmetic must agree with Python to the
float").

**Likely fix:** carry per-axis spacing instead of a scalar, and define finer
as "no axis coarser, at least one axis strictly finer". Keep the scalar
`minVoxelSize` where it belongs — the march step — and stop reusing it as a
refinement relation.

**To collect when fixing:** the file, or just each group's x/y/z lengths and
spacings. That confirms whether it is the vertical tie above or something
else about the rectangular footprint.

---

## 6. Ocean disappears in a region, periodic off, looking down from just outside

**Status:** FIXED in the 2026-08-11 optimization pass ("the ocean stops
disappearing when the ray misses the box") — exactly the likely-fix below:
an `ocean_consumed` flag replaces the `t_ocean > t_far` inference, zero
pixels changed on all eight goldens plus two non-broken non-periodic views,
32014 pixels restored in the repro. The 50-outer-width clamp was examined
and deliberately left (documented at the use site: invisible under the
shipped haze; removing it is a look change).
**Found:** 2026-08-10, deployed web build

**Repro:** turn periodic off, put the camera outside the domain but close to
it, look down. A region of the ocean is missing (sky shows through instead).

**Cause (likely): rays that miss the domain box fall through both ocean
paths.** The ocean is shaded in two places, and neither covers the gap.

**One:** inside the march ([raymarch.wgsl:1725](web/soar/raymarch.wgsl:1725))
— but the whole march is behind a guard
([raymarch.wgsl:1721](web/soar/raymarch.wgsl:1721)):

```wgsl
    if (t_near >= 0.0 && t_near < t_far) {
        for (var i: i32 = 0; i < max_view_steps; i = i + 1) {
            // witness.py:621-646 tests ocean before the t_far break so an
            // ocean plane coincident with the box floor is still shaded.
            if (ocean_on && t >= t_ocean) {
```

**Two:** the far-water fallback
([raymarch.wgsl:2094](web/soar/raymarch.wgsl:2094)), which requires
`t_ocean > t_far`:

```wgsl
    if (ocean_on
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_ocean < 1e29
        && t_ocean > t_far) {
```

`ray_box` ([raymarch.wgsl:464](web/soar/raymarch.wgsl:464)) is a plain slab
test:

```wgsl
    let t_near = max(max(tmin.x, tmin.y), tmin.z);
    let t_far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(t_near, t_far);
```

so on a **miss** it returns `t_near > t_far` with `t_far` still a positive
finite number — the nearest exit plane of a box the ray never entered. A ray
that misses the box therefore skips the march (condition 1 fails), and then
fails condition 2 whenever that leftover `t_far` happens to exceed `t_ocean`.
The ocean plane is hit, and nothing shades it.

Whether `t_far > t_ocean` holds varies smoothly with direction, which is why
the result is a contiguous *region* of missing water rather than scattered
pixels.

**Why periodic off is the trigger:** in the periodic branch
([raymarch.wgsl:1588](web/soar/raymarch.wgsl:1588)) the interval comes from
the z slab plus the march cap, not from a box:

```wgsl
        let tz0 = (u.bmin.z - u.cam_origin.z) * inv_dir.z;
        let tz1 = (u.bmax.z - u.cam_origin.z) * inv_dir.z;
        t_near = max(min(tz0, tz1), 0.0);
```

Looking down always crosses the slab, so `t_near < t_far` holds and the march
runs — the miss case cannot arise. Only the `else` branch
([raymarch.wgsl:1599](web/soar/raymarch.wgsl:1599)) produces a degenerate
interval:

```wgsl
    } else {
        let hit = ray_box(u.cam_origin.xyz, inv_dir);
        t_near = max(hit.x, 0.0);
        t_far = hit.y;
    }
```

**Likely fix:** stop inferring "the march did not handle the ocean" from
`t_ocean > t_far`. Track it directly — a flag set where the march shades the
ocean — and let the fallback shade whenever the ocean was hit, transmittance
survives, and the march did not already consume it. That also removes the
dependence on a `t_far` that is meaningless on a miss.

Worth checking at the same time whether the 50-outer-width clamp
([raymarch.wgsl:2101](web/soar/raymarch.wgsl:2101)) contributes a second,
larger-scale version of the same hole:

```wgsl
        if (abs(ocean_hit.x - center.x) < outer_size.x * 50.0
            && abs(ocean_hit.y - center.y) < outer_size.y * 50.0) {
```

---

## 7. The pause menu overlaps the hint strip, printing the field name twice

**Status:** open
**Found:** 2026-08-11, deployed web build

**Repro:** pause on a field whose menu is tall — the screenshot that found
this had minimap, bird, periodic, FOV and a field footer all present. The
bottom of `#menu` lands on top of `#hint`, and the only part of the hint left
showing is its subtitle: the same field label the menu already prints under
its own `field` heading. "group parent — small_c002_s1000.nc", twice, in two
boxes that visibly collide.

**Both halves come from one string.** `Viewer.sourceLabel` is rendered in the
menu footer, in `_panel_main`:

```js
    source.append(el("h3", null, "field"));
    source.append(el("div", null, app.sourceLabel));
```

and again into the hint's subtitle, once per field load:

```js
    this.ui.setSubtitle(this.sourceLabel);
```

where `setSubtitle` writes `#hint .sub`:

```js
  setSubtitle(text) { this.hintSub.textContent = text; }
```

**Why only the subtitle shows.** `#hint`'s first line is the controls strip
("Click to fly · WASD move · …"), built in `UI._build`. In the screenshot the
menu covers it, leaving the `.sub` line peeking out below the menu's bottom
edge — which is what makes the duplication obvious rather than merely
redundant.

**The asymmetry that causes it.** `viewer.css` hides the hint while the
pointer is captured:

```css
#viewer.captured #hint { display: none; }
```

but there is no `#viewer.menu-open #hint` rule, whereas `#toolbar` has both:

```css
#viewer.captured #toolbar,
#viewer.menu-open #toolbar { display: none; }
```

So the hint stays visible under an open menu, and nothing bounds the menu's
height to keep it clear.

**Likely fix, two parts, both probably wanted:**

1. Hide `#hint` while the menu is open — the same `menu-open` treatment
   `#toolbar` already gets. While the menu is up it is redundant anyway: the
   menu states the controls and names the field.
2. Give `#menu` a `max-height` with internal scrolling, so a tall menu can
   never reach whatever sits at the bottom of the viewport. Hiding the hint
   fixes this collision; it does not stop the menu running off a short
   window, which the FOV slider and field footer at the end make likely.

**While in there:** decide whether the field name belongs in the hint at all,
given the menu prints it. The hint's job is the controls strip; the subtitle
is the only thing that survived the overlap, which is a hint that it is the
part carrying its weight least.

---

## 8. The landing page sometimes shows a black backdrop instead of a still

**Status:** open, cause found by reading — intermittent, no reliable repro yet
**Found:** 2026-08-11, deployed web build

**Symptom:** the landing page occasionally comes up with a black background
rather than a demo still. Not reproducible on demand.

**Cause: two `showReel` calls in flight at once turn the backdrop off.**
`showReel` awaits the image load, but updates `reelActive` and `reelShown`
only *after* the await, so an overlapping call reads stale state:

```js
  image.classList.add("on");
  dom.reel[reelActive].classList.remove("on");
  reelActive = next;
  reelShown = demo.id;
```

Walk two hovers through it. Start with `reelActive = 1`, so `reel[1]` is the
visible one.

- Call A computes `next = 0`, points `reel[0]` at A's still, awaits.
- Call B arrives before A resolves. `reelActive` is *still* 1, so B computes
  `next = 0` as well — the same element. It overwrites `reel[0].src` with B's
  still and awaits.
- One `load` event now fires on `reel[0]`, and both listeners are on that
  element, so **both calls resume**.
- A resumes: shows `reel[0]`, hides `reel[1]`, sets `reelActive = 0`.
- B resumes: shows `reel[0]`, then runs `dom.reel[reelActive].classList
  .remove("on")` with `reelActive` now 0 — **it hides the element it just
  showed.**

Neither reel carries `on`, and the backdrop is black. It stays black until
the next hover lands on a *different* card, which is why it reads as random
and clears by itself.

**Why the landing page specifically.** `buildRail` awaits the first still to
put something on screen immediately:

```js
  if (all.length) await showReel(all[0], root);
```

That call is in flight for as long as the first image takes to load. A mouse
that reaches a demo card during that window is the second caller. So the race
is widest exactly at page load, on a slow or cold fetch — which also explains
why it resists reproduction on a warm cache.

**Two smaller paths to the same black, worth fixing in the same pass:**

- The early-out `reelShown === demo.id` is also set after the await, so it
  does not dedupe in-flight calls — two quick hovers over one card both
  proceed.
- An errored image still gets shown. The `error` listener resolves the same
  promise as `load`, deliberately, so a missing still costs one blank fade
  instead of a stuck reel. But if the failure is `all[0]` on the initial
  call, the page *opens* black and stays that way until a hover. A still
  that 404s or fails on a flaky fetch is another route into the symptom, and
  worth distinguishing from the race when diagnosing.

**Likely fix:** flip `reelActive` and `reelShown` synchronously, before the
await, so overlapping calls cannot pick the same buffer or resolve out of
order. Better still, give each call a generation token and have it drop out
after the await if a newer call has started — the last hover is the only one
whose result anyone wants.

**To confirm when fixing:** throttle the network to a slow profile, hard
reload, and sweep the mouse across the demo rail as the page appears. That
is the window the race lives in.

---

## 9. The bottom quality tier is named "Potato"

**Status:** open — a rename, decided with Thomas (2026-08-11), not yet applied
**Found:** 2026-08-11, during the optimization pass

Not a defect; a naming decision recorded so it survives to a session that can
apply it. "Potato" is PC-gaming slang ("can it run on a potato?"), fine while
the tier was buried in a menu. The optimization pass makes it the headline:
startup auto-detection announces the chosen tier in a toast, so on exactly
the machines soar is trying to welcome — a colleague's MacBook — the first
thing the app says is "Auto-detected quality: Potato", which reads as a
verdict on their laptop. The audience is atmospheric scientists following a
link from a talk, not gamers.

**Rename `potato` → `minimal` everywhere.** Behold already uses the sober
scale (`BEHOLD_QUALITY_ROWS`: Min/Low/Medium/High/Max); soar's tiers should
speak the same language. Sites, all anchored by the key `potato`:

- `QUALITY_PRESETS` / `QUALITY_TIER_NAMES` in web/soar/constants.js — the
  preset key, and the label:

```js
  potato: { name: "potato", label: "Potato — smooth stills, rough flight",
```

  The label's second half is stale after the optimization pass anyway: with
  the progressive hold ladder every tier converges to the same full-res
  still, so the bottom tier is no longer "rough" anywhere but in flight.

- `chooseQualityTier` in web/soar/uniforms.js returns `"potato"` as its
  floor.
- The quality panel's special-case copy in web/soar/ui.js ("Potato switches
  to exact High sampling…") — rewrite for the ladder behavior while renaming.
- The parked special case in web/soar/renderer.js (`_applyEffectiveQuality`,
  `this.qualityTier === "potato"`) — being replaced by the ladder in the
  optimization pass; whichever lands second inherits the rename.
- Saved-PNG metadata embeds the tier string (`renderMetadata()` in
  web/soar/viewer.js: `tier: this.renderer.qualityTier`), so old captures
  will say `potato` forever; nothing to migrate, just expect both strings
  when reading old files.

If any personality is wanted, put it in the toast copy, not the tier name —
the name ends up in menus, metadata and reproduction commands.

---

## 10. A "max" tier: multiple spp per frame, gated on measured headroom

**Status:** open — feature design, decided with Thomas (2026-08-11), for
after the optimization pass lands
**Found:** 2026-08-11, during the optimization pass

Every frame today is exactly one march pass (1 spp); quality accrues only by
temporal accumulation across frames. Once the backend optimizations land, a
tier ABOVE high becomes affordable on strong GPUs: same settings, N march
passes per displayed frame, accumulated before presenting. Two payoffs:
flight looks markedly cleaner (the motion blend receives an already-averaged
sample), and a parked view converges to the final still N× faster in
wall-clock.

**Sizing (measured 2026-08-11, RTX 5080, 2560×1440, thick view):** high is
6.3 ms/frame on the 256³ field and 48 ms/frame on the deployed 1024×1024×206
demo. So 4 spp is comfortable on the small field today (~25 ms), and the big
field needs the backend pass first. An M-class Mac never gets there — which
is fine, because:

**This must be probe-gated, never a default.** The startup auto-tier probe
escalates only when the previous rung's measured frame time proves headroom;
max is simply one more rung after high. N passes in one submit lengthen the
non-preemptible GPU burst per frame — the exact mechanism that freezes
laptops (see the optimization pass notes) — so max must never be reachable
except through a measurement, and the manual quality panel should carry the
measured cost next to it.

**Implementation sketch, small once the progressive pass is merged:**

- `QUALITY_PRESETS` in web/soar/constants.js grows `sppPerFrame` (1
  everywhere except max).
- `Renderer.drawFrame` loops the march + accumulate encode pair
  `sppPerFrame` times, advancing the uniform frame index per pass — the
  stratified-lattice jitter streams key off it (`row(4, ..., frameIndex)`),
  so the accumulation mathematics is unchanged: N in-frame passes are
  indistinguishable from N accumulated frames.
- The accumulation plan counts `nextCount += sppPerFrame` instead of `+ 1`.
- The offline capture paths already accumulate explicitly and need nothing.

---

## 11. Cancelling the group/units question reports a load failure

**Status:** open
**Found:** 2026-08-11, deployed web build

**Repro:** menu → "Open a file…" → pick a `.nc` → the group (or units)
question appears → click "Cancel". The app shows the failure panel, headed
"Could not open this field." A deliberate cancel is reported as a fault.

**Wanted instead:** the button says **Back**, and it returns to choosing a
file rather than to an error.

**Cause: cancel is thrown as an error and caught by the same handler as a
real failure.** `Viewer._ask` rejects on cancel:

```js
      const cancel = () => {
        this.ui.close();
        reject(new Error("Cancelled before the field was loaded."));
      };
```

That rejection propagates out of `loadField` into `pickFile`'s catch, which
cannot tell it apart from a corrupt file or a missing variable:

```js
    } catch (err) {
      this.onFailure?.("Could not open this field.",
                       String(err?.message || err), err?.advice || "");
    }
```

So the user reads "Could not open this field. Cancelled before the field was
loaded." — the app calling their own choice an error.

Note that `pickFile` *already* handles this correctly one step earlier. When
the OS file picker itself is dismissed, it returns to where it came from with
no fuss:

```js
    if (!file) { this.paused ? this.ui.open("main") : this.resume(); return; }
```

The question panels simply are not wired to that path.

**Both panels are affected.** `_panel_groups` and `_panel_units` share the
label and the callback:

```js
    m.append(item("Cancel", null, onCancel));
```

**Likely fix:**

1. Make cancel distinguishable from failure — a sentinel error type or a
   resolved `{cancelled: true}` rather than a message string to match on.
   String-matching the message would break the moment the wording changes.
2. On cancel, re-open the file picker so a wrong file can be swapped for the
   right one, and fall back to the main menu if that is dismissed too.
3. Relabel to "Back" in both panels.

**Two things to decide when fixing:**

- Where "Back" goes from the *units* panel when the file also asked a group
  question. Back to the group choice is the honest answer, not back to the
  file picker — but that means the ask sequence needs to be re-enterable
  rather than a straight-line `await` chain.
- `_ask`'s `done()` restores the loading overlay and `cancel()` does not:

  ```js
      const done = (value) => {
        this.ui.close();
        this.setLoadingVisible?.(true);
        resolve(value);
      };
  ```

  That is fine while cancel ends in a failure panel. Once cancel returns to
  the picker instead, the overlay state has to be put back deliberately or
  the next attempt starts with stale chrome.

---

## 12. Choosing "Minimap: fullscreen" from the menu leaves the menu on top of it

**Status:** open
**Found:** 2026-08-11, deployed web build

**Repro:** pause, click the "Minimap" row until it reads `fullscreen`. The
fullscreen map is drawn, and the pause menu stays open over the middle of it.
The map's whole purpose in that mode is to be looked at and clicked, and the
menu covers both.

**Cause.** The menu row toggles the mode and then re-opens itself, in
`_panel_main`:

```js
      () => { app.toggleMinimap(); this.open("main"); }));
```

Re-opening is right for the corner and off modes — the row's own label has to
update, and you stay in the menu. It is wrong for fullscreen, which is a mode
whose entire point is the view underneath.

The mismatch is visible in what the app says at that moment.
`setMinimapMode` announces:

```js
      this.ui.say("Minimap fullscreen — click to travel, M to dismiss.", 4);
```

but a click cannot travel while the menu is over the map, so the instruction
is wrong as given. Pressing `M` from the keyboard reaches the same mode with
no menu in the way, which is why this only shows up via the menu row.

**Likely fix:** close the menu (resume, free-mouse) when the toggle lands on
`fullscreen`, and re-open it only for `corner`/`off`. Worth checking against
`setMinimapMode`'s existing return-to-flight logic, which already handles
leaving fullscreen and knows whether the menu is open.

**Related:** the same screenshot shows bug 7 again — the menu's lower edge
overlapping the hint strip behind it. Same session, different cause; fixing
either does not fix the other.

---

## 13. "Render this view in behold…" should become a terminal-render panel

**Status:** open — design change, not a defect
**Raised:** 2026-08-11

Thomas's words, kept close to verbatim because several of these are taste
calls that should not be paraphrased into something else:

> "Render this view in behold…" should go "Render this view in terminal" and
> then get two options: witness and behold rendering. It would also be nice
> if in the "render view" panel, 1) gui was not in center but along bottom so
> the view you are rendering could be seen 2) controls hint should not be
> shown there and 3) the numerical values for location + view direction have
> text boxes to be changed (and ofc the view updates). Also, the flight video
> doesn't really make sense to be shown here. Witness could include both
> nests, if nests are selected.

**Where it lives now.** `_panel_behold` builds a single behold command from
`app.beholdCommand()`, with a quality segmented control and, when a nest is
loaded, an outer/nest chooser — because behold renders one field only:

```js
      m.append(el("div", "row",
        "behold renders one field, not a nested pair. The other one will " +
        "be absent from its frame."));
```

**The point about nests is the substantive one.** That caveat is a behold
limitation, not a general one. `witness` drives the same WGSL the browser
does and does render the nested pair, so a witness command can reproduce
what is actually on screen — nest included — where a behold command cannot.
Offering both turns that row from an apology into a choice: behold for path
tracing one field, witness for exactly this view. The outer/nest chooser
should then appear only on the behold branch.

**Layout items 1 and 2 are the same underlying want** — see the view you are
about to render. Item 2 ("controls hint should not be shown there") is the
`#hint` element from bug 7; a `menu-open` rule would take care of it there
and here at once, so do bug 7 first and check whether item 2 is already
done. Item 1 needs this panel to opt out of the centred `#menu` box, which is
a layout mode `#menu` does not currently have.

**Item 3 is the largest piece and worth scoping separately.** Editable camera
numbers that drive the live view means the panel becomes an input surface,
not just a printed command: parse, validate, clamp to the domain, push into
`Viewer.camera`, reset accumulation, and re-render the command string on
every change. Consider whether it wants to be its own panel rather than more
rows on this one.

**One thing to resolve before starting: "the flight video doesn't really make
sense to be shown here."** There is no flight-video control in
`_panel_behold` today — video lives in `_panel_capture` and `_panel_track`.
So this is either (a) a note that video must not be folded into the new
terminal-render panel as a third option, or (b) a remark about a different
panel that landed in this list. Ask before designing around it.

---

## 14. Firefox crashes on teardown: "Queue[Id(2,4)] does not exist"

**Status:** open — browser crash, one occurrence, not reproduced
**Found:** 2026-08-11, https://thomasddewitt.com/thought-cloud/soar/

**Repro (once):** do a render, go back to the start page, then switch to
another application. Firefox dies.

**Crash fingerprint**

| | |
|---|---|
| `MozCrashReason` | `Queue[Id(2,4)] does not exist` |
| Signal | `SIGSEGV / SEGV_MAPERR` at `0x0`, thread 0 |
| Process | **main** (not content, not GPU) |
| Crash ID | `806f15b8-4608-4154-b813-b67338fe8be5` |
| Firefox | 153.0.1, release, Fedora 44, Wayland, GNOME |
| GPU | RTX 5080, nvidia 595.80.0.0, `gpuProcess: unused`, WebRender |
| Second adapter | `0x1002:0x13c0` (AMD iGPU) — dual-GPU machine, inactive |

**We already know this error.** `Viewer.dispose` names it verbatim:

```js
    // Unconfigure before destroy: the swapchain holds textures on this device,
    // and leaving it configured against a destroyed device is the state
    // Firefox's own error message calls "Queue[Id] does not exist".
    this.context?.unconfigure();
    this.device.destroy();
```

So the ordering fix is in and the crash happened anyway. That makes this
either a second route into the same state, or a case where those two lines
never ran.

**Leading hypothesis: teardown stalls before it reaches them.** `dispose`
awaits the queue draining first:

```js
    try {
      await this.device.queue.onSubmittedWorkDone();
    } catch {
```

Everything above — including the `unconfigure()`/`destroy()` pair — is behind
that await. A render submits a lot of work, so right after one is exactly
when the drain is slowest, and the reported sequence puts the app switch
inside that window. This codebase has already been bitten by a GPU promise
that does not settle in a backgrounded tab; `showReel` carries the note:

```js
    // Wait for `load`, not `decode()`. decode() is the nicer primitive —
    // it resolves when the bitmap is ready to paint, so the fade never
    // reveals a half-drawn image — but it does not settle at all in a
    // backgrounded tab, which left the backdrop black with nothing logged.
```

If `onSubmittedWorkDone()` behaves the same way when the window is occluded,
the swapchain stays configured against a device nobody will now destroy —
precisely the state the comment above warns about.

**Second candidate: a capture outliving the device.** `dispose` sets
`_videoAbort` and the video path checks it, but `saveScreenshot` has no such
check — it awaits `renderStill` and then the PNG encode with no way to be
told the viewer is gone. A still's readback can therefore still be pending
when the device is destroyed.

**This is also a Firefox bug, and worth reporting as one.** A content page
must not be able to segfault the browser's *main* process; the correct
behaviour for a stale queue ID is a lost device, not a null deref. The crash
ID above identifies the report already submitted. Being a dual-GPU Wayland
box with `gpuProcess: unused` may matter — WebGPU faults land in-process
there, with nothing to contain them.

**Follow-up (2026-08-11, same day, local build of the optimization pass):**
recurred repeatedly (~2 min apart) while flying `localhost:8765/soar/` in
Firefox 153.0.1 — new variant of the fingerprint, this time a Rust
assertion rather than the "does not exist" message:

```
MozCrashReason: assertion `left == right` failed: Queue[Id(0,1)] is no longer alive
  left: 1
 right: 2
```

`SIGSEGV / SEGV_MAPERR` at `0x0`, thread 0, process **main**, crash event
`558d6766-e22c-4934-9576-8ae012565c8c`, same box/driver as above.

**Why the frequency jumped:** the optimization pass had multiplied
queue-wait traffic — probe calibration waited on the queue at every field
load, and every hold-ladder rung climb reallocated the render targets
behind a drain-and-destroy. Each `onSubmittedWorkDone` on this Firefox is a
~100 ms poll AND a spin of the lifetime-assert wheel. Mitigated the same
day (commit "the ladder stops draining the queue"): render targets are now
POOLED per rung size (drain only on a genuine resize past the pool cap),
the probe-clock calibration verdict is cached per session, and
`_releaseField`'s drain is guarded like `dispose`'s. Queue waits during
ordinary flight are now zero once the probe has settled. The crash itself
remains Firefox's to fix — if it recurs at this reduced exposure, collect
the crash ID from about:crashes and it belongs upstream with both
fingerprints.

**What to try when fixing:**

1. Put a timeout on the drain, so `unconfigure()` and `destroy()` run even if
   `onSubmittedWorkDone()` never settles. Losing the drain is survivable;
   skipping the teardown is what crashes.
2. Unconfigure the canvas *before* awaiting anything, rather than after.
3. Give `saveScreenshot` the abort check the video path has, and have
   `dispose` wait for (or cancel) an in-flight capture.
4. Try to reproduce with `visibilitychange` — render, exit to the start page,
   and background the tab in the same beat.
