# Soar — bug log

Running list of bugs found while flying the deployed build. Not being fixed
right now; this file is the queue for a later session. Newest at the bottom.

**Every line number below is against commit `e150a47` ("soar ships as one
folder"), which is exactly what was deployed when these were found.** Line
numbers drift; the quoted snippets do not. If a number no longer lands, grep
the snippet.

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

**Status:** open, by design today — this is a change of design, not a defect
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

**Status:** open, strong candidate cause found by reading — Thomas can
reproduce on request
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
