// Flight-track recording and resampling. Port of cloudyview/soar/track.py.
//
// Recording captures only the TRACK — per-frame (t, camera) samples plus the
// render header — not pixels. The video pass then resamples that track at an
// exact output frame rate and renders each frame with full converged temporal
// accumulation, so the result has none of the in-flight motion speckle. That
// is the whole reason a screen recording is not the same thing: a hand-flown
// sample stream is irregular, and every output frame here takes as long as it
// takes without changing the video's timing by a millisecond.
//
// Track schema is shared with the desktop, so a track recorded in the browser
// can be re-rendered by `render_track` in Python and vice versa:
//   {"schema": "cloudyview.track.v2",
//    "header": <render metadata>,
//    "samples": [[t, x, y, z, azimuth, elevation, fov, sx, sy], ...]}
// with x/y/z in the relative-coordinate convention — the CLOUD frame, folded
// at the cloud period, exactly as v1 wrote them — and sx/sy the camera's
// position in the scene's surface-tile frame, in TILE-RELATIVE units
// (surfacePosition / tileExtent, so the period is 1.0 and the schema needs
// no tile size). The two frames wrap at independent periods, which is why a
// sample carries both: fold one into the other and a recorded city flight
// replays over the wrong district — or the day ocean teleports under a
// replayed crossing. Every scene's camera carries a surface frame (the day
// ocean's included), so v2 samples are 9 columns with the first seven
// byte-identical to v1's; a camera without one writes 7 columns, and a v1
// track loads and resamples the same way.

"use strict";

export const TRACK_SCHEMA = "cloudyview.track.v2";

// Relative x/y span the domain as [-1, 1], so a periodic flight-path wrap is
// a near-full-span jump between consecutive samples and nothing else is.
const REL_PERIOD = 2.0;
const WRAP_JUMP_THRESHOLD = 1.0;

/** Make a wrapped coordinate continuous across period jumps. */
export function unwrapPeriodic(values, period = REL_PERIOD,
                               threshold = WRAP_JUMP_THRESHOLD) {
  const out = Float64Array.from(values);
  let correction = 0.0;
  for (let i = 1; i < out.length; i++) {
    const jump = values[i] - values[i - 1];
    if (jump > threshold) correction -= period;
    else if (jump < -threshold) correction += period;
    out[i] += correction;
  }
  return out;
}

/**
 * Angles in degrees, made continuous — numpy's unwrap, in degrees.
 *
 * Without this, flying through north makes azimuth jump 359 -> 1 and the
 * interpolated camera spins all the way round the other way in one frame.
 */
export function unwrapDegrees(values) {
  const out = Float64Array.from(values);
  let correction = 0.0;
  for (let i = 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    let dmod = (((d + 180.0) % 360.0) + 360.0) % 360.0 - 180.0;
    if (dmod === -180.0 && d > 0.0) dmod = 180.0;
    if (Math.abs(d) >= 180.0) correction += dmod - d;
    out[i] = values[i] + correction;
  }
  return out;
}

/**
 * Non-uniform (time-parameterized) Catmull-Rom through every sample.
 *
 * Barry-Goldman, with the real sample times as knots, so irregular in-flight
 * frame timing interpolates correctly instead of being treated as evenly
 * spaced — which is what turns a stutter in the recording into a lurch in the
 * video. Endpoints clamp their outer control points.
 */
export function catmullRom(times, values, tOut) {
  const n = times.length;
  const out = new Float64Array(tOut.length);
  let i = 0;                     // tOut is ascending, so the knot only marches
  for (let k = 0; k < tOut.length; k++) {
    const t = tOut[k];
    while (i + 1 < n && times[i + 1] <= t) i++;
    i = Math.min(Math.max(i, 0), n - 2);

    const i0 = Math.max(i - 1, 0), i1 = i, i2 = i + 1, i3 = Math.min(i + 2, n - 1);
    const t0 = times[i0], t1 = times[i1], t2 = times[i2], t3 = times[i3];
    const p0 = values[i0], p1 = values[i1], p2 = values[i2], p3 = values[i3];
    // Degenerate knot spacing (clamped ends, duplicate stamps) falls back to
    // linear inside the segment.
    if (t2 <= t1) { out[k] = p1; continue; }

    const lerp = (pa, pb, ta, tb) =>
      tb <= ta ? pa : pa + (pb - pa) * ((t - ta) / (tb - ta));

    const a1 = lerp(p0, p1, t0, t1);
    const a2 = lerp(p1, p2, t1, t2);
    const a3 = lerp(p2, p3, t2, t3);
    const b1 = lerp(a1, a2, t0, t2);
    const b2 = lerp(a2, a3, t1, t3);
    out[k] = lerp(b1, b2, t1, t2);
  }
  return out;
}

/**
 * Resample hand-flown samples at exact 1/fps steps.
 *
 * Returns `[{t, position, azimuth, elevation, fov, surfacePosition}, ...]`
 * with position in relative coordinates and surfacePosition tile-relative
 * (or null — see below). Azimuth is unwrapped before interpolation so 359 to
 * 1 goes through 0 rather than the long way round; in a periodic domain each
 * wrapped position column is unwrapped the same way at its own frame's
 * period and re-wrapped afterwards.
 */
const frameCountBetween = (t0, t1, fps) =>
  Math.floor((t1 - t0 + 1e-9) * fps) + 1;

/**
 * How many frames `resampleTrack` would produce — without producing them.
 *
 * The menu shows this number the instant a recording stops, purely to say
 * "which becomes 900 frames at 30 fps". Resampling to count was fine while a
 * track was a minute of flying; it is not fine for a track whose length is
 * whatever the clock did, and it is the arithmetic below either way.
 */
export function resampledFrameCount(samples, fps) {
  const times = distinctAscendingTimes(samples, fps);
  return frameCountBetween(times[0], times[times.length - 1], fps);
}

/** The sample times a resample would actually interpolate over. */
function distinctAscendingTimes(samples, fps) {
  if (!(fps > 0)) throw new Error(`fps must be positive; got ${fps}.`);
  const sorted = samples.map((s) => s[0]).sort((a, b) => a - b);
  const distinct = sorted.filter((t, i) => i === 0 || t > sorted[i - 1]);
  if (distinct.length < 2) {
    throw new Error(
      `The track collapses to ${distinct.length} sample(s) with distinct ` +
      "times, which is not enough to interpolate. Fly for longer.");
  }
  return distinct;
}

export function resampleTrack(samples, fps, { periodic = true } = {}) {
  if (!(fps > 0)) throw new Error(`fps must be positive; got ${fps}.`);

  const sorted = samples.map((s) => Array.from(s))
    .sort((a, b) => a[0] - b[0]);
  const unique = sorted.filter((s, i) => i === 0 || s[0] > sorted[i - 1][0]);
  if (unique.length < 2) {
    throw new Error(
      `The track collapses to ${unique.length} sample(s) with distinct ` +
      "times, which is not enough to interpolate. Fly for longer.");
  }
  // 9 columns is a v2 sample (every scene's camera carries a surface frame
  // now, the day ocean's included); 7 is a v1 recording — those predate the
  // surface frame and replay with zero drift (the old behavior, not
  // repairable: the fold already lost which tile copy the flight was in).
  // A mix is a corrupt track, not a choice to make quietly.
  const widths = new Set(unique.map((s) => s.length));
  if (widths.size !== 1 || !(widths.has(7) || widths.has(9))) {
    throw new Error(
      "Track samples must uniformly have 7 columns (v1 / no surface frame) " +
      `or 9 (v2 with one); got widths {${[...widths].join(", ")}}.`);
  }
  const hasSurface = widths.has(9);

  const column = (j) => unique.map((s) => s[j]);
  const times = column(0);
  // Each column unwraps and folds at ITS OWN frame's period: the cloud
  // frame's x/y at REL_PERIOD as ever, the surface frame's sx/sy at 1.0
  // (tile-relative units). Callers do not choose periods — the columns carry
  // their frames, and folding either at the other's period is precisely the
  // district drift the two-frame schema exists to prevent.
  let x = column(1), y = column(2);
  if (periodic) { x = unwrapPeriodic(x); y = unwrapPeriodic(y); }
  const az = unwrapDegrees(column(4));
  let sx = null, sy = null;
  if (hasSurface && periodic) {
    sx = unwrapPeriodic(column(7), 1.0, 0.5);
    sy = unwrapPeriodic(column(8), 1.0, 0.5);
  } else if (hasSurface) {
    sx = column(7); sy = column(8);
  }

  const count = frameCountBetween(times[0], times[times.length - 1], fps);
  const t0 = times[0];
  const tOut = new Float64Array(count);
  for (let k = 0; k < count; k++) tOut[k] = t0 + k / fps;

  const cols = {
    x: catmullRom(times, x, tOut),
    y: catmullRom(times, y, tOut),
    z: catmullRom(times, column(3), tOut),
    az: catmullRom(times, az, tOut),
    el: catmullRom(times, column(5), tOut),
    fov: catmullRom(times, column(6), tOut),
    ...(hasSurface ? { sx: catmullRom(times, sx, tOut),
                       sy: catmullRom(times, sy, tOut) } : {}),
  };
  if (periodic) {
    // Fold back at the same period each unwrap used: x/y centred into
    // [-1, 1) bit for bit as v1 did, sx/sy into [0, 1) — the tile frame's
    // own convention.
    for (const key of ["x", "y"]) {
      for (let k = 0; k < count; k++) {
        cols[key][k] = (((cols[key][k] + 1.0) % REL_PERIOD) + REL_PERIOD)
                       % REL_PERIOD - 1.0;
      }
    }
    for (const key of hasSurface ? ["sx", "sy"] : []) {
      for (let k = 0; k < count; k++) {
        cols[key][k] = ((cols[key][k] % 1.0) + 1.0) % 1.0;
      }
    }
  }

  const frames = [];
  for (let k = 0; k < count; k++) {
    frames.push({
      t: tOut[k],
      position: [cols.x[k], cols.y[k], cols.z[k]],
      azimuth: ((cols.az[k] % 360.0) + 360.0) % 360.0,
      elevation: Math.min(90.0, Math.max(-90.0, cols.el[k])),
      fov: cols.fov[k],
      // Tile-relative, like the samples; null on a track with no surface
      // frame, and the replay then folds against the static offset.
      surfacePosition: hasSurface ? [cols.sx[k], cols.sy[k]] : null,
    });
  }
  return frames;
}

/**
 * A track long enough that it is no longer a flight anyone meant to record.
 *
 * Ten minutes is 18,000 frames at 30 fps, which is already a big video and a
 * big render. The point of the cap is not the video, though — it is that the
 * clock below has to stop somewhere, and a bound stated in seconds of flying
 * is one a person can reason about.
 */
export const MAX_TRACK_SECONDS = 600.0;

/**
 * The in-flight recorder: appends one sample per rendered frame.
 *
 * Samples are the camera's own relative coordinates, taken after the frame
 * that showed them, so the track describes what was on screen rather than
 * what was about to be.
 *
 * The clock is the flight's, not the wall's: it advances by the delta of each
 * rendered frame. Wall time made "start recording, switch to another app for
 * the afternoon, come back" into a track claiming to be five hours long —
 * requestAnimationFrame does not fire in a hidden tab, so no samples describe
 * those hours, but the first one afterwards carried the whole gap in its
 * timestamp. Resampling that at 30 fps asks for half a million frames.
 */
export class TrackRecorder {
  constructor() {
    this.samples = [];
    this.elapsed = 0.0;
    this._recording = false;
  }

  get recording() { return this._recording; }
  get duration() {
    return this.samples.length ? this.samples[this.samples.length - 1][0] : 0.0;
  }
  /** True once the flight has run past what a track is allowed to hold. */
  get full() { return this.elapsed >= MAX_TRACK_SECONDS; }

  start() {
    this.samples = [];
    this.elapsed = 0.0;
    this._recording = true;
  }

  /** Advance the flight clock by one rendered frame. */
  advance(deltaSeconds) {
    if (!this._recording) return;
    this.elapsed += Math.max(0.0, deltaSeconds);
  }

  /** One sample at the current flight time. `camera` is the live FlightCamera. */
  sample(camera) {
    if (!this._recording) return;
    const rel = camera.relativePosition();
    const s = [
      this.elapsed,
      rel[0], rel[1], rel[2],
      camera.azimuth, camera.elevation, camera.fov,
    ];
    // The surface frame, tile-relative: x/y above are folded at the cloud
    // period, which loses which tile copy the flight is in — sx/sy is where
    // that lives (the city district, or the day ocean's phase). Appended, so
    // the first seven columns stay byte-identical to a v1 sample.
    if (camera.surfacePosition) {
      s.push(camera.surfacePosition[0] / camera.surfaceTileExtent,
             camera.surfacePosition[1] / camera.surfaceTileExtent);
    }
    this.samples.push(s);
  }

  stop() {
    this._recording = false;
    return this.samples;
  }
}

/** The JSON a track file holds, ready for download or for Python. */
export function trackPayload(header, samples) {
  return {
    schema: TRACK_SCHEMA,
    header,
    samples: samples.map((s) => Array.from(s, Number)),
  };
}
