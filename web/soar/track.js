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
//   {"schema": "cloudyview.track.v1",
//    "header": <render metadata>,
//    "samples": [[t, x, y, z, azimuth, elevation, fov], ...]}
// with the camera fields in the relative-coordinate convention.

"use strict";

export const TRACK_SCHEMA = "cloudyview.track.v1";

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
 * Returns `[{t, position, azimuth, elevation, fov}, ...]` with position in
 * relative coordinates. Azimuth is unwrapped before interpolation so 359 to 1
 * goes through 0 rather than the long way round; in a periodic domain the
 * x/y wrap is unwrapped the same way and re-wrapped afterwards.
 */
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

  const column = (j) => unique.map((s) => s[j]);
  const times = column(0);
  let x = column(1), y = column(2);
  if (periodic) { x = unwrapPeriodic(x); y = unwrapPeriodic(y); }
  const az = unwrapDegrees(column(4));

  const t0 = times[0], t1 = times[times.length - 1];
  const count = Math.floor((t1 - t0 + 1e-9) * fps) + 1;
  const tOut = new Float64Array(count);
  for (let k = 0; k < count; k++) tOut[k] = t0 + k / fps;

  const cols = {
    x: catmullRom(times, x, tOut),
    y: catmullRom(times, y, tOut),
    z: catmullRom(times, column(3), tOut),
    az: catmullRom(times, az, tOut),
    el: catmullRom(times, column(5), tOut),
    fov: catmullRom(times, column(6), tOut),
  };
  if (periodic) {
    for (const key of ["x", "y"]) {
      for (let k = 0; k < count; k++) {
        cols[key][k] = (((cols[key][k] + 1.0) % REL_PERIOD) + REL_PERIOD)
                       % REL_PERIOD - 1.0;
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
    });
  }
  return frames;
}

/**
 * The in-flight recorder: appends one sample per rendered frame.
 *
 * Samples are the camera's own relative coordinates, taken after the frame
 * that showed them, so the track describes what was on screen rather than
 * what was about to be.
 */
export class TrackRecorder {
  constructor() {
    this.samples = [];
    this.startedAt = null;
  }

  get recording() { return this.startedAt !== null; }
  get duration() {
    return this.samples.length ? this.samples[this.samples.length - 1][0] : 0.0;
  }

  start(nowSeconds) {
    this.samples = [];
    this.startedAt = nowSeconds;
  }

  /** One sample. `camera` is the live FlightCamera. */
  sample(nowSeconds, camera) {
    if (this.startedAt === null) return;
    const rel = camera.relativePosition();
    this.samples.push([
      nowSeconds - this.startedAt,
      rel[0], rel[1], rel[2],
      camera.azimuth, camera.elevation, camera.fov,
    ]);
  }

  stop() {
    this.startedAt = null;
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
