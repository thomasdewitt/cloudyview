// Viewer methods (autotier). Choosing the quality tier by measurement: the
// startup probe and its two clocks, then the in-flight governor.

"use strict";

import * as K from "../constants.js";
import { escalateQualityTier } from "../uniforms.js";

export const autotier = {
  /**
   * The tier the first frame of a newly loaded field is drawn at.
   *
   * When the probe is going to run this is the FLOOR, and that is a safety
   * property rather than a preference. The deployed demo field is
   * 1024x1024x206; a high-tier frame of it at Retina resolution is most of a
   * second of fragment work that nothing can preempt, which on a Mac freezes
   * the whole machine (the compositor shares the GPU) and can lose the device
   * to Metal's command-buffer watchdog. The old default of "high, and never
   * adapt" is exactly that frame, submitted before anything has been
   * measured. So: start at the floor, and never render a tier that a
   * completed measurement below it has not shown to be affordable.
   */
  _startingQualityTier() {
    if (!this._autoTier) return this.qualityTier ?? K.DEFAULT_QUALITY_TIER;
    return K.QUALITY_TIERS_CHEAPEST_FIRST[0];
  },

  /** Arm the probe for a freshly loaded field, unless the user opted out. */
  _beginAutoTier() {
    clearTimeout(this._confirmTimer);
    this._confirmTimer = null;
    this._confirmed = false;
    this._governor = null;
    if (!this._autoTier) { this._probe = null; return; }
    this._armProbe(K.QUALITY_TIERS_CHEAPEST_FIRST[0]);
  },

  _armProbe(tier) {
    this._probe = {
      tier, frame: 0, samples: [],
      // Which stopwatch to believe — decided by _chooseProbeClock at the
      // first probe frame, by measuring the stopwatch itself. "queue" is the
      // drain-submit-drain GPU clock; "cadence" is rAF wall time.
      clock: null,
      overheadMs: 0,
      // The tier climbed FROM into the current one, so the cadence clock can
      // step back a rung when the tier it climbed to breaks the beat.
      climbedFrom: null,
      // Median measured at each tier that finished sampling, so settling on
      // a stepped-back tier reports that tier's own number.
      medians: {},
    };
  },

  /**
   * Decide which clock the probe may believe, by timing the clock itself.
   *
   * See the AUTO_TIER_CLOCK_* block in constants.js for the whole argument.
   * The short of it: onSubmittedWorkDone is only a GPU fence on some
   * browsers. On Firefox it resolves on an internal poll cadence — ~100 ms
   * on a machine whose actual frame is one or two — and a probe that
   * believed it concluded a 5080 could not afford Minimal (then "potato"). So the round-trip
   * is measured here on an EMPTY queue: anything it costs is the clock's
   * own overhead. The minimum of a few rounds, because the first may still
   * be draining real work from the load.
   *
   * `?probeclock=queue|cadence` in the URL forces the choice, for testing
   * either path on a browser whose calibration would pick the other.
   */
  async _chooseProbeClock() {
    const probe = this._probe;
    const forced = new URLSearchParams(location.search).get("probeclock");
    if (forced === "queue" || forced === "cadence") {
      probe.clock = forced;
      console.info(`soar: probe clock forced to '${forced}' by ?probeclock=`);
      return;
    }
    // One calibration per session. The verdict is a property of the browser
    // and device, not of the field, so a hot-swapped file reuses it — which
    // also keeps queue waits (a crash lottery on Firefox, see bug 14) off
    // the reload path.
    if (this._probeClockVerdict) {
      probe.clock = this._probeClockVerdict.clock;
      probe.overheadMs = this._probeClockVerdict.overheadMs;
      return;
    }
    const remember = () => {
      this._probeClockVerdict = {
        clock: probe.clock, overheadMs: probe.overheadMs,
      };
    };
    const rounds = [];
    try {
      for (let i = 0; i < K.AUTO_TIER_CLOCK_CALIBRATION_ROUNDS; i++) {
        const t0 = performance.now();
        await this.device.queue.onSubmittedWorkDone();
        rounds.push(performance.now() - t0);
      }
    } catch (err) {
      // The queue clock does not even run here (headless Chrome rejects it
      // outright for swapchain frames). The cadence clock owes it nothing.
      probe.clock = "cadence";
      remember();
      console.info(
        `soar: onSubmittedWorkDone rejects on this browser ` +
        `(${err?.message || err}); probing by frame cadence.`);
      return;
    }
    const overhead = Math.min(...rounds);
    if (overhead <= K.AUTO_TIER_CLOCK_OVERHEAD_MAX_MS) {
      probe.clock = "queue";
      probe.overheadMs = overhead;
      remember();
    } else {
      probe.clock = "cadence";
      remember();
      console.info(
        `soar: waiting on an EMPTY queue costs ${overhead.toFixed(1)} ms ` +
        `here, so onSubmittedWorkDone is a poll cadence, not a fence — ` +
        `probing by frame cadence instead.`);
    }
  },

  /**
   * Account for one probe frame. `ms` is wall time across a frame that was
   * followed by device.queue.onSubmittedWorkDone(), so it is that frame's GPU
   * time — rAF deltas cannot be used here, because they saturate at the
   * vsync interval and are pipelined besides.
   *
   * Strictly one rung at a time: the escalation below happens only after a
   * full measurement has come back, and the next frame is the first one drawn
   * at the new tier. No frame is ever in flight at a tier whose affordability
   * is still being established.
   */
  _probeFrame(ms) {
    const probe = this._probe;
    if (!probe) return;
    probe.frame += 1;
    const warmup = probe.clock === "cadence"
      ? K.AUTO_TIER_WARMUP_FRAMES_CADENCE
      : K.AUTO_TIER_WARMUP_FRAMES;
    if (probe.frame <= warmup) return;
    probe.samples.push(ms);
    if (probe.samples.length < K.AUTO_TIER_SAMPLE_FRAMES) return;

    // The median, not the mean and not the minimum. A mean is dragged by any
    // one stall — a GC pause, the compositor taking the GPU — and every such
    // contamination is upward, so the mean of three is biased high. The
    // minimum over-corrects the other way: on a machine whose frame times are
    // genuinely spread, the best of three is a frame time it will rarely hit
    // again, and escalating on it is exactly the wrong call on exactly the
    // hardware this protects. The middle sample is neither.
    const measured = [...probe.samples].sort((a, b) => a - b)[
      Math.floor(probe.samples.length / 2)];
    probe.medians[probe.tier] = measured;

    const climb = (next) => {
      probe.climbedFrom = probe.tier;
      probe.tier = next;
      probe.frame = 0;
      probe.samples = [];
      this.renderer.setQualityTier(next);
    };

    if (probe.clock === "cadence") {
      // The cadence clock cannot predict the next tier (it saturates at the
      // vsync interval), only falsify the current one: frames arriving on
      // the beat mean this tier is affordable, frames off it mean it is not.
      // So: climb while the beat holds, and when the tier climbed TO breaks
      // it, step back to the one that held. See AUTO_TIER_CADENCE_HOLD_MS.
      const order = K.QUALITY_TIERS_CHEAPEST_FIRST;
      const at = order.indexOf(probe.tier);
      const holds = measured <= K.AUTO_TIER_CADENCE_HOLD_MS;
      if (holds && at < order.length - 1) {
        climb(order[at + 1]);
        return;
      }
      let tier = probe.tier;
      if (!holds && probe.climbedFrom) {
        tier = probe.climbedFrom;
        this.renderer.setQualityTier(tier);
      }
      this._probe = null;
      this._settleAutoTier(tier, probe.medians[tier] ?? measured, "cadence");
      return;
    }

    const next = escalateQualityTier(probe.tier, measured);
    if (next) {
      climb(next);
      return;
    }
    this._probe = null;
    this._settleAutoTier(probe.tier, measured, "queue");
  },

  /**
   * Wait for the GPU queue to go idle, for timing. True if it did.
   *
   * This is the probe's clock and it is allowed to fail. `onSubmittedWorkDone`
   * is core WebGPU, but it is not universally honoured: headless Chrome
   * rejects it with "A valid external Instance reference no longer exists"
   * for every frame that touched the canvas swapchain — reproducible in
   * twenty lines with no soar involved — and a driver that behaves like that
   * in the field is entirely possible.
   *
   * What must NOT happen is what happened before this existed: the rejection
   * reaching the frame's own catch, which treats a throw as a lost device and
   * stops rendering for good. A stopwatch that breaks is not a renderer that
   * broke. So the failure is caught here and ends the probe instead — and
   * ending the probe means staying at the tier last PROVEN affordable, which
   * is the floor unless a completed measurement already justified more. No
   * measurement, no escalation: that is the same invariant the whole probe
   * runs on, arriving at the same answer from the other direction.
   */
  async _awaitQueueIdle() {
    try {
      await this.device.queue.onSubmittedWorkDone();
      return true;
    } catch (err) {
      // The queue clock died mid-probe. That is a broken stopwatch, not a
      // broken renderer — and the probe still has its other clock. Demote to
      // cadence, restart the current tier's samples (they were taken with a
      // clock that just proved untrustworthy), and carry on climbing.
      const probe = this._probe;
      if (probe) {
        probe.clock = "cadence";
        probe.frame = 0;
        probe.samples = [];
        console.warn(
          `soar: the queue clock failed mid-probe ` +
          `(${err?.message || err}); continuing by frame cadence.`);
      }
      return false;
    }
  },

  /** Record the probe's verdict and say it out loud. */
  _settleAutoTier(tier, measuredMs, clock) {
    this.qualityTier = tier;
    // A low verdict gets one silent second opinion before it is announced.
    // Probe noise is one-sided: load turbulence — the volume upload's tail,
    // shader warmup, GC, the compositor — can only make a machine look
    // SLOWER than it is, never faster. So a verdict at or above
    // AUTO_TIER_CONFIRM_FROM proves itself, while a 5080 that measured as
    // Minimal (seen ~10% of loads) deserves re-measuring once the
    // turbulence has passed. The retry walks upward from the tier already
    // proven — same rung-at-a-time invariant, so it can only raise the
    // answer, and it never runs when the user has picked a tier by hand.
    const order = K.QUALITY_TIERS_CHEAPEST_FIRST;
    if (order.indexOf(tier) < order.indexOf(K.AUTO_TIER_CONFIRM_FROM)
        && !this._confirmed) {
      this._confirmed = true;
      console.info(
        `soar: auto quality ${tier} measured ${measuredMs.toFixed(2)} ms ` +
        `(${clock} clock) — low verdict, re-confirming in ` +
        `${K.AUTO_TIER_CONFIRM_DELAY_MS} ms`);
      this._confirmTimer = setTimeout(() => {
        this._confirmTimer = null;
        if (!this._autoTier || this._probe || this._disposed) return;
        this._armProbe(this.qualityTier);
        this._wake("auto-tier confirm");
      }, K.AUTO_TIER_CONFIRM_DELAY_MS);
      return;
    }
    // The verdict is final now, so the tier's own haze, smoothing and LOD
    // come with it — not during the probe's walk, which passes through tiers
    // it has no intention of settling on.
    this._applyTierDefaults(tier);
    console.info(
      `soar: auto quality ${tier} — ${measuredMs.toFixed(2)} ms/frame probed ` +
      `at ${this.canvas.width}x${this.canvas.height} (${clock} clock)`);
    // The verdict, said out loud — once, here, whatever the mode.
    //
    // The probe's answer is provisional and the governor announces the
    // settled one, but the governor only ever speaks when it MOVES, and
    // QUALITY_TIERS_CHEAPEST_FIRST tops out at High: a machine that probes
    // straight to High has no rung left to climb, so it never moves and
    // would say nothing at all — no message about quality on the whole
    // flight, on exactly the hardware most likely to be flying. Announcing
    // the verdict costs one toast and restores the invariant that the tier is never
    // changed silently, which the governor's own two toasts assume.
    //
    // The floor's old "expect a slideshow" stays retired: that was a
    // judgement about the machine, and the fps readout is the honest form of
    // it.
    this._announceAutoTier(
      `set to ${this._tierLabel(tier)} — change any time in the menu`);
    this._armGovernor();
  },

  //
  // The probe's cadence rule transplanted to steady state — see the
  // GOVERNOR_* block in constants.js. Fed one rAF delta per marched,
  // camera-moving frame by _frame; anything else resets the window, so a
  // verdict only ever comes from a consecutive stretch of real flight.

  _armGovernor() {
    if (!this._autoTier) { this._governor = null; return; }
    this._governor = {
      frame: 0, samples: [], startedAt: 0,
      // The rung climbed FROM into the current one. Non-null means the
      // current tier is on trial: breaking the beat steps back there.
      trialFrom: null,
      // tier -> performance.now() before which a rung that failed its trial
      // is not climbed onto again, and tier -> how many trials it has failed.
      // Two strikes refuses the rung for the rest of this field: a rung that
      // failed twice, a minute apart, is not going to be saved by a third
      // look, and re-trying it forever means a sluggish stretch every
      // cooldown for as long as the flight lasts.
      refusedUntil: {},
      refusals: {},
    };
  },

  _governorReset() {
    const g = this._governor;
    if (!g) return;
    g.frame = 0;
    g.samples = [];
  },

  _governorFrame(ms) {
    const g = this._governor;
    if (!g || this._probe || !this._autoTier) return;
    // A hand-set render scale — or any Advanced override — would be stomped
    // by a tier change (a preset click resets everything, and that is the
    // rule), so the governor stands down rather than undoing the user's
    // hand.
    if (this.renderer.qualityIsCustom || this.qualityCustom) {
      this._governorReset();
      return;
    }
    g.frame += 1;
    // The rAF delta describes the PREVIOUS frame, and the first frames after
    // a reset may describe a hold rung or a render-target reallocation —
    // the same lag the cadence probe warms up over.
    if (g.frame <= K.AUTO_TIER_WARMUP_FRAMES_CADENCE) return;
    if (g.samples.length === 0) g.startedAt = performance.now();
    g.samples.push(ms);
    if (g.samples.length < K.GOVERNOR_MIN_FRAMES
        || performance.now() - g.startedAt < K.GOVERNOR_MIN_SECONDS * 1000) {
      return;
    }
    const median = [...g.samples].sort((a, b) => a - b)[
      Math.floor(g.samples.length / 2)];
    const order = K.QUALITY_TIERS_CHEAPEST_FIRST;
    const at = order.indexOf(this.qualityTier);
    const holds = median <= K.AUTO_TIER_CADENCE_HOLD_MS;
    if (!holds && g.trialFrom) {
      // The climbed-to tier failed its trial: back to the rung that held,
      // and this one is not retried until the cooldown passes — or at all,
      // after a second strike.
      const refused = this.qualityTier;
      g.refusals[refused] = (g.refusals[refused] ?? 0) + 1;
      g.refusedUntil[refused] = g.refusals[refused] >= 2
        ? Infinity
        : performance.now() + K.GOVERNOR_RETRY_COOLDOWN_MS;
      this._governorSetTier(g.trialFrom, refused, median, "stepped back",
        `set to ${this._tierLabel(g.trialFrom)} — change any time in the menu`);
      g.trialFrom = null;
    } else if (holds) {
      const wasTrial = g.trialFrom !== null;
      g.trialFrom = null;   // this tier is proven now, trial or not
      const next = at >= 0 && at < order.length - 1 ? order[at + 1] : null;
      if (next && (g.refusedUntil[next] ?? 0) <= performance.now()) {
        const from = this.qualityTier;
        this._governorSetTier(next, from, median, "raised",
          `testing ${this._tierLabel(next)}`);
        g.trialFrom = from;
      } else if (wasTrial) {
        // A trial that held and has nowhere further to climb right now:
        // the testing toast gets its answer.
        console.info(
          `soar: auto quality settled at ${this.qualityTier} — measured ` +
          `${median.toFixed(2)} ms/frame in flight`);
        this.ui.say(`Auto quality: set to ${this._tierLabel(this.qualityTier)}` +
                    ` — change any time in the menu.`, 4);
      }
    }
    this._governorReset();
  },

  _tierLabel(tier) {
    return K.QUALITY_PRESETS[tier].label.split(" —")[0];
  },

  /** One governor tier move: renderer, defaults, and a quiet announcement. */
  _governorSetTier(tier, from, medianMs, verb, toast) {
    this.qualityTier = tier;
    this.renderer.setQualityTier(tier);
    this._applyTierDefaults(tier);
    console.info(
      `soar: auto quality ${verb} to ${tier} — ${from} measured ` +
      `${medianMs.toFixed(2)} ms/frame in flight`);
    this._announceAutoTier(toast);
  },

  /**
   * Every auto-quality toast, in one place.
   *
   * say() replaces what is on screen rather than queueing, and the load's own
   * notes — assumed units, a skipped group, flying without the minimap — land
   * a second or two before the probe's verdict does. Wiping those with a
   * sentence about quality loses the messages that exist to break exactly
   * that silence, so while they are still on screen this repeats them above
   * the tier line rather than replacing them.
   */
  _announceAutoTier(line) {
    const message = `Auto quality: ${line}.`;
    if (this._loadNotes && performance.now() < this._loadNotesUntil) {
      this.ui.say(`${this._loadNotes}\n\n${message}`, 8);
      return;
    }
    this.ui.say(message, 4);
  },
};
