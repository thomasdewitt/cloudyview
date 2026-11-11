// Viewer methods (commands). What made this picture: the render metadata and
// the witness and behold commands that reproduce it.

"use strict";

import * as K from "../constants.js";
import { cameraBasis, worldToRelative } from "../camera.js";
import { viewSpansDomainEdge } from "../field.js";

export const commands = {
  /**
   * What made this picture, so the picture can be made again.
   *
   * `look` is what the capture actually marched (_captureLook, with the
   * metered exposure filled in) — the honesty trail: a named-tier capture no
   * longer marches the session's haze, LOD or exposure, so recording the live
   * values would misreport every one of them. A saved track's header is a
   * capture too: the video's look, metered at its first frame (_trackLook).
   */
  renderMetadata(size, look, { lensWater = false } = {}) {
    const preset = K.QUALITY_PRESETS[look.preset];
    const parked = K.QUALITY_PRESETS.high;   // the top hold rung's sampling
    // The cloud-folded relative position, as ever — with the city block
    // below carrying the district, the cloud frame is unambiguous on its own.
    const rel = this.camera.relativePosition();
    // The camera's OWN surface frame, not a recompute from the world
    // position: with two independently-folded frames the recompute is the
    // thing that can drift from what was rendered. Gated on the scene, not
    // on the frame — every camera carries a surface frame now (the day
    // ocean's included), and the city block only means something over a
    // city.
    const cityPos = this.scene.city && this.camera.surfacePosition
      ? [...this.camera.surfacePosition, this.camera.position[2]]
      : null;
    return {
      schema: "cloudyview.render.v1",
      source: {
        path: this.scene.sourceName ?? null,
        ice_path: this.scene.iceSourceName ?? null,
        liquid_var: this.scene.liquidVar ?? null,
        ice_var: this.scene.iceVar ?? null,
        // A coarsened demo's block average (see scene.js), so `soar` re-renders
        // a track over the grid that was flown rather than the fine file.
        // Absent otherwise, like the city block.
        ...(this.scene.coarsen ? { coarsen: this.scene.coarsen } : {}),
      },
      camera: {
        position: rel,
        azimuth: this.camera.azimuth,
        elevation: this.camera.elevation,
        fov: this.camera.fov,
      },
      sun: { azimuth: this.sunAzimuth, elevation: this.sunElevation },
      // Where in the CITY this still was taken, present only over a city.
      // The camera block above is relative to the cloud box, so on its own it
      // does not pin a place: stand a different-sized field over the same
      // tile and the same relative triple is a different district. The tile
      // frame is fixed world metres, so these three numbers do pin one —
      // hand them to night_city_harness.py --city-camera.
      //
      // Additive, and the schema string stays v1: every reader keys on the
      // presence of a block, and cloudyview/render_metadata.py's SCHEMA is
      // the only other copy of that string. Do not bump one without the
      // other.
      ...(cityPos ? {
        city: {
          position_m: cityPos,
          tile_offset_m: [...this.scene.cityOffsetM],
          tile_extent_m: this.scene.oceanTileExtent,
          // The surface offset this render carried (u.surface_offset; the
          // static offset mod the tile until a period crossing moves it).
          // With the folded camera position above, this — not
          // tile_offset_m — is what reproduces the district and the water
          // phase: hand it to --surface-origin.
          surface_offset_m: this._surfaceOffsetM(),
        },
      } : {}),
      render: {
        renderer: "soar-web",
        size,
        // "custom" when the capture carried the session's hand-set look;
        // `quality` is then the preset that supplied lighting method and spp
        // (the live tier), which is also what --quality below names.
        tier: look.tier,
        quality: look.preset,
        render_scale: 1.0,
        // Parked sampling — High's, for every tier (the top hold rung's
        // rule): what the capture actually marched, not the preset's flight
        // stepping.
        step_factor: parked.stepFactor,
        light_step_factor: parked.lightStepFactor,
        light_cache: preset.lightCache,
        periodic: this.renderer.periodic,
        // Lens water (lens.wgsl): on for a track flown with the switch on,
        // false for a still, which is always dry. A track without the key
        // (older than the switch) replays dry.
        lens_water: Boolean(lensWater),
        accumulate_frames: K.PARKED_ACCUM_FRAMES_BY_TIER[look.preset],
        tone_map_gamma: this.toneMapGamma,
        tone_map_white_point: this.toneMapWhitePoint,
        contrast: this.contrast,
        // The look the CAPTURE marched, not the session's (2026-09-01): a
        // named tier is the pure parked preset — its haze default, its
        // parked LOD, a once-metered exposure — and Custom is the session's
        // values, so `look` is right in both cases where the live fields
        // were right in neither. This block has misreported before (the
        // unscaled-constants bug, then the finer-slider bug); the rule now
        // is that everything here comes from `look`.
        haze: look.haze,
        haze_height_dependent: look.hazeHeightDependent,
        // The scene clock the still was taken at (witness --time): for a
        // surface with a clock (the city's steam), and 0 over one without
        // (the sea), as u.scene_clock is.
        scene_time: this.scene.clockPeriod === null ? 0 : this.sceneTime,
        exposure: look.exposure,
        lod_strength: look.lodStrength,
        light_march_lod_degrees:
          K.APP_LIGHT_MARCH_LOD_DEGREES * look.lodStrength,
        view_step_lod_degrees:
          K.APP_VIEW_STEP_LOD_DEGREES * look.lodStrength,
      },
      timestamp: new Date().toISOString(),
      // witness, not behold: the still was rendered by the witness/soar
      // renderer, so this is the command that reproduces THIS image —
      // nests, wrap and image controls included. At the capture's look
      // rather than the live view's, for the same reason as the render
      // block above.
      reproduction_command: this.witnessCommand({
        size,
        lodStrength: look.lodStrength,
        haze: look.haze,
        hazeHeightDependent: look.hazeHeightDependent,
        exposure: look.exposure,
        quality: look.preset,
      }),
    };
  },

  /** True when the view sees wrapped copies — behold's volume does not tile. */
  viewSpansDomainEdge() {
    return viewSpansDomainEdge(
      this.camera.position,
      cameraBasis(this.camera.azimuth, this.camera.elevation),
      this.camera.fov,
      this.canvas.width / this.canvas.height,
      this.scene.bmin, this.scene.bmax);
  },

  /** Group name of the nested level, for the menu and the behold command. */
  get nestName() {
    return this.scene?.nestGroup ?? null;
  },

  /**
   * Which of the loaded fields the behold command names, and the box its
   * coordinates are measured in.
   *
   * behold renders one field from one group, so a nested scene has to be
   * asked about; the outer field is the answer until it is. The box travels
   * with the group because the relative position means "this far across THIS
   * field" — quoting the outer domain's fraction at a nest a fortieth of its
   * width puts the camera kilometres from where it was framed.
   */
  beholdTarget() {
    if (this.scene?.nested && this.beholdField === "nest") {
      return {
        group: this.scene.nestGroup,
        bmin: this.scene.nestBmin,
        bmax: this.scene.nestBmax,
      };
    }
    return {
      group: this.scene?.groupPath ?? null,
      bmin: this.scene?.bmin,
      bmax: this.scene?.bmax,
    };
  },

  beholdGroup() {
    return this.beholdTarget().group;
  },

  /**
   * The loader's resolved selection, as the CLI flags that reproduce it.
   *
   * Everything the loader inferred or asked — variables, axis assignment,
   * coordinate names, units, timestep, an attached ice file — written out
   * unconditionally, on the same argument as the image controls: what a
   * terminal render needs to place the field where this one did must not
   * depend on the CLI's own inference happening to agree. `--group` is NOT
   * here: behold renders the nest group when there is one, so each command
   * names its own.
   *
   * A demo scene carries none of these fields, which is right — its command
   * already names a file the browser never had — except a coarsened bake's
   * `--coarsen`: that command names the FINE source file, and the flag is
   * how the CLI makes the grid the demo shipped from it.
   */
  _selectionFlags(quote) {
    const s = this.scene;
    const flags = [];
    if (s?.coarsen) flags.push("--coarsen", String(s.coarsen));
    if (s?.liquidVar) flags.push("--liquid-water-var", quote(s.liquidVar));
    if (s?.iceFrom) {
      flags.push("--ice-file", quote(s.iceFrom.filename));
      if (s.iceFrom.iceVar) {
        flags.push("--ice-water-var", quote(s.iceFrom.iceVar));
      }
      if (s.iceFrom.group) {
        flags.push("--ice-water-group", quote(s.iceFrom.group));
      }
      if (s.iceFrom.units) {
        // The ice units the browser ASKED for, when the attached file's
        // variable declared none. --units covers only the main file's
        // variables, so without this the answer was dropped and the CLI
        // render of a split-file load failed or guessed where the browser
        // had asked.
        flags.push("--ice-units", quote(s.iceFrom.units));
      }
    } else if (s?.iceVar) {
      flags.push("--ice-water-var", quote(s.iceVar));
    } else if (s?.liquidVar) {
      // A fact, not a default: this view has no ice in it, and the flag
      // keeps it that way even for a file the CLI could infer ice from.
      flags.push("--no-ice");
    }
    // Emitted only when the file has several steps — exactly when the CLI
    // demands the flag and the loader asked the question.
    if (s?.timeDim) flags.push("--timestep", String(s.timestep ?? 0));
    if (s?.unitsAssumed) flags.push("--units", quote(s.unitsAssumed));
    // The coordinate-units answer: which length a units string nothing
    // recognized turned out to mean. Same argument as --units — the CLI's
    // own reading must not be trusted to reach the same answer.
    if (s?.coordUnitsAssumed) {
      flags.push("--coord-units", quote(s.coordUnitsAssumed));
    }
    // The grid-spacing answer, for the axes it placed and only those: the
    // CLI refuses a spacing on an axis with meters of its own, the same rule
    // that made the browser ask for exactly these.
    const gs = s?.gridSpacingAssumed;
    if (gs) {
      for (const axis of ["x", "y", "z"]) {
        if (gs.spacing[axis] !== undefined) {
          flags.push(`--d${axis}`, String(gs.spacing[axis]));
        }
      }
      if (gs.zFirst) flags.push("--z-first", gs.zFirst);
    }
    for (const axis of ["x", "y", "z"]) {
      if (s?.dimNames?.[axis]) {
        flags.push(`--${axis}-dim`, quote(s.dimNames[axis]));
      }
      if (s?.coordNames?.[axis]) {
        flags.push(`--${axis}-coord`, quote(s.coordNames[axis]));
      }
    }
    return flags;
  },

  beholdCommand() {
    const { group, bmin, bmax } = this.beholdTarget();
    const rel = worldToRelative(this.camera.position, bmin, bmax);
    const n = (v) => Number(v).toPrecision(12).replace(/\.?0+$/, "");
    const q = (v) => (/\s/.test(String(v))
      ? JSON.stringify(String(v)) : String(v));
    // A browser never learns where the file it was handed lives, so the
    // path here is a name to be completed in the terminal, not a path.
    const source = this.scene.sourceName ?? "<your-file.nc>";
    return [
      "behold", q(source), this.beholdQuality, "--gpu",
      ...(group ? ["--group", q(group)] : []),
      ...this._selectionFlags(q),
      "--camera-position", n(rel[0]), n(rel[1]), n(rel[2]),
      "--camera-azimuth", n(this.camera.azimuth),
      "--camera-elevation", n(this.camera.elevation),
      "--fov", n(this.camera.fov),
      "--sun-azimuth", n(this.sunAzimuth),
      "--sun-elevation", n(this.sunElevation),
    ].join(" ");
  },

  /**
   * Unlike behold, witness drives the exact WGSL this view was drawn with,
   * so its command reproduces the view completely: both nested levels, the
   * periodic wrap, and the image controls — which are display state a
   * default render would not carry. The image controls are always written
   * out rather than only when non-default, because "exactly this view" must
   * survive a change of defaults.
   *
   * The overrides are the capture's honesty trail: renderMetadata passes the
   * look the capture actually marched (a named tier's parked defaults, or
   * Custom's session values), so the command reproduces the CAPTURE — at
   * the capture's pixel size too, which is not the canvas's. With no
   * overrides — the terminal panel — the live values go out.
   */
  /**
   * The surface offset this view renders with (u.surface_offset, folded into
   * [0, tileExtent)): the same fold-first subtraction packUniforms makes,
   * or null when the camera has no surface frame.
   */
  _surfaceOffsetM() {
    const sp = this.camera.surfacePosition;
    if (!sp) return null;
    const t = this.camera.surfaceTileExtent;
    const fold = (v) => { const r = v % t; return r < 0 ? r + t : r; };
    return [fold(fold(this.camera.position[0]) - sp[0]),
            fold(fold(this.camera.position[1]) - sp[1])];
  },

  /**
   * `--surface-origin` for the reproduction command, or nothing: the
   * baseline is the scene's static offset folded — the value every fresh
   * CLI render uses — so the flag appears exactly when this flight's cloud
   * crossings have moved the surface frame off it.
   */
  _surfaceOriginFlags(n) {
    const off = this._surfaceOffsetM();
    if (!off) return [];
    const t = this.camera.surfaceTileExtent;
    const fold = (v) => { const r = v % t; return r < 0 ? r + t : r; };
    const stat = this.scene.city ? this.scene.cityOffsetM : [0.0, 0.0];
    if (off.every((v, i) => {
      const d = fold(v - fold(stat[i]));
      return Math.min(d, t - d) < 1e-6 * t;
    })) return [];
    return ["--surface-origin", n(off[0]), n(off[1])];
  },

  witnessCommand({ size = [this.canvas.width, this.canvas.height],
                   lodStrength = this.lodStrength,
                   haze = this.haze,
                   hazeHeightDependent = this.hazeHeightDependent,
                   exposure = this.exposure,
                   quality = this.capturePresetTier(this.captureStillTier),
                 } = {}) {
    const n = (v) => Number(v).toPrecision(12).replace(/\.?0+$/, "");
    const q = (v) => (/\s/.test(String(v))
      ? JSON.stringify(String(v)) : String(v));
    const source = this.scene.sourceName ?? "<your-file.nc>";
    const rel = worldToRelative(
      this.camera.position, this.scene.bmin, this.scene.bmax);
    return [
      "witness", q(source),
      "--size", size[0], size[1],
      ...(this.scene.groupPath ? ["--group", q(this.scene.groupPath)] : []),
      ...(this.scene.nested && this.scene.nestGroup
          ? ["--nest-group", q(this.scene.nestGroup)] : []),
      ...this._selectionFlags(q),
      "--camera-position", n(rel[0]), n(rel[1]), n(rel[2]),
      "--camera-azimuth", n(this.camera.azimuth),
      "--camera-elevation", n(this.camera.elevation),
      "--fov", n(this.camera.fov),
      "--sun-azimuth", n(this.sunAzimuth),
      "--sun-elevation", n(this.sunElevation),
      // The scene clock the view was held at: the city's steam. A surface
      // with no clock (the sea) reads none — u.scene_clock is 0
      // there — so it writes none.
      ...(this.scene.clockPeriod === null ? [] : ["--time", n(this.sceneTime)]),
      // What the world stands on, always written: without it a cyberpunk
      // still's command rendered the day ocean. The city's tile offset picks
      // the district under the field, and the CLI has no default to agree
      // with (cyberpunk and the night-city harness park different ones), so
      // it travels with the surface.
      "--surface",
      this.scene.city ? "city" : "ocean",
      ...(this.scene.city
          ? ["--city-offset", n(this.scene.cityOffsetM[0]),
             n(this.scene.cityOffsetM[1])] : []),
      ...(this.renderer.periodic ? ["--periodic"] : []),
      // Only when period crossings moved this view off the scene's static
      // surface offset: the folded camera position above no longer says
      // which district is underneath, so the offset has to.
      ...this._surfaceOriginFlags(n),
      "--exposure", n(exposure),
      "--gamma", n(this.toneMapGamma),
      "--white-point", n(this.toneMapWhitePoint),
      "--contrast", n(this.contrast),
      "--haze", n(haze),
      ...(hazeHeightDependent
          ? ["--haze-height-dependent"] : ["--no-haze-height-dependent"]),
      // Always written, like the image controls above: the quality tier moves
      // this slider on its own, so a command without it reproduces the view
      // only by luck of which tier was in force.
      "--lod", n(lodStrength),
      // A PRESET name, never "custom" — the CLI has no custom row; a Custom
      // capture reproduces as its preset tier with the look flags above
      // carrying the hand-set values. The preset supplies parked sampling,
      // the light cache and the parked sample count, the same recipe
      // the in-app capture marches.
      "--quality", quality,
    ].join(" ");
  },
};
