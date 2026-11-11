// Cloud uplight: the city as the cloud field's second light source. The
// glow mip under the sample (footprint ~ its height, so the mip level is
// log2(z / cell)) times the two-stream transmittance of the cloud below.
const CITY_UPLIGHT_STRENGTH: f32 = 0.5;
// Only glow above the citywide background lifts a cloud base: the bias is
// what keeps sprawl bases moonlit blue-grey while the district overhead
// burns amber. Without it even a 0.02-radiance warm pedestal reads, because
// the night clouds it lands on sit near 0.05.
const CITY_UPLIGHT_GLOW_BIAS: f32 = 0.15;
const CITY_UPLIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.38, 0.10);
// The glow feeding the uplight saturates: g/(HALF+g). Over the megatower
// district the raw mip runs to ~3x the p99.5 level, and any linear scaling
// bright enough to warm an ordinary district's cloud base bleaches the
// district's whole sky cream — the saturation is what lets both exist.
const CITY_UPLIGHT_GLOW_HALF: f32 = 0.6;
const CITY_UP_PROBE_SPAN: f32 = 1400.0;
const CITY_UP_PROBE_STEPS: i32 = 12;
const CITY_UPLIGHT_FLOOR: f32 = 0.02;
// e-folding of the amber in probe tau: 1/0.25 = 4 optical depths of skin.
const CITY_UPLIGHT_TAU_SCALE: f32 = 0.25;

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let g_hg = u.params.z;
    let ambient_strength = u.params.w;
    let exposure = u.cam_right.w;
    let jitter_on = u.cam_up.w;
    let jitter_scale = clamp(u.flags.y, 0.0, 1.0);
    let sun = u.sun_dir.xyz;
    let gradient_shading_strength = u.cb_realism.x;
    let deep_shadow_ms_suppression = u.cb_realism.y;
    let ambient_occlusion_strength = u.cb_realism.z;
    let bounce_depth_attenuation = u.cb_realism.w;
    let gradient_coarse_weight = u.cb_params.x;
    let gradient_coarse_radius_m = u.cb_params.y;
    let ambient_occlusion_floor = u.cb_params.z;

    // The camera ray, shared with the surface pre-pass (camera_ray), so the
    // surface it traced is the surface this march composites.
    let cr = camera_ray(frag_pos);
    let frame_index = cr.frame_index;
    let strat_pixel = cr.strat_pixel;
    let dir = cr.dir;

    // safe_inv_dir, not 1.0 / dir: an exactly axis-aligned pixel ray would
    // divide by zero, and the infinity makes the slab test below
    // indeterminate (the sun path's same bug, desert-ground 2026-09-24).
    let inv_dir = safe_inv_dir(dir);
    let periodic_on = PERIODIC_DOMAIN;
    var t_near: f32;
    var t_far: f32;
    if (periodic_on) {
        // Tiled domain: only the z slab bounds the march (rays never exit
        // horizontally); the horizontal exit is replaced by the clear-air
        // transmittance / wrap-ceiling cap.
        let tz0 = (u.bmin.z - u.cam_origin.z) * inv_dir.z;
        let tz1 = (u.bmax.z - u.cam_origin.z) * inv_dir.z;
        t_near = max(min(tz0, tz1), 0.0);
        t_far = min(
            max(tz0, tz1),
            periodic_march_cap(u.cam_origin.z, dir)
        );
    } else {
        let hit = ray_box(u.cam_origin.xyz, inv_dir);
        t_near = max(hit.x, 0.0);
        t_far = hit.y;
    }

    // Under CITY the ocean plane is compiled out entirely: the surface below
    // is the traced city, which fills the same role (an opaque floor at
    // t_city) through the same guards.
    let ocean_on = !CITY && u.ocean_params.z > 0.5;
    var t_ocean = 1e30;
    if (ocean_on && dir.z < -1e-8) {
        let t_ocean_candidate = (u.ocean.x - u.cam_origin.z) / dir.z;
        if (t_ocean_candidate > 0.0) {
            t_ocean = t_ocean_candidate;
        }
    }

    // The opaque surface below — the traced city — was
    // found by the surface pre-pass (fs_surface) for this very ray: its
    // fogged or hazed radiance and its distance, one texel per pixel. It is
    // composited when the march reaches it, exactly as the ocean plane is,
    // so the march loop only ever multiplies it by transmittance. Reading it
    // rather than tracing it here keeps every surface register out of this
    // kernel (see the surface pre-pass section).
    var t_city = 1e30;
    var city_rgb = vec3<f32>(0.0);
    if (CITY) {
        let sp = textureLoad(surface_buf, vec2<i32>(frag_pos.xy), 0);
        t_city = sp.w;
        city_rgb = sp.xyz;
    }
    let t_surface = select(t_ocean, t_city, CITY);
    // The veil (the city's steam): composited once,
    // where the march passes its depth — so a cloud in front of it stays in
    // front and a cloud behind it is seen through it — and only if it lies
    // in front of the surface.
    var veil = Veil(vec3<f32>(0.0), 1.0, 1e30);
    if (CITY) {
        veil = veil_unpack(textureLoad(veil_buf, vec2<i32>(frag_pos.xy), 0));
    }
    // Under the live view's steam the veil is only its depth (trans 1), and
    // the march notes the cloud's transmittance there, t_front, in the
    // sample's alpha for fs_steam_live; elsewhere the alpha is 1.
    let steam_live = city_steam_live();
    var veil_pending = CITY && (veil.trans < 1.0 || steam_live)
        && veil.t < t_surface;
    var t_front = 1.0;

    let cos_scatter = dot(dir, sun);
    let phase = hg_phase(cos_scatter, g_hg);
    // Once-scattered diffraction spike (see DIFFRACTION_G). Constant along
    // the ray, like `phase`; the window makes it exactly zero beyond 30 deg
    // from the sun, so any view not containing the near-sun sky is
    // bit-identical.
    let phase_diffraction = DIFFRACTION_WEIGHT
        * hg_phase(cos_scatter, DIFFRACTION_G)
        * smoothstep(DIFFRACTION_WINDOW_COS_START,
                     DIFFRACTION_WINDOW_COS_FULL, cos_scatter);

    // Scattering-geometry weight on the powder term (see POWDER_FWD_*). One
    // value per pixel: cos_scatter is constant along a view ray. 1 keeps the
    // full boundary darkening, 1 - POWDER_FWD_FADE is what survives at exact
    // forward scattering.
    let g2_hg = g_hg * g_hg;
    // clamped short of 1 so the smoothstep below can never be degenerate; the
    // select handles the isotropic case, where no forward cone exists at all
    // and powder must stay whole in every direction.
    let powder_fwd_cos_start = clamp(
        (1.0 + g2_hg - pow(max(1.0 - g2_hg, 1e-6), 2.0 / 3.0))
        / (2.0 * max(g_hg, 1e-4)),
        -1.0, 0.999
    );
    let powder_weight = 1.0 - POWDER_FWD_FADE * select(
        0.0,
        smoothstep(powder_fwd_cos_start, 1.0, cos_scatter),
        g_hg > 1e-4
    );

    // Aerial perspective (witness iter_008): this sightline's horizon sky
    // color (solar disc excluded) — the same asymptotic target the ocean
    // haze uses, so cloud and water converge to one haze color.
    // Ice-detection mode: 0.0 unless research mode has switched it on, and
    // every branch on it below leaves the math bit-identical when off.
    let ice_on = u.light_cache.w > 0.5;
    // In ice mode most of the aerial haze is given up (select is 1.0 with
    // the mode off): full strength converges every distant cloud onto the
    // one horizon color and the phase information drowns in it.
    let aerial_strength = u.sky_horizon.w * select(1.0, 0.3, ice_on);
    var aer = vec3<f32>(0.0);
    if (aerial_strength > 0.0) {
        if (CITY) {
            // At night the horizon asymptote is the host-packed night haze
            // color directly; the daytime sky model has nothing to add.
            aer = u.sky_horizon.xyz;
        } else {
            let ah_len = length(dir.xy);
            var ah = dir.xy;
            if (ah_len > 1e-8) {
                ah = dir.xy / ah_len;
            }
            if (ice_on) {
                // Distance fades into the alien horizon, not the blue one.
                aer = alien_sky(vec3<f32>(ah, 0.0), sun, vec3<f32>(0.0));
            } else {
                aer = sky_radiance(
                    vec3<f32>(ah, 0.0), sun,
                    u.sky_horizon.xyz, u.sky_bloom.xyz, vec3<f32>(0.0),
                    u.cloud_sun.w
                );
            }
        }
    }

    // Jittered first step: decorrelates the sampling shells between
    // neighboring pixels, killing the coherent ring/banding artifact.
    // Stratified over the accumulation (iter_011), so the shells a pixel tests
    // across frames tile its entry interval instead of clumping.
    // The entry step scale is the outer level's: the nest is required to lie
    // strictly inside the outer AABB, so a ray always enters through it.
    let jitter = strat1(frame_index, strat_pixel, STRAT_STREAM_ENTRY,
                        STRAT_ALPHA_3);
    // Independent stream for the sky probe's quadrature offset.
    let probe_jitter = strat1(frame_index, strat_pixel,
                              STRAT_STREAM_SKY_PROBE,
                              STRAT_ALPHA_10) - 0.5;
    // Independent stream for the sun march's quadrature offset (iter_003).
    // Under the same jitter_on switch as the view march: turning jitter off
    // asks for a deterministic march, and this is one, artifact and all.
    let shadow_jitter = jitter_on * jitter_scale
                        * strat1(frame_index, strat_pixel,
                                 STRAT_STREAM_SUN_MARCH,
                                 STRAT_ALPHA_4);
    // Independent stream for the solar-cone draw (iter_007). Two uniforms:
    // squared radius and azimuth on the disc.
    let sun_cone_seed = strat2(frame_index, strat_pixel,
                               STRAT_STREAM_SUN_CONE,
                               vec2<f32>(STRAT_ALPHA_5, STRAT_ALPHA_6));
    let penumbra_tan = SUN_ANGULAR_RADIUS * SUN_CONE_WIDEN
                       * jitter_on * jitter_scale;
    // Built once here, not once per draw: same crosses, same normalize, same
    // order, so every deflected direction is the number it always was.
    let sun_frame = sun_tangent_frame(sun);
    // Independent stream for the forward pre-march's quadrature offset.
    let ahead_jitter = jitter_on * jitter_scale
                       * strat1(frame_index, strat_pixel,
                                STRAT_STREAM_PREMARCH,
                                STRAT_ALPHA_9);
    let entry_dt = u.bmin.w;
    var t = t_near + jitter_on * jitter * jitter_scale * entry_dt;

    var transmittance = 1.0;
    var col = vec3<f32>(0.0);
    var tau_depth = 0.0;
    // Whether the march below shaded the water, tracked rather than inferred
    // (docs/soar-bugs.md 6). The far-water fallback used to decide the same
    // thing from `t_ocean > t_far`, which is only meaningful when the ray hit
    // the box at all: ray_box returns t_near > t_far on a MISS with t_far
    // still a positive finite number — the nearest exit plane of a box the
    // ray never entered. Such a ray skips the march (the guard below wants
    // t_near < t_far) and then failed the fallback too whenever that
    // leftover t_far happened to exceed t_ocean, so the ocean was hit and
    // nothing shaded it. Because t_far varies smoothly with direction that
    // came out as a contiguous region of sky where the water should be,
    // reachable with periodicity off and the camera just outside the domain
    // looking down.
    var ocean_consumed = false;
    // The march reached the water. Shading it is left to the one
    // ocean_shade_dispatch call after the loop rather than done here: the
    // ocean shader carries two sun marches, and inlining it twice into
    // fs_main cost the ocean views 7-9% for nothing (2026-09-23). The two
    // paths computed the same expression on the same values — transmittance
    // is not touched between this break and that call — so it is one call
    // with two ways in.
    var ocean_in_march = false;
    // Optical depth this ray has already crossed, so the pre-march total can
    // be turned into "what is still ahead of this sample". Unlike tau_depth
    // this never resets: it is the ray's own coordinate along its chord.
    var tau_view = 0.0;
    // Whether the previous view sample sat inside the nest, so the march can
    // notice the fine->coarse crossing (docs/soar-bugs.md 4).
    var was_in_nest = false;

    // The periodic march can legitimately cover several domain widths, so
    // it gets more step headroom; the non-periodic bound is untouched.
    let max_view_steps = select(MAX_VIEW_STEPS, MAX_VIEW_STEPS_PERIODIC,
                                periodic_on);

    // How much new hemisphere the sightline probe covers that iter_001's
    // vertical probe did not: the sine of the angle between the two.
    let ahead_novelty = sqrt(max(1.0 - dir.z * dir.z, 0.0));

    // Spectrum of the buried residual of the diffuse fills — per frame, not
    // per sample (it depends only on the two lighting spectra).
    let deep_tint = deep_fill_tint(u.ambient_tint.xyz);

    // One coarse pass over the whole sightline before anything is composited.
    var tau_total = 0.0;
    if (t_near >= 0.0 && t_near < t_far) {
        tau_total = premarch_tau_ahead(
            u.cam_origin.xyz, dir, t_near, min(t_far, t_surface), ahead_jitter
        );
    }

    if (t_near >= 0.0 && t_near < t_far) {
        for (var i: i32 = 0; i < max_view_steps; i = i + 1) {
            // witness.py:621-646 tests ocean before the t_far break so an
            // ocean plane coincident with the box floor is still shaded.
            // The water is shaded after the loop, by the same call that
            // shades water the march never reached (see ocean_in_march).
            if (ocean_on && t >= t_ocean) {
                ocean_in_march = true;
                ocean_consumed = true;
                break;
            }
            if (veil_pending && t >= veil.t) {
                t_front = transmittance;
                col = col + transmittance * veil.rgb;
                transmittance = transmittance * veil.trans;
                veil_pending = false;
            }
            if (CITY && t >= t_city) {
                col = col + transmittance * city_rgb;
                transmittance = 0.0;
                ocean_consumed = true;
                break;
            }

            if (t >= t_far || transmittance < TRANSMITTANCE_CUTOFF) {
                break;
            }
            let p = u.cam_origin.xyz + t * dir;
            // The active level sets the step scale, so the march refines on
            // entering the nest and coarsens on leaving it. The dt-invariant
            // powder term below is what lets the two levels composite
            // without a brightness seam (witness.py header).
            let level = sample_level_at(p);
            // Crossing OUT of the nest, the step length jumps fine -> coarse,
            // but every ray crosses at the same geometric plane and arrives
            // with at most a fine step's worth of phase — so sample phase
            // past the seam covered barely half the coarse period, and the
            // coherent remainder survived accumulation as stripes parallel
            // to the seam (docs/soar-bugs.md 4; the A/B pinned the view
            // march, not the light march). The cure is the file's usual one:
            // an independent per-pixel-per-frame uniform phase, spanning the
            // FULL effective coarse step — adding a full-period uniform
            // modulo the period is uniform whatever the crossing
            // distribution was, where re-using the entry draw or spanning
            // only the step difference both leave a coherent residual
            // (measured; the first attempt did the former and the stripes
            // survived it). The reverse crossing needs nothing: a coarse
            // period of phase folded into a fine one already covers it.
            if (NESTED && was_in_nest && !level.in_nest) {
                was_in_nest = false;
                let seam_jitter = strat1(frame_index, strat_pixel,
                                         STRAT_STREAM_NEST_SEAM,
                                         STRAT_ALPHA_10);
                let skip = jitter_on * seam_jitter * jitter_scale
                           * max(u.bmin.w, t * u.periodic.z);
                if (skip > 0.0) {
                    t = t + skip;
                    continue;
                }
            }
            was_in_nest = level.in_nest;
            let sigma = level.sigma;

            // Distance LOD: the step floor grows with distance so far
            // wrapped copies march in angular, not metric, resolution
            // (0 = fixed dt). Dense-sigma refinement still applies.
            var dt = step_dt_for_sigma(
                sigma, max(level.dt_view, t * u.periodic.z)
            );
            if (t + dt > t_far) {
                dt = t_far - t;
            }
            if (ocean_on && t + dt > t_ocean) {
                dt = max(0.0001, t_ocean - t);
            }
            if (veil_pending && t + dt > veil.t) {
                dt = max(0.0001, veil.t - t);
            }
            if (CITY && t + dt > t_city) {
                dt = max(0.0001, t_city - t);
            }

            let d_tau = sigma * dt;
            if (d_tau < EMPTY_DTAU_CUTOFF) {
                tau_depth = 0.0;
                t = t + dt;
                continue;
            }

            tau_depth = tau_depth + d_tau;
            tau_view = tau_view + d_tau;

            // Ice-detection tint for this sample's cloud source terms.
            // Exactly 1.0 with the mode off, so the multiplies below are
            // bit-exact no-ops.
            var phase_tint = vec3<f32>(1.0);
            if (ice_on) {
                phase_tint = ice_tint(ice_fraction_at(p));
            }

            // Aerial perspective: clear-air transmittance camera->sample via
            // the closed-form exponential atmosphere (witness._render_image).
            var air_t = 1.0;
            if (aerial_strength > 0.0) {
                let aer_beta0 = u.sky_bloom.w;
                let aer_h = u.sky_disc.w;
                let aer_z0 = max(u.cam_origin.z, 0.0);
                let aer_z1 = max(p.z, 0.0);
                let aer_mu = dir.z;
                var tau_air: f32;
                if (aer_h <= 0.0) {
                    tau_air = aer_beta0 * t;          // uniform: no z profile
                } else if (aer_mu > 1e-6 || aer_mu < -1e-6) {
                    tau_air = aer_beta0 * (aer_h / aer_mu)
                        * (exp(-aer_z0 / aer_h) - exp(-aer_z1 / aer_h));
                } else {
                    tau_air = aer_beta0 * t * exp(-aer_z0 / aer_h);
                }
                air_t = exp(-aerial_strength * tau_air);
            }

            // A view ray takes many steps, so its shadow rays advance the
            // offset by the golden ratio instead of reusing one value: the
            // sequence is low-discrepancy along the march, which averages the
            // quadrature noise down within a single frame rather than leaving
            // it all for temporal accumulation.
            let step_shadow_jitter = fract(
                shadow_jitter + f32(i) * 0.6180339887498949
            );
            // Draw this sample's point on the solar disc (iter_007). The R2
            // additive recurrence advances both coordinates along the view
            // march for the same reason the golden ratio advances the
            // quadrature phase above: the many shadow rays that composite into
            // one pixel then cover the disc low-discrepancy within a single
            // frame, so soar's first frame shows a soft edge with light dither
            // rather than a hard edge that only melts once accumulation
            // settles.
            let sun_shadow = sun_cone_dir(
                sun, sun_frame,
                fract(sun_cone_seed + f32(i) * vec2<f32>(
                    0.7548776662466927, 0.5698402909980532
                )),
                penumbra_tan
            );
            // The cache replaces the march, the disc jitter and the
            // distance-LOD coarsening in one move: it is the central-sun,
            // zero-jitter tau at the cache's own resolution, trilinearly
            // filtered. See Uniforms.light_cache.
            var tau_sun: f32;
            if (LIGHT_CACHE) {
                tau_sun = sample_light_tau(p);
            } else {
                tau_sun = light_march_tau(p, sun_shadow, t * u.periodic.y,
                                          step_shadow_jitter);
            }
            let light_transfer_split_strength = u.ambient_tint.w;
            // Unconditional: the diffuse beam and the high-sun skylight
            // consume this gate too, so computing it only when the storm
            // machinery is enabled would hand a buried storm the full
            // shallow treatment when suppression/AO/split are all zero
            // (codex review 2026-08-11, finding 3).
            // Log-space: per-e-fold travel between tau 15 and 1000 (see the
            // DEEP_SHADOW_TAU constants). The max() guard only matters for
            // tau_sun < 1, which is far below the gate's onset anyway.
            let deep_shadow_gate = smoothstep(
                DEEP_SHADOW_LOG_TAU_START, DEEP_SHADOW_LOG_TAU_FULL,
                log(max(tau_sun, 1e-3))
            );

            // Moderate-shadow gate: where the beam and MS octaves are spent
            // but the deep-shadow machinery has not engaged. This is where
            // the skylight fill lives. Exponential onset — see the
            // SHADOW_SKYLIGHT_TAU_ONSET comment for why not a smoothstep.
            let shadow_gate = 1.0 - exp(-tau_sun / SHADOW_SKYLIGHT_TAU_ONSET);

            // Sky visibility, measured (hoisted from the diffuse block below
            // so the MS floor can use it too). Run for any meaningfully
            // shadowed sample — elsewhere tau_sun still carries the shading
            // and this costs nothing. One probe serves every consumer.
            var t_sky = 1.0;
            var shallow_open = 0.0;
            if (shadow_gate > 0.0 || deep_shadow_gate > 0.0) {
                if (!SKY_PROBE) {
                    // Sky probe disabled (a cost/look toggle, J in the
                    // browser): every consumer sees a fully open sky, which
                    // is the neutral the floors already assume. Buried
                    // samples brighten — that is the look being priced.
                    shallow_open = 1.0;
                } else {
                    t_sky = sky_probe_transmittance(p, g_hg, probe_jitter);
                    shallow_open = smoothstep(
                        SHALLOW_OPEN_TSKY_START, SHALLOW_OPEN_TSKY_FULL, t_sky
                    );
                }
            }
            // A buried storm sample keeps the full tuned suppression; an
            // optically open shallow shadow keeps only a quarter of it.
            let storm_weight = mix(
                SHALLOW_SUPPRESSION_KEEP, 1.0, 1.0 - shallow_open
            );

            // Diffusion depth for the isotropic tail (see MS_TAIL_FLOOR).
            // The knee is what keeps this a *contrast* change rather than a
            // dimming: below it the factor is exactly 1, so thin cloud and
            // the directly lit shoulder of a turret are bit-identical to
            // iter_001 and only the genuinely self-shadowed parts move.
            let ms_tail_factor = MS_TAIL_FLOOR + (1.0 - MS_TAIL_FLOOR)
                * diffuse_transmittance(
                    max(tau_sun - MS_TAIL_TAU_KNEE, 0.0), g_hg
                );

            // Local surface frame. Hoisted above the MS loop (iter_006) so the
            // tail can be weighted by the beam's incidence on the skin; the
            // gradient *shading* below consumes exactly these values in exactly
            // the same way it did, so that term is untouched.
            //
            // The stencil is skipped on the entry skin, where it cannot
            // matter. surface_gate's depth factor is a smoothstep that is
            // exactly 0 at or below GRADIENT_SHADING_TAU_START, and with the
            // gate at exactly 0 every consumer of the gradient is exactly
            // neutral: tail_mu_factor is mix(1, x, 0), gradient_factor is
            // max(0.2, 1 + 0) and fill_shape is mix(1, 1, w). Leaving grad at
            // zero takes the grad_len guards below instead, which leave the
            // same three ones in place. That saves twelve texture reads per
            // sample over the first quarter optical depth after every cloud
            // entry — most of a thin field's samples, a few percent of a
            // thick one's.
            var grad = vec3<f32>(0.0);
            var grad_len = 0.0;
            var surface_gate = 0.0;
            if (gradient_shading_strength > 0.0
                && tau_depth > GRADIENT_SHADING_TAU_START) {
                let grad_conf_v = sigma_gradient(
                    p, sigma, gradient_coarse_weight, gradient_coarse_radius_m,
                    t, u.cb_params.w, level.in_nest
                );
                grad = grad_conf_v.xyz;
                grad_len = length(grad);
                surface_gate = smoothstep(
                    GRADIENT_SHADING_TAU_START,
                    GRADIENT_SHADING_TAU_FULL,
                    tau_depth
                ) * smoothstep(
                    GRADIENT_SHADING_CONF_START,
                    GRADIENT_SHADING_CONF_FULL,
                    grad_conf_v.w
                );
            }

            // Beam incidence on the skin, relative to a horizontal face (see
            // the TAIL_MU_* block). Exactly 1 where the field is level, so it
            // costs no brightness on flat cloud top and only the flanks of
            // turrets and the walls of the crevices between them give the tail
            // up.
            var tail_mu_factor = 1.0;
            if (grad_len > 1e-12) {
                let mu0 = max(-dot(grad, sun) / grad_len, 0.0);
                let mu_ratio = clamp(
                    mu0 / max(sun.z, TAIL_MU_REF_MIN), 0.0, 1.0
                );
                tail_mu_factor = mix(
                    1.0,
                    TAIL_MU_FLOOR + (1.0 - TAIL_MU_FLOOR) * mu_ratio,
                    surface_gate * (1.0 - deep_shadow_gate)
                );
            }

            var ms = vec3<f32>(0.0);
            var ms_atten = 1.0;
            for (var octave: i32 = 0; octave < MS_OCTAVES; octave = octave + 1) {
                let t_sun_ms = exp(-tau_sun * ms_atten);
                let blend = min(1.0, f32(octave) * MS_BLEND_RATE);
                let oct_phase = phase * (1.0 - blend) + ISO_PHASE * blend;
                var contrib = ms_atten * t_sun_ms * oct_phase;
                let iso_gate = smoothstep(0.35, 1.0, blend);
                // Octaves 0-1 still carry the beam and keep their exact
                // baseline value; the tail is put on the diffusion depth in
                // proportion to how isotropic it has already become, so the
                // handover is continuous and nothing is double-counted.
                contrib = contrib
                    * mix(1.0, ms_tail_factor * tail_mu_factor, iso_gate);
                if (deep_shadow_ms_suppression > 0.0) {
                    let ms_floor = max(
                        DEEP_SHADOW_MS_FLOOR,
                        1.0 - deep_shadow_ms_suppression
                              * storm_weight
                              * deep_shadow_gate
                              * iso_gate
                    );
                    contrib = contrib * ms_floor;
                }
                ms = ms + contrib * u.cloud_sun.xyz;
                ms_atten = ms_atten * MS_ATTEN;
            }

            // The diffraction spike joins the ladder as part of the
            // once-scattered term: octave 0's transmittance, no tail or
            // suppression factors (octave 0's iso_gate is 0), and it shares
            // the direct-boost and gradient multipliers below exactly as
            // octave 0 does. The guard is per-pixel-coherent (the window is
            // constant along the ray), so far-from-sun pixels skip the exp.
            if (phase_diffraction > 0.0) {
                ms = ms + phase_diffraction * exp(-tau_sun) * u.cloud_sun.xyz;
            }

            // Light-transfer split, warm side: modest boost of the unoccluded
            // direct/MS source at low sun (witness iter_006).
            if (light_transfer_split_strength > 0.0) {
                let direct_factor = 1.0 + light_transfer_split_strength
                    * LIGHT_TRANSFER_DIRECT_BOOST
                    * exp(-tau_sun);
                ms = ms * direct_factor;
            }

            // fill_shape carries the surface's billow orientation into the
            // diffuse fills below (2026-08-11): once tau_sun saturates, ms
            // is spent and every remaining light source was orientation-
            // blind, so shaded cloud went flat. Real anvil undersides
            // (IMG_7053) keep soft billow definition deep into shadow. The
            // fills take the same measured gradient at reduced weight —
            // diffuse light is directional enough to shade billows, just
            // more softly than the beam.
            var fill_shape = 1.0;
            if (gradient_shading_strength > 0.0) {
                if (grad_len > 1e-12) {
                    var n_dot_sun = -dot(grad, sun) / grad_len;
                    if (n_dot_sun < 0.0) {
                        n_dot_sun = n_dot_sun * GRADIENT_SHADING_SHADOW_SIDE_SCALE;
                    }
                    let gradient_factor = max(
                        0.20,
                        1.0 + gradient_shading_strength
                              * surface_gate
                              * n_dot_sun
                    );
                    ms = ms * gradient_factor;
                    fill_shape = mix(1.0, gradient_factor,
                                     FILL_GRADIENT_WEIGHT);
                }
            }

            // Powder is a function of cumulative optical depth since the current
            // cloud entry, not the current step size (witness.py:729-732).
            // powder_weight is the scattering-geometry dependence powder has
            // always been missing; at weight 1 this is exactly the old
            // expression. Note no extra depth gate is needed: the deficit it
            // scales, exp(-POWDER_COEFF * tau_depth), is already dead by
            // tau_depth ~ 2, so the change is confined to the thin skin and
            // the edges, which is where the physics lives.
            let powder = 1.0 - powder_weight * exp(-POWDER_COEFF * tau_depth);
            let scatter_weight = d_tau * powder * transmittance * air_t;
            col = col + ice_recolor(scatter_weight * ms, phase_tint, ice_on);

            // (t_sky and shallow_open were measured above the MS loop.)

            // Second probe direction, free from the pre-march: how much cloud
            // lies beyond this sample along the sightline. A skin sample on a
            // wisp is open both ways and keeps its full diffuse fill; the
            // same skin on a deep mass is open only backward, toward the
            // camera, and is a poor place for diffuse light to reach. This is
            // the whole of the entry-face thick/thin distinction, and because
            // the direction is the sightline the structure it reports is
            // registered with the silhouette: a luminous fringe around every
            // edge and gap, with the body behind it falling away.
            //
            // It multiplies the diffuse terms rather than feeding t_sky,
            // deliberately. Both consumers of t_sky wrap it in a floor
            // (ambient_occlusion_floor, SKY_PROBE_FILL_FLOOR) that exists so
            // one vertical measurement cannot drive a sample to black; the
            // forward measurement is independent evidence about a different
            // part of the hemisphere and should not be spent inside another
            // term's safety floor. It carries its own.
            //
            // Its weight is the sine of the angle between the two probe
            // directions (ahead_novelty, hoisted out of the loop). A sightline
            // pointing straight up is re-measuring what iter_001 already
            // measured and must not be counted twice; an oblique or horizontal
            // one is genuinely new information about the hemisphere. That is
            // the whole of the two-direction quadrature: coincident samples
            // collapse to one, orthogonal samples both count.
            var ahead_factor = 1.0;
            if (deep_shadow_gate > 0.0) {
                let tau_ahead = max(tau_total - tau_view, 0.0);
                let t_ahead = AHEAD_FLOOR + (1.0 - AHEAD_FLOOR)
                    * diffuse_transmittance(tau_ahead, g_hg);
                ahead_factor = mix(
                    1.0, t_ahead,
                    AHEAD_WEIGHT * ahead_novelty * deep_shadow_gate
                );
            }

            // Ambient: height-based on the outer box (witness._render_image).
            // The ramp floor is by openness: a storm interior keeps the tuned
            // 0.3, an open shallow base — which sees sky sideways and bright
            // surface below — keeps 0.6 of the full fill.
            let h = clamp((p.z - u.bmin.z) / (u.bmax.z - u.bmin.z), 0.0, 1.0);
            let height_floor = mix(
                AMBIENT_HEIGHT_FLOOR, AMBIENT_HEIGHT_FLOOR_SHALLOW,
                shallow_open * deep_shadow_gate
            );
            let height_ramp = height_floor + (1.0 - height_floor) * h;
            let amb = ambient_strength * height_ramp * ahead_factor
                      * fill_shape;
            // The constant deep-shadow floor becomes the T_sky -> 0 limit
            // of a measured factor: fully buried samples land on exactly
            // ambient_occlusion_floor as before, and everything less
            // buried lifts continuously toward unoccluded.
            //
            //   amb_factor = (1 - s) + s*floor + s*(1 - floor)*t_sky
            //
            // written out rather than as a mix() because its three pieces do
            // not share a spectrum. The first two are the fill a buried
            // sample keeps; the third is the part measured to arrive from the
            // sky. Summed, the weights are exactly the old amb_factor, so
            // this is a spectral split at unchanged luminance.
            let ao_s = select(
                0.0,
                clamp(ambient_occlusion_strength, 0.0, 1.0)
                    * storm_weight * deep_shadow_gate,
                ambient_occlusion_strength > 0.0
            );
            // Open shallow shadow is filled by skylight and keeps the
            // ambient's blue; only genuinely buried fill decays to the
            // neutral deep tint (same luminance, storm-tuned chroma).
            let fill_tint = mix(deep_tint, u.ambient_tint.xyz, shallow_open);
            let amb_w_deep = ao_s * ambient_occlusion_floor;
            let amb_w_sky = (1.0 - ao_s)
                + ao_s * (1.0 - ambient_occlusion_floor) * t_sky;
            col = col + ice_recolor(
                transmittance * d_tau * amb * air_t
                * (amb_w_sky * u.ambient_tint.xyz
                   + amb_w_deep * fill_tint), phase_tint, ice_on);

            // Light-transfer split, cool side: a skylight floor restored only
            // in saturated sun shadow; lit faces keep their contrast. The
            // tuned low-sun path, unchanged.
            if (light_transfer_split_strength > 0.0
                && deep_shadow_gate > 0.0) {
                // Same measured visibility: this fill is skylight too, and it
                // is the larger of the two diffuse terms, so leaving it flat
                // would wash the structure back out.
                // Same spectral split as the ambient above: the floor part is
                // sunlight that diffused in, the t_sky part is sky.
                let fill_w_deep = deep_shadow_gate * SKY_PROBE_FILL_FLOOR;
                let fill_w_sky = (1.0 - deep_shadow_gate)
                    + deep_shadow_gate * (1.0 - SKY_PROBE_FILL_FLOOR) * t_sky;
                let sky_fill = light_transfer_split_strength
                    * LIGHT_TRANSFER_SHADOW_SKYLIGHT
                    * height_ramp
                    * deep_shadow_gate * ahead_factor * fill_shape;
                col = col + ice_recolor(
                    transmittance * d_tau * sky_fill * air_t
                    * (fill_w_sky * u.ambient_tint.xyz
                       + fill_w_deep * fill_tint), phase_tint, ice_on);
            }

            // High-sun shadow skylight (2026-08-11): where the split above
            // is gated off, moderately shadowed samples used to get no
            // skylight at all — the beam and the MS octaves are spent by
            // tau_sun ~ 20, so a fair-weather base went from white to dark
            // grey across a few steps. Openness decides the coefficient
            // (a buried storm interior keeps almost nothing, which is what
            // keeps storm bases dark), and the measured sky visibility
            // carves the spatial structure so bases brighten toward their
            // edges and thin spots.
            let high_sun_fill = (1.0 - light_transfer_split_strength)
                * mix(HIGH_SUN_SHADOW_SKYLIGHT_STORM,
                      HIGH_SUN_SHADOW_SKYLIGHT_SHALLOW,
                      shallow_open)
                * shadow_gate * height_ramp * ahead_factor * fill_shape;
            if (high_sun_fill > 0.0) {
                let vis = SKY_PROBE_FILL_FLOOR
                    + (1.0 - SKY_PROBE_FILL_FLOOR) * t_sky;
                col = col + ice_recolor(
                    transmittance * d_tau * high_sun_fill * air_t
                    * vis * u.ambient_tint.xyz, phase_tint, ice_on);
            }

            // Diffused-beam glow (see the constants block): isotropic
            // re-emission of the diffusion tail of the direct beam. The
            // dominant base-lighting term for moderately thick fair-weather
            // cloud; fades to a quarter for buried storm interiors.
            let diffuse_beam = DIFFUSE_BEAM_STRENGTH
                * diffuse_transmittance(tau_sun, g_hg)
                * (1.0 - exp(-tau_sun / DIFFUSE_BEAM_TAU_ONSET))
                * mix(1.0, DIFFUSE_BEAM_STORM_KEEP,
                      deep_shadow_gate * (1.0 - shallow_open))
                * fill_shape;
            if (diffuse_beam > 0.0) {
                col = col + ice_recolor(
                    transmittance * d_tau * diffuse_beam * air_t
                    * ISO_PHASE * DIFFUSE_BEAM_TINT * u.cloud_sun.xyz,
                    phase_tint, ice_on);
            }

            // City uplight: the second light source of the night. The glow
            // mip under the sample at a footprint of its own height (the
            // solid-angle average of the city it actually sees), times the
            // two-stream transmittance of the cloud below, delivered like
            // the ambient fill. Replaces the daytime surface bounce, which
            // is gated off under CITY just below.
            if (CITY) {
                // Never below mip 1: mip 0's r is the density the heights
                // come from, the glow starts at mip 1 (export_web_assets).
                let up_lod = clamp(
                    // @domain: a positive max over u.ocean_params.x, the surface tile's cell size (> 0)
                    log2(max(p.z, u.ocean_params.x) / u.ocean_params.x),
                    1.0, u.ocean_params.w);
                let g_raw = max(
                    // p is a WORLD cloud sample: convert for the glow
                    // (city frame), keep world for the tau probe below.
                    city_glow_sample(city_fold_xy(p.xy), up_lod)
                        - CITY_UPLIGHT_GLOW_BIAS,
                    0.0);
                let g_up = g_raw / (CITY_UPLIGHT_GLOW_HALF + g_raw);
                let tau_dn = city_uplight_probe_tau(p, probe_jitter);
                let t_dn = exp(-tau_dn * CITY_UPLIGHT_TAU_SCALE);
                let up = CITY_UPLIGHT_STRENGTH * g_up
                    * (CITY_UPLIGHT_FLOOR + (1.0 - CITY_UPLIGHT_FLOOR) * t_dn);
                col = col + transmittance * d_tau * up * air_t
                            * fill_shape * CITY_UPLIGHT_COLOR;
            }

            // Surface bounce is anchored at physical z=0, not the AABB floor.
            if (BOUNCE_STRENGTH > 0.0 && !CITY) {
                let bounce_frac = clamp(1.0 - p.z / u.bmax.z, 0.0, 1.0);
                var bounce = BOUNCE_STRENGTH * bounce_frac;
                if (bounce_depth_attenuation > 0.0) {
                    bounce = bounce * exp(-bounce_depth_attenuation * tau_depth);
                }
                // The surface underneath is in this cloud's own shadow. The
                // bounce pedestal only exists in proportion to the sunlight
                // that actually reaches the ground, and under a saturated
                // column that is nearly none — which is why a real storm
                // base is dark and this one was not. tau_depth alone could
                // never see it: every base sample has just entered the
                // cloud, so the existing depth attenuation is the same
                // number everywhere across a base and the term composited
                // as a flat pedestal over the whole frame. Faded in by the
                // saturation gate, so thin cloud keeps its bounce exactly.
                //
                // A lateral floor (iter_001 flagged this as the one thing it
                // would tune): the ground a base sample sees is a hemisphere,
                // not a point, and a cloud shadow is finite. Beyond its edge
                // is sunlit surface, and that light reaches the base without
                // ever passing through the saturated column overhead — so the
                // sunward attenuation must not run to zero. T(80) = 0.10 with
                // no floor; the floor puts it back to a small but nonzero
                // pedestal, and because BOUNCE_TINT is warm it restores
                // warmth exactly at the cloud base, where the accumulated
                // blue collected worst.
                bounce = bounce * mix(
                    1.0,
                    max(diffuse_transmittance(tau_sun, g_hg),
                        BOUNCE_LATERAL_FLOOR),
                    deep_shadow_gate
                );
                col = col + ice_recolor(
                    transmittance * d_tau * bounce * air_t * BOUNCE_TINT,
                    phase_tint, ice_on);
            }

            // This step's attenuation, computed once. The aerial in-scatter
            // below and the transmittance update are the same exp of the same
            // operand; the compiler cannot always see that across the branch,
            // and it is one transcendental per view step either way.
            let step_atten = exp(-d_tau);

            // Aerial in-scatter: sky light scattered into the path replaces
            // exactly the radiance this sample occludes.
            if (aerial_strength > 0.0 && air_t < 1.0) {
                let aer_in = transmittance
                    * (1.0 - step_atten)
                    * (1.0 - air_t);
                col = col + aer_in * aer;
            }

            transmittance = transmittance * step_atten;
            t = t + dt;
        }
    }

    // Ocean for rays that exit or MISS the outer box without becoming opaque.
    // The condition is what it always meant to say — the water was hit, the
    // ray still carries light, and the march did not already shade it — with
    // no reference to a t_far that a missing ray never had (see
    // ocean_consumed above).
    //
    // The 50-outer-width clamp below is a second, much larger-scale version
    // of the same shape and is deliberately left alone: past it the water is
    // unshaded and the sky shows through, but at that range the ocean's own
    // aerial haze (ocean_realism_b.y * t_hit) has already carried it onto the
    // horizon sky colour, so the seam is invisible at the shipped settings
    // and only appears with the legacy no-haze ocean. Removing it is a look
    // change — water drawn to the edge of the float — not a bug fix.
    var shade_ocean = ocean_in_march;
    if (ocean_on
        && !ocean_consumed
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_ocean < 1e29) {
        let ocean_hit = u.cam_origin.xyz + t_ocean * dir;
        let outer_size = u.bmax.xyz - u.bmin.xyz;
        let center = 0.5 * (u.bmin.xy + u.bmax.xy);
        shade_ocean = abs(ocean_hit.x - center.x) < outer_size.x * 50.0
            && abs(ocean_hit.y - center.y) < outer_size.y * 50.0;
    }
    if (shade_ocean) {
        let ocean_hit = u.cam_origin.xyz + t_ocean * dir;
        col = col + ice_recolor(transmittance
                    * ocean_shade_dispatch(
                        ocean_hit, dir, sun,
                        sun_cone_dir(sun, sun_frame, sun_cone_seed,
                                     penumbra_tan),
                        t_ocean, shadow_jitter),
                    ICE_OCEAN_TINT, ice_on);
        transmittance = 0.0;
    }

    // A veil the march never reached (the ray left the volume first, or
    // never entered it): in front of the surface and the sky, behind every
    // cloud the march did see.
    if (veil_pending) {
        t_front = transmittance;
        col = col + transmittance * veil.rgb;
        transmittance = transmittance * veil.trans;
    }

    // City for rays that exit or miss the volume without becoming opaque —
    // the night twin of the far-water fallback above, minus the range clamp
    // (the city fog has already carried a distant hit onto the haze).
    if (CITY
        && !ocean_consumed
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_city < 1e29) {
        col = col + transmittance * city_rgb;
        transmittance = 0.0;
    }

    if (transmittance > TRANSMITTANCE_CUTOFF) {
        var sky: vec3<f32>;
        if (CITY) {
            sky = night_sky_radiance(dir, sun);
        } else if (ice_on) {
            sky = alien_sky(dir, sun, u.sky_disc.xyz);
        } else {
            sky = sky_radiance(
                dir, sun,
                u.sky_horizon.xyz, u.sky_bloom.xyz, u.sky_disc.xyz,
                u.cloud_sun.w
            );
        }
        col = col + transmittance * sky;
    }
    if (!TONE_MAP) {
        return vec4<f32>(col, 1.0);
    }
    return vec4<f32>(tone_map(col, exposure, u.periodic.w),
                     select(1.0, t_front, steam_live));
}
