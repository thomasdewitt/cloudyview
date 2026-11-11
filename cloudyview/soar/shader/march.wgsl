// The cloud march: fs_main, one view ray per fragment.
//
// The ray is intersected with the volume (in a periodic domain, with its z
// slab and a range cap) and marched front to back. Each step that holds
// cloud adds its source terms, weighted by the step's optical depth d_tau
// and by the transmittance so far, then multiplies the transmittance by
// exp(-d_tau). What lies behind the cloud is composited where the ray
// reaches it: the sea or the city, the city's veil, the sky. The constants
// and derivations of the lighting terms are in clouds.wgsl.
//
// Carried along the ray:
//   transmittance  exp(-optical depth crossed so far)
//   tau_depth      optical depth since the ray last entered cloud
//   tau_view       optical depth crossed in total
// Measured at each cloud sample:
//   tau_sun        optical depth toward the sun
//   t_sky          two-stream sky visibility overhead (the sky probe)
//   tau_ahead      optical depth still ahead on the sightline (the pre-march)

// City uplight: the city as the night clouds' second light source. At a
// cloud sample it is the city's glow averaged over a footprint as wide as
// the sample is high (the glow texture's mip level log2(z / cell)), times
// the transmittance of the cloud below the sample.
const CITY_UPLIGHT_STRENGTH: f32 = 0.5;
// Only glow above the citywide background lights a cloud base, so a base
// over ordinary sprawl stays moonlit and one over a bright district turns
// amber. The night clouds are dark enough that even a small warm pedestal
// would show on all of them.
const CITY_UPLIGHT_GLOW_BIAS: f32 = 0.15;
const CITY_UPLIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.38, 0.10);
// The glow saturates as g / (HALF + g). The brightest district's glow is
// several times an ordinary bright district's, and a linear scale strong
// enough to warm an ordinary cloud base would bleach the whole sky over the
// brightest.
const CITY_UPLIGHT_GLOW_HALF: f32 = 0.6;
// The probe straight down from the sample: its span (m) and step count.
const CITY_UP_PROBE_SPAN: f32 = 1400.0;
const CITY_UP_PROBE_STEPS: i32 = 12;
// The uplight a sample keeps under any depth of cloud.
const CITY_UPLIGHT_FLOOR: f32 = 0.02;
// The uplight falls as exp(-TAU_SCALE * tau_down): one e-folding per
// 1 / 0.25 = 4 optical depths of cloud below.
const CITY_UPLIGHT_TAU_SCALE: f32 = 0.25;

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let g_hg = CLOUD_G;
    let exposure = u.exposure;
    let jitter_on = u.march_jitter;
    let jitter_scale = clamp(u.jitter_scale, 0.0, 1.0);
    let sun = u.sun_dir;

    // The camera ray, shared with the surface passes (camera_ray), so the
    // surface they traced is the surface this march composites.
    let cr = camera_ray(frag_pos);
    let frame_index = cr.frame_index;
    let strat_pixel = cr.strat_pixel;
    let dir = cr.dir;

    // safe_inv_dir, not 1.0 / dir: an axis-aligned ray would divide by zero
    // (see safe_inv_dir).
    let inv_dir = safe_inv_dir(dir);
    let periodic_on = PERIODIC_DOMAIN;
    var t_near: f32;
    var t_far: f32;
    if (periodic_on) {
        // Tiled domain: only the z slab bounds the march. In place of a
        // horizontal exit there is the range cap (periodic_march_cap).
        let tz0 = (u.bmin.z - u.cam_origin.z) * inv_dir.z;
        let tz1 = (u.bmax.z - u.cam_origin.z) * inv_dir.z;
        t_near = max(min(tz0, tz1), 0.0);
        t_far = min(
            max(tz0, tz1),
            periodic_march_cap(u.cam_origin.z, dir)
        );
    } else {
        let hit = ray_box(u.cam_origin, inv_dir);
        t_near = max(hit.x, 0.0);
        t_far = hit.y;
    }

    // Under CITY there is no ocean plane. The surface below is the traced
    // city, an opaque floor at t_city that goes through the same guards.
    let ocean_on = !CITY && u.surface_enabled > 0.5;
    var t_ocean = 1e30;
    if (ocean_on && dir.z < -1e-8) {
        let t_ocean_candidate = (u.surface_z - u.cam_origin.z) / dir.z;
        if (t_ocean_candidate > 0.0) {
            t_ocean = t_ocean_candidate;
        }
    }

    // The opaque surface below (the traced city) was found for this very ray
    // by the surface passes: its radiance, fog included, and its distance,
    // one texel per pixel. It is composited where the march reaches it, as
    // the ocean plane is.
    var t_city = 1e30;
    var city_rgb = vec3<f32>(0.0);
    if (CITY) {
        let sp = textureLoad(surface_buf, vec2<i32>(frag_pos.xy), 0);
        t_city = sp.w;
        city_rgb = sp.xyz;
    }
    let t_surface = select(t_ocean, t_city, CITY);
    // The veil (the city's steam) is composited once, where the march passes
    // its depth, so that a cloud in front of it stays in front and a cloud
    // behind it is seen through it. Only a veil in front of the surface
    // counts.
    var veil = Veil(vec3<f32>(0.0), 1.0, 1e30);
    if (CITY) {
        veil = veil_unpack(textureLoad(veil_buf, vec2<i32>(frag_pos.xy), 0));
    }
    // When the steam is drawn after the accumulation the veil holds only its
    // depth (trans = 1). The march then records the cloud's transmittance at
    // that depth, t_front, and returns it in alpha for fs_steam_live.
    // Otherwise alpha is 1.
    let steam_live = city_steam_live();
    var veil_pending = CITY && (veil.trans < 1.0 || steam_live)
        && veil.t < t_surface;
    var t_front = 1.0;

    let cos_scatter = dot(dir, sun);
    let phase = hg_phase(cos_scatter, g_hg);
    // The diffraction spike's phase function (see DIFFRACTION_G). Like
    // `phase` it is constant along the ray, and it is exactly zero beyond
    // 30 deg from the sun.
    let phase_diffraction = DIFFRACTION_WEIGHT
        * hg_phase(cos_scatter, DIFFRACTION_G)
        * smoothstep(DIFFRACTION_WINDOW_COS_START,
                     DIFFRACTION_WINDOW_COS_FULL, cos_scatter);

    // The powder term's dependence on scattering angle (see POWDER_COEFF).
    // One value per pixel: 1 keeps the whole darkening, and
    // 1 - POWDER_FWD_FADE is left at exact forward scattering.
    let g2_hg = g_hg * g_hg;
    // mu_cross, clamped short of 1 so that the smoothstep below is never
    // degenerate. The select handles isotropic scattering (g = 0), which
    // has no forward cone: powder stays whole in every direction.
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

    // Aerial perspective. `aer` is the colour distant cloud fades into: the
    // sky at the horizon in this sightline's azimuth, solar disc excluded.
    // The sea's haze fades into the same colour, so cloud and water converge
    // on one.
    //
    // ice_on is the false-colour ice view (ice.wgsl).
    let ice_on = u.ice_mode > 0.5;
    // The ice view runs at 0.3 of the haze strength: at full strength every
    // distant cloud converges on the one horizon colour and the phase
    // colouring is lost in it.
    let aerial_strength = select(1.0, 0.3, ice_on);
    var aer = vec3<f32>(0.0);
    if (CITY) {
        // At night the target is the host's night haze colour itself.
        aer = u.sky_horizon;
    } else {
        let ah_len = length(dir.xy);
        var ah = dir.xy;
        if (ah_len > 1e-8) {
            ah = dir.xy / ah_len;
        }
        if (ice_on) {
            // The green sky's horizon.
            aer = alien_sky(vec3<f32>(ah, 0.0), sun, vec3<f32>(0.0));
        } else {
            aer = sky_radiance(
                vec3<f32>(ah, 0.0), sun,
                u.sky_horizon, u.sky_bloom, vec3<f32>(0.0)
            );
        }
    }

    // Jittered first step: the march starts a random fraction of a step past
    // t_near, which decorrelates the sampling shells of neighbouring pixels
    // (coherent, they show as rings). The draw is stratified over the
    // accumulation, so the shells a pixel tests over its frames tile the
    // step. The step is the outer level's: the nest lies strictly inside the
    // outer box, so a ray always enters through the outer level.
    let jitter = strat1(frame_index, strat_pixel, STRAT_STREAM_ENTRY,
                        STRAT_ALPHA_3);
    // The quadrature offset of the city's downward probe
    // (city_uplight_probe_tau), in [-0.5, 0.5).
    let probe_jitter = strat1(frame_index, strat_pixel,
                              STRAT_STREAM_CITY_PROBE,
                              STRAT_ALPHA_10) - 0.5;
    // The sun march's quadrature offset. With jitter off it is zero and the
    // sun march is deterministic.
    let shadow_jitter = jitter_on * jitter_scale
                        * strat1(frame_index, strat_pixel,
                                 STRAT_STREAM_SUN_MARCH,
                                 STRAT_ALPHA_4);
    // The draw on the solar disc: two uniforms, squared radius and azimuth.
    let sun_cone_seed = strat2(frame_index, strat_pixel,
                               STRAT_STREAM_SUN_CONE,
                               vec2<f32>(STRAT_ALPHA_5, STRAT_ALPHA_6));
    let penumbra_tan = SUN_ANGULAR_RADIUS * SUN_CONE_WIDEN
                       * jitter_on * jitter_scale;
    // The disc's tangent frame, built once per fragment.
    let sun_frame = sun_tangent_frame(sun);
    // The forward pre-march's quadrature offset.
    let ahead_jitter = jitter_on * jitter_scale
                       * strat1(frame_index, strat_pixel,
                                STRAT_STREAM_PREMARCH,
                                STRAT_ALPHA_9);
    let entry_dt = u.dt_view;
    var t = t_near + jitter_on * jitter * jitter_scale * entry_dt;

    var transmittance = 1.0;
    var col = vec3<f32>(0.0);
    var tau_depth = 0.0;
    // Whether the surface (the water, or the city) has been composited. It
    // is tracked and not inferred afterwards from t_ocean > t_far: on a miss
    // ray_box returns t_near > t_far with t_far still a positive finite
    // number (the nearest exit plane of a box the ray never entered), so a
    // test against t_far would decide the far-water fallback wrongly for
    // rays that pass beside the box.
    var ocean_consumed = false;
    // The march reached the water. It is shaded by the one
    // ocean_shade call after the loop, which also shades water the
    // march never reached: the ocean shader holds two sun marches, and a
    // second inlined copy of it would enlarge fs_main for nothing.
    // Transmittance is not touched between the break and that call.
    var ocean_in_march = false;
    // Optical depth this ray has crossed in total, which turns the
    // pre-march's total into what is still ahead of a sample. Unlike
    // tau_depth it never resets.
    var tau_view = 0.0;
    // Whether the previous view sample was inside the nest, to detect the
    // fine -> coarse crossing.
    var was_in_nest = false;

    // The periodic march can cover many domain widths, so it has a larger
    // step cap.
    let max_view_steps = select(MAX_VIEW_STEPS, MAX_VIEW_STEPS_PERIODIC,
                                periodic_on);

    // The pre-march's weight: the sine of the angle between the sightline
    // and the vertical, i.e. how much of the hemisphere the sightline
    // covers that the vertical sky probe does not.
    let ahead_novelty = sqrt(max(1.0 - dir.z * dir.z, 0.0));

    // Colour of the fill a buried sample keeps (deep_fill_tint). It depends
    // only on the two lighting colours, so once per fragment.
    let deep_tint = deep_fill_tint(u.ambient_tint);

    // The forward pre-march: one coarse pass over the whole sightline before
    // anything is composited.
    var tau_total = 0.0;
    if (t_near >= 0.0 && t_near < t_far) {
        tau_total = premarch_tau_ahead(
            u.cam_origin, dir, t_near, min(t_far, t_surface), ahead_jitter
        );
    }

    if (t_near >= 0.0 && t_near < t_far) {
        for (var i: i32 = 0; i < max_view_steps; i = i + 1) {
            // The surface is tested before the t_far break, so that a plane
            // coincident with the box floor is still composited.
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
            let p = u.cam_origin + t * dir;
            // The active level sets the step, so the march refines on
            // entering the nest and coarsens on leaving it.
            let level = sample_level_at(p);
            // Leaving the nest, the step jumps from fine to coarse. Every
            // ray crosses the nest's face at the same plane and arrives with
            // at most a fine step of phase, so past the seam the samples of
            // neighbouring pixels and successive frames would cover only a
            // fraction of the coarse step, and the coherent remainder would
            // survive accumulation as stripes parallel to the seam. One more
            // uniform phase is drawn, spanning the full coarse step: a
            // uniform over a whole period, added modulo the period, is
            // uniform whatever the incoming distribution was. Entering the
            // nest needs nothing, since a coarse period of phase folded into
            // a fine one already covers it.
            if (NESTED && was_in_nest && !level.in_nest) {
                was_in_nest = false;
                let seam_jitter = strat1(frame_index, strat_pixel,
                                         STRAT_STREAM_NEST_SEAM,
                                         STRAT_ALPHA_10);
                let skip = jitter_on * seam_jitter * jitter_scale
                           * max(u.dt_view, t * u.view_lod_tan);
                if (skip > 0.0) {
                    t = t + skip;
                    continue;
                }
            }
            was_in_nest = level.in_nest;
            let sigma = level.sigma;

            // Distance LOD: the step is at least t * u.view_lod_tan, a fixed
            // angle, so distant cloud is marched at angular and not metric
            // resolution (0 = fixed dt). The shortening in dense cloud still
            // applies. The step then lands exactly on the far face, the
            // water, the veil and the city, whichever comes first.
            var dt = step_dt_for_sigma(
                sigma, max(level.dt_view, t * u.view_lod_tan)
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
            // Clear air: nothing to shade, and the depth since cloud entry
            // resets.
            if (d_tau < EMPTY_DTAU_CUTOFF) {
                tau_depth = 0.0;
                t = t + dt;
                continue;
            }

            tau_depth = tau_depth + d_tau;
            tau_view = tau_view + d_tau;

            // The ice view's hue for this sample; vec3(1) with the mode off.
            var phase_tint = vec3<f32>(1.0);
            if (ice_on) {
                phase_tint = ice_tint(ice_fraction_at(p));
            }

            // Aerial perspective: the clear-air transmittance from the
            // camera to the sample, in closed form for a haze extinction
            // beta0 exp(-z / H):
            //   tau_air = beta0 (H / mu) (exp(-z0 / H) - exp(-z1 / H)),
            // mu = dir.z, with the limit beta0 t exp(-z0 / H) for a
            // horizontal ray and beta0 t for the uniform haze (H <= 0).
            var air_t = 1.0;
            let aer_beta0 = u.haze_beta0;
            let aer_h = u.haze_scale_height;
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

            // A view ray takes many steps, so its shadow rays advance the
            // quadrature offset by the golden ratio and do not reuse one
            // value. The sequence is low-discrepancy along the march, which
            // averages the quadrature noise down within a single frame.
            let step_shadow_jitter = fract(
                shadow_jitter + f32(i) * 0.6180339887498949
            );
            // This sample's point on the solar disc. The R2 additive
            // recurrence (the golden ratio's 2D analogue) advances both
            // coordinates along the march for the same reason: the shadow
            // rays that composite into one pixel cover the disc within a
            // single frame, so a shadow's edge is soft from the first frame
            // and not only once the accumulation settles.
            let sun_shadow = sun_cone_dir(
                sun, sun_frame,
                fract(sun_cone_seed + f32(i) * vec2<f32>(
                    0.7548776662466927, 0.5698402909980532
                )),
                penumbra_tan
            );
            // Under LIGHT_CACHE the baked value stands in for the march: the
            // central sun, no disc draw, no distance LOD (sample_light_cache).
            // The same read returns the optical depth overhead, which the
            // sky visibility below is made from.
            var tau_sun: f32;
            var tau_overhead = 0.0;
            if (LIGHT_CACHE) {
                let cached = sample_light_cache(p);
                tau_sun = cached.x;
                tau_overhead = cached.y;
            } else {
                tau_sun = light_march_tau(p, sun_shadow, t * u.light_lod_tan,
                                          step_shadow_jitter);
            }
            let light_transfer_split_strength = u.light_split;
            // The deep-shadow gate, in log tau (see DEEP_SHADOW_TAU_START).
            // It is computed whatever the strengths of the deep-shadow terms
            // are, because the diffused beam and the high-sun skylight read
            // it too. The max() guards the log as tau_sun -> 0, far below
            // the gate's onset.
            let deep_shadow_gate = smoothstep(
                DEEP_SHADOW_LOG_TAU_START, DEEP_SHADOW_LOG_TAU_FULL,
                log(max(tau_sun, 1e-3))
            );

            // The moderate-shadow gate, which the high-sun skylight fill
            // uses (see SHADOW_SKYLIGHT_TAU_ONSET).
            let shadow_gate = 1.0 - exp(-tau_sun / SHADOW_SKYLIGHT_TAU_ONSET);

            // Sky visibility overhead, for every consumer, from the baked
            // optical depth above the sample. It is needed only where a
            // gate is open, which is where tau_sun is not zero.
            var t_sky = 1.0;
            var shallow_open = 0.0;
            if (shadow_gate > 0.0 || deep_shadow_gate > 0.0) {
                if (!LIGHT_CACHE) {
                    tau_overhead = sample_light_cache(p).y;
                }
                t_sky = diffuse_transmittance(tau_overhead, g_hg);
                shallow_open = smoothstep(
                    SHALLOW_OPEN_TSKY_START, SHALLOW_OPEN_TSKY_FULL, t_sky
                );
            }
            // A buried sample keeps the whole suppression; an open shallow
            // shadow keeps SHALLOW_SUPPRESSION_KEEP of it.
            let storm_weight = mix(
                SHALLOW_SUPPRESSION_KEEP, 1.0, 1.0 - shallow_open
            );

            // Diffusion depth of the isotropic octaves (see MS_TAIL_FLOOR).
            // Exactly 1 below the knee.
            let ms_tail_factor = MS_TAIL_FLOOR + (1.0 - MS_TAIL_FLOOR)
                * diffuse_transmittance(
                    max(tau_sun - MS_TAIL_TAU_KNEE, 0.0), g_hg
                );

            // The local surface frame: the density gradient and the gate on
            // its use (see GRADIENT_SHADING_*).
            //
            // The stencil, twelve texture reads, is skipped on the entry
            // skin, where it cannot matter. At tau_depth <=
            // GRADIENT_SHADING_TAU_START the gate's depth factor is exactly
            // 0, and with the gate at 0 every consumer of the gradient is
            // neutral: tail_mu_factor is mix(1, x, 0), gradient_factor is
            // max(0.2, 1 + 0) and fill_shape is mix(1, 1, w). Leaving grad
            // at zero takes the grad_len guards below instead, which leave
            // the same three ones in place.
            var grad = vec3<f32>(0.0);
            var grad_len = 0.0;
            var surface_gate = 0.0;
            if (tau_depth > GRADIENT_SHADING_TAU_START) {
                let grad_conf_v = sigma_gradient(p, sigma, t, level.in_nest);
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

            // Beam incidence on the skin relative to a horizontal face (see
            // TAIL_MU_FLOOR). 1 where the cloud top is level.
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

            // The octaves (see MS_OCTAVES).
            var ms = vec3<f32>(0.0);
            var ms_atten = 1.0;
            for (var octave: i32 = 0; octave < MS_OCTAVES; octave = octave + 1) {
                let t_sun_ms = exp(-tau_sun * ms_atten);
                let blend = min(1.0, f32(octave) * MS_BLEND_RATE);
                let oct_phase = phase * (1.0 - blend) + ISO_PHASE * blend;
                var contrib = ms_atten * t_sun_ms * oct_phase;
                let iso_gate = smoothstep(0.35, 1.0, blend);
                // Octaves 0 and 1 carry the beam and are left as they are.
                // The diffusion-depth and incidence factors come in with
                // iso_gate, i.e. in proportion to how isotropic the octave
                // already is, so the handover is continuous.
                contrib = contrib
                    * mix(1.0, ms_tail_factor * tail_mu_factor, iso_gate);
                let ms_floor = max(
                    DEEP_SHADOW_MS_FLOOR,
                    1.0 - DEEP_SHADOW_MS_SUPPRESSION
                          * storm_weight
                          * deep_shadow_gate
                          * iso_gate
                );
                contrib = contrib * ms_floor;
                ms = ms + contrib * u.cloud_sun;
                ms_atten = ms_atten * MS_ATTEN;
            }

            // The diffraction spike joins as part of the once-scattered
            // term: octave 0's transmittance, none of the tail or
            // suppression factors, and the direct boost and gradient factor
            // below, as octave 0 has them. The guard is constant along the
            // ray, so pixels far from the sun skip the exp.
            if (phase_diffraction > 0.0) {
                ms = ms + phase_diffraction * exp(-tau_sun) * u.cloud_sun;
            }

            // Light-transfer split, direct side: a modest boost of the
            // unoccluded beam at low sun.
            if (light_transfer_split_strength > 0.0) {
                let direct_factor = 1.0 + light_transfer_split_strength
                    * LIGHT_TRANSFER_DIRECT_BOOST
                    * exp(-tau_sun);
                ms = ms * direct_factor;
            }

            // Gradient shading of the beam terms, and fill_shape: the same
            // factor at reduced weight, for the diffuse fills below (see
            // FILL_GRADIENT_WEIGHT). Without it shadowed cloud, which only
            // the fills light, has no billow relief.
            var fill_shape = 1.0;
            if (grad_len > 1e-12) {
                var n_dot_sun = -dot(grad, sun) / grad_len;
                if (n_dot_sun < 0.0) {
                    n_dot_sun = n_dot_sun * GRADIENT_SHADING_SHADOW_SIDE_SCALE;
                }
                let gradient_factor = max(
                    0.20,
                    1.0 + GRADIENT_SHADING_STRENGTH
                          * surface_gate
                          * n_dot_sun
                );
                ms = ms * gradient_factor;
                fill_shape = mix(1.0, gradient_factor,
                                 FILL_GRADIENT_WEIGHT);
            }

            // Powder (see POWDER_COEFF): a function of the optical depth
            // since cloud entry, not of the step size. It needs no depth
            // gate of its own, since exp(-POWDER_COEFF * tau_depth) is
            // negligible by tau_depth ~ 3.
            let powder = 1.0 - powder_weight * exp(-POWDER_COEFF * tau_depth);
            let scatter_weight = d_tau * powder * transmittance * air_t;
            col = col + ice_recolor(scatter_weight * ms, phase_tint, ice_on);

            // The pre-march's factor on the diffuse fills (see
            // AHEAD_STEP_SCALE): how buried the sample is along the
            // sightline.
            //
            // It multiplies the fills and does not feed t_sky. Both
            // consumers of t_sky wrap it in a floor (AMBIENT_OCCLUSION_FLOOR,
            // SKY_PROBE_FILL_FLOOR) that exists so that one vertical
            // measurement cannot drive a sample to black. The forward
            // measurement is independent evidence about another part of the
            // hemisphere, and it carries its own floor.
            //
            // ahead_novelty weights it: a sightline straight up repeats the
            // sky probe and must not be counted twice, while a horizontal
            // one is wholly new.
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

            // Ambient fill. It ramps with height in the outer box, from a
            // floor that is AMBIENT_HEIGHT_FLOOR for a buried sample and
            // AMBIENT_HEIGHT_FLOOR_SHALLOW for an open shallow base.
            let h = clamp((p.z - u.bmin.z) / (u.bmax.z - u.bmin.z), 0.0, 1.0);
            let height_floor = mix(
                AMBIENT_HEIGHT_FLOOR, AMBIENT_HEIGHT_FLOOR_SHALLOW,
                shallow_open * deep_shadow_gate
            );
            let height_ramp = height_floor + (1.0 - height_floor) * h;
            let amb = AMBIENT_STRENGTH * height_ramp * ahead_factor
                      * fill_shape;
            // Ambient occlusion from the measured sky visibility. With
            // s = strength * storm_weight * deep_shadow_gate and
            // floor = AMBIENT_OCCLUSION_FLOOR,
            //
            //   amb_factor = (1 - s) + s*floor + s*(1 - floor)*t_sky,
            //
            // so a fully buried sample (t_sky = 0) keeps `floor` and less
            // buried ones rise continuously to 1. It is written as two
            // weights and not as a mix() because the pieces differ in
            // colour: the s*floor part is the fill a buried sample keeps
            // (fill_tint), and the rest arrives from the sky (the ambient
            // tint).
            let ao_s = storm_weight * deep_shadow_gate;
            // An open shallow shadow is filled by skylight and keeps the
            // ambient's blue. Only buried fill takes the deep tint.
            let fill_tint = mix(deep_tint, u.ambient_tint, shallow_open);
            let amb_w_deep = ao_s * AMBIENT_OCCLUSION_FLOOR;
            let amb_w_sky = (1.0 - ao_s)
                + ao_s * (1.0 - AMBIENT_OCCLUSION_FLOOR) * t_sky;
            col = col + ice_recolor(
                transmittance * d_tau * amb * air_t
                * (amb_w_sky * u.ambient_tint
                   + amb_w_deep * fill_tint), phase_tint, ice_on);

            // Light-transfer split, shadow side: a skylight fill in
            // saturated sun shadow only, so lit faces keep their contrast.
            if (light_transfer_split_strength > 0.0
                && deep_shadow_gate > 0.0) {
                // The same visibility and the same split by colour as the
                // ambient: the floor part is sunlight that diffused in, the
                // t_sky part is sky.
                let fill_w_deep = deep_shadow_gate * SKY_PROBE_FILL_FLOOR;
                let fill_w_sky = (1.0 - deep_shadow_gate)
                    + deep_shadow_gate * (1.0 - SKY_PROBE_FILL_FLOOR) * t_sky;
                let sky_fill = light_transfer_split_strength
                    * LIGHT_TRANSFER_SHADOW_SKYLIGHT
                    * height_ramp
                    * deep_shadow_gate * ahead_factor * fill_shape;
                col = col + ice_recolor(
                    transmittance * d_tau * sky_fill * air_t
                    * (fill_w_sky * u.ambient_tint
                       + fill_w_deep * fill_tint), phase_tint, ice_on);
            }

            // High-sun shadow skylight: the counterpart of the fill above
            // where the split is off (see HIGH_SUN_SHADOW_SKYLIGHT_STORM).
            // Openness sets the coefficient: a buried storm interior keeps
            // almost none, which is what keeps storm bases dark. The sky
            // visibility gives it spatial structure, so bases brighten
            // toward their edges and thin spots.
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
                    * vis * u.ambient_tint, phase_tint, ice_on);
            }

            // Diffused beam (see DIFFUSE_BEAM_STRENGTH): isotropic emission
            // of the sun's diffusion tail. It is the main light on the base
            // of a moderately thick fair-weather cloud. A buried storm
            // interior keeps DIFFUSE_BEAM_STORM_KEEP of it.
            let diffuse_beam = DIFFUSE_BEAM_STRENGTH
                * diffuse_transmittance(tau_sun, g_hg)
                * (1.0 - exp(-tau_sun / DIFFUSE_BEAM_TAU_ONSET))
                * mix(1.0, DIFFUSE_BEAM_STORM_KEEP,
                      deep_shadow_gate * (1.0 - shallow_open))
                * fill_shape;
            if (diffuse_beam > 0.0) {
                col = col + ice_recolor(
                    transmittance * d_tau * diffuse_beam * air_t
                    * ISO_PHASE * DIFFUSE_BEAM_TINT * u.cloud_sun,
                    phase_tint, ice_on);
            }

            // City uplight (see CITY_UPLIGHT_STRENGTH): the glow under the
            // sample over a footprint of the sample's own height, times the
            // transmittance of the cloud below, delivered like the ambient
            // fill. It takes the place of the surface bounce.
            if (CITY) {
                // Never below mip 1: mip 0's red channel holds the density
                // the building heights come from, and the glow starts at
                // mip 1.
                let up_lod = clamp(
                    // @domain: a positive max over u.tile_cell, the surface tile's cell size (> 0)
                    log2(max(p.z, u.tile_cell) / u.tile_cell),
                    1.0, u.tile_max_lod);
                let g_raw = max(
                    // p is a world point: folded into the city's frame for
                    // the glow, kept in world space for the tau probe.
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

            // Surface bounce (see BOUNCE_STRENGTH), measured from z = 0 and
            // not from the box floor.
            if (!CITY) {
                let bounce_frac = clamp(1.0 - p.z / u.bmax.z, 0.0, 1.0);
                var bounce = BOUNCE_STRENGTH * bounce_frac;
                bounce = bounce * exp(-BOUNCE_DEPTH_ATTENUATION * tau_depth);
                // The ground under a saturated column is in the cloud's own
                // shadow, and the bounce exists only in proportion to the
                // sunlight that reaches the ground. tau_depth cannot see
                // that: every base sample has only just entered the cloud.
                // So under the deep-shadow gate the bounce is scaled by the
                // column's two-stream transmittance, floored at
                // BOUNCE_LATERAL_FLOOR for the sunlit ground beyond the
                // shadow's edge.
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

            // This step's attenuation, computed once for the in-scatter and
            // for the transmittance update.
            let step_atten = exp(-d_tau);

            // Aerial in-scatter. The source terms above were attenuated by
            // air_t; the fraction 1 - air_t of this step's opacity is filled
            // with the haze colour.
            if (air_t < 1.0) {
                let aer_in = transmittance
                    * (1.0 - step_atten)
                    * (1.0 - air_t);
                col = col + aer_in * aer;
            }

            transmittance = transmittance * step_atten;
            t = t + dt;
        }
    }

    // The water, for rays that leave or miss the box without becoming
    // opaque: it was hit, the ray still carries light, and the march did not
    // already reach it.
    //
    // Beyond 50 box widths from the box's centre the water is not shaded and
    // the sky shows through. At that range the sea's own haze
    // (ocean_realism_b.y * t_hit) has carried it onto the horizon colour, so
    // the seam shows only with that haze off.
    var shade_ocean = ocean_in_march;
    if (ocean_on
        && !ocean_consumed
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_ocean < 1e29) {
        let ocean_hit = u.cam_origin + t_ocean * dir;
        let outer_size = u.bmax - u.bmin;
        let center = 0.5 * (u.bmin.xy + u.bmax.xy);
        shade_ocean = abs(ocean_hit.x - center.x) < outer_size.x * 50.0
            && abs(ocean_hit.y - center.y) < outer_size.y * 50.0;
    }
    if (shade_ocean) {
        let ocean_hit = u.cam_origin + t_ocean * dir;
        col = col + ice_recolor(transmittance
                    * ocean_shade(
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

    // The city, for rays that leave or miss the volume without becoming
    // opaque: the counterpart of the water above, without the range clamp
    // (the city's fog has already carried a distant hit onto the haze).
    if (CITY
        && !ocean_consumed
        && transmittance > TRANSMITTANCE_CUTOFF
        && t_city < 1e29) {
        col = col + transmittance * city_rgb;
        transmittance = 0.0;
    }

    // Whatever transmittance is left sees the sky.
    if (transmittance > TRANSMITTANCE_CUTOFF) {
        var sky: vec3<f32>;
        if (CITY) {
            sky = night_sky_radiance(dir, sun);
        } else if (ice_on) {
            sky = alien_sky(dir, sun, u.sky_disc);
        } else {
            sky = sky_radiance(
                dir, sun,
                u.sky_horizon, u.sky_bloom, u.sky_disc
            );
        }
        col = col + transmittance * sky;
    }
    // Alpha is 1, except under the live steam, where it carries t_front.
    if (!TONE_MAP) {
        return vec4<f32>(col, 1.0);
    }
    return vec4<f32>(tone_map(col, exposure, u.gamma),
                     select(1.0, t_front, steam_live));
}
