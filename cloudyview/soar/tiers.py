"""What each soar quality tier sets, for a render made from Python.

The browser's copies are in web/soar/constants.js (QUALITY_PRESETS,
PARKED_ACCUM_FRAMES_BY_TIER, CITY_ROOF_DENSITY_BY_TIER); tests pin the two
together.
"""

# The capture recipe of each tier: what `witness --quality` and a `soar` video
# render with. A capture marches with the sampling the browser reaches when
# the camera is parked, which is the High tier's step factors on every tier.
# So the tiers differ in lighting method and in accumulated passes, not in
# stepping.
#   step_factor, light_step_factor   march steps, in units of the smallest voxel
#   light_cache   read the sun's optical depth from the baked light cache
#                 instead of marching to the sun at every cloud sample
#   accumulate    passes averaged into the image
QUALITY_PRESETS = {
    "max":     {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": False, "accumulate": 32},
    "high":    {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True,  "accumulate": 32},
    "medium":  {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True,  "accumulate": 24},
    "low":     {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True,  "accumulate": 16},
    "minimal": {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True,  "accumulate": 8},
}

# The light cache's resolution: one cache texel per this many voxels along
# each axis.
LIGHT_CACHE_DIVISOR = 2

# The night city's roof clutter by tier: a factor on the probability of each
# clutter slot on a roof (tanks, condensers, vents, pipe runs, packaged units;
# never the stair or the skylights). 1 is the density the roofs were tuned
# at; the higher tiers can afford more and the lower ones draw less.
CITY_ROOF_DENSITY_BY_TIER = {
    "max": 1.4, "high": 1.4, "medium": 1.2, "low": 0.7, "minimal": 0.5,
}
# The densities the shader's far-field mean roof was measured over
# (cc_rooftopworks_mech_rho in shader/city/components/rooftopworks.wgsl).
# A density outside them would draw near roofs the far mean does not stand
# for, so pack_uniforms refuses it.
CITY_ROOF_DENSITY_RANGE = (0.5, 1.4)
