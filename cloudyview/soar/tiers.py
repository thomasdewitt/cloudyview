"""What each soar quality tier sets, for a render made from Python.

The browser's table is QUALITY_PRESETS in web/soar/constants.js; tests pin
the two together.
"""

# The capture recipe of each tier: what `witness --quality` and a `soar` video
# render with. A capture marches with the sampling the browser reaches when
# the camera is parked, which is the High tier's step factors on every tier.
# So the tiers differ in lighting method, in accumulated passes and in the
# city's roof clutter, not in stepping.
#   step_factor, light_step_factor   march steps, in units of the smallest voxel
#   light_cache   read the sun's optical depth from the baked light cache
#                 instead of marching to the sun at every cloud sample
#   ahead_march   run the forward pre-march (the optical depth still ahead
#                 on each sightline, which opens the deep-shadow fill toward
#                 a cloud's far edge). Low and Minimal leave it out: it was
#                 27-31% of their march for under one display level
#   fine_gradient  blend the one-voxel stencil into the shading normal.
#                 Minimal leaves it out: 15.7% of its march for a change at
#                 its own sampling grain (0.54 of 255 RMS against 0.63)
#   accumulate    passes averaged into the image (the browser's parkedFrames)
#   city_roof_density   the night city's roof clutter: a factor on the
#                 probability of each clutter slot on a roof (tanks,
#                 condensers, vents, pipe runs, packaged units; never the
#                 stair or the skylights). 1 is the density the roofs were
#                 tuned at; the higher tiers can afford more.
QUALITY_PRESETS = {
    "max":     {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": False, "ahead_march": True, "fine_gradient": True,  "accumulate": 32,
                "city_roof_density": 1.4},
    "high":    {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True, "ahead_march": True, "fine_gradient": True,   "accumulate": 32,
                "city_roof_density": 1.4},
    "medium":  {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True, "ahead_march": True, "fine_gradient": True,   "accumulate": 24,
                "city_roof_density": 1.2},
    "low":     {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True, "ahead_march": False, "fine_gradient": True,  "accumulate": 16,
                "city_roof_density": 0.7},
    "minimal": {"step_factor": 2.0, "light_step_factor": 2.0,
                "light_cache": True, "ahead_march": False, "fine_gradient": False,  "accumulate": 8,
                "city_roof_density": 0.5},
}

# The light cache's resolution: one cache texel per this many voxels along
# each axis.
LIGHT_CACHE_DIVISOR = 2

# The densities the shader's far-field mean roof was measured over
# (cc_rooftopworks_mech_rho in shader/city/components/rooftopworks.wgsl).
# A density outside them would draw near roofs the far mean does not stand
# for, so pack_uniforms refuses it.
CITY_ROOF_DENSITY_RANGE = (0.5, 1.4)
