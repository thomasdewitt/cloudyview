"""The bird: a small animated flying subject for the soar fly-through.

A stylized swift — nineteen triangles of procedural numpy mesh, no assets —
that leads the flight a few metres ahead of and below the camera. It flaps
(faster when slow, tucking into a glide above a speed threshold), banks into
turns, pitches with climbs and descents, and bobs gently with its own
wingbeat. All state is exponentially smoothed so it feels alive rather than
bolted to the camera, and its fragment shader attenuates by the extinction
field so it fades naturally when it flies into cloud.

Entirely additive: `InteractiveRenderer.render(..., bird=True)` opts in
offscreen (default off, so existing renders/tests/benchmarks are untouched);
the windowed app has it on by default with `B` to toggle.
"""

from math import atan2, cos, degrees, exp, pi, radians, sin
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import wgpu

from ..angles import direction_from_azimuth_elevation
from ..camera import Camera

SHADER_PATH = Path(__file__).parent / "bird.wgsl"

_UNIFORM_NBYTES = 3 * 64 + 4 * 16  # 3 mat4 + 4 vec4

# --- Placement (metres, camera-relative) -----------------------------------
DISTANCE = 8.5         # ahead of the camera along the smoothed view direction
DROP = 2.4             # below the view center (screen-space, via camera up)
SCALE = 1.25           # mesh scale: ~1.8 m span (the third-person cheat —
                       # a real swift would be an unreadable speck at 100 fov)
NEAR, FAR = 0.5, 400.0

# --- Animation --------------------------------------------------------------
FLAP_AMPLITUDE = 0.55  # rad of wingtip rotation about the body axis
REST_DIHEDRAL = 0.10   # rad, wings-slightly-raised carry angle while flapping
GLIDE_DIHEDRAL = 0.18  # rad, the stiff shallow V of a glide
FLAP_HZ_SLOW = 4.2     # wingbeat when hovering / slow
FLAP_HZ_FAST = 2.2     # wingbeat just below the glide threshold
GLIDE_LO = 70.0        # m/s: flap starts fading into a glide
GLIDE_HI = 110.0       # m/s: fully tucked into the glide
BOB_AMPLITUDE = 0.10   # m, body bob coupled to the wingbeat
IDLE_BOB = 0.05        # m, slow ambient bob so a parked bird still breathes
IDLE_BOB_HZ = 0.45

# --- Feel (exponential smoothing time constants, seconds) -------------------
TAU_HEADING = 0.22     # view/heading lag: the bird swings through turns
TAU_BANK = 0.30
TAU_PITCH = 0.35
TAU_SPEED = 0.30
TAU_FLAP_AMP = 0.45    # blend between flapping and glide

BANK_PER_DEG_S = 0.40  # deg of roll per deg/s of heading rate
BANK_MAX = 50.0        # deg
PITCH_MAX = 25.0       # deg
BODY_PITCH = 8.0       # deg, resting nose-up attitude (slow-flight posture;
                       # also presents more wing area to the from-below view)


def _smoothstep(x: float, lo: float, hi: float) -> float:
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _build_mesh() -> Tuple[np.ndarray, int]:
    """Procedural swift: body octahedron, forked tail, swept sickle wings.

    Returns (vertex_data, n_vertices) where vertex_data is float32 rows of
    [pos.xyz, normal.xyz, span_frac] with vertices duplicated per face
    (flat-face normals, no index buffer). Local frame: +x right, +y forward,
    +z up; the flap pivot (shoulder line) is at z = 0. Units are metres.
    """
    # Body (slender fusiform, slightly deeper than wide).
    N = (0.00, 0.46, 0.000)   # nose
    U = (0.00, 0.12, 0.055)   # crown
    D = (0.00, 0.10, -0.095)  # belly keel
    L = (-0.075, 0.06, 0.005)  # left shoulder
    R = (0.075, 0.06, 0.005)   # right shoulder
    T = (0.00, -0.32, 0.005)  # tail root

    faces = [
        (N, L, U), (N, U, R), (U, L, T), (U, T, R),   # back
        (N, D, L), (N, R, D), (D, T, L), (D, R, T),   # belly
    ]

    # Forked tail: angled slightly downward so it joins the belly
    # silhouette from the usual seen-from-below viewpoint.
    TL = (-0.16, -0.64, -0.030)
    TR = (0.16, -0.64, -0.030)
    TN = (0.00, -0.46, -0.010)
    faces += [(T, TL, TN), (T, TN, TR)]

    # Right wing: swept-back sickle in two panels (arm + hand), tapering to
    # a point. Leading edge shoulder -> wrist -> tip; the wrist is raised
    # and the tip dropped (gull-like arch) so the thin surface never goes
    # fully edge-on and rasterizes into dots at flap extremes.
    WLR = (0.05, 0.12, 0.020)    # leading root
    WTR = (0.03, -0.14, 0.005)   # trailing root (buried in the body)
    WW = (0.40, 0.04, 0.070)     # wrist (leading, arched high)
    WTM = (0.34, -0.20, 0.030)   # trailing mid
    WTIP = (0.72, -0.26, 0.000)  # wingtip (dropped below the wrist arch)
    right_wing = [(WLR, WW, WTR), (WW, WTM, WTR), (WW, WTIP, WTM)]
    faces += right_wing
    # Left wing: mirror x.
    faces += [tuple((-vx, vy, vz) for (vx, vy, vz) in tri[::-1])
              for tri in right_wing]

    tip_x, deadzone = 0.72, 0.06
    rows = []
    for tri in faces:
        a, b, c = (np.asarray(v, dtype=np.float64) for v in tri)
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise ValueError(f"degenerate bird face: {tri}")
        n /= norm
        for v in (a, b, c):
            frac = np.sign(v[0]) * max(abs(v[0]) - deadzone, 0.0) \
                / (tip_x - deadzone)
            rows.append([*(v * SCALE), *n, frac])
    data = np.asarray(rows, dtype=np.float32)
    return data, len(rows)


def _perspective_vp(origin: np.ndarray, camera: Camera,
                    size: Tuple[int, int]) -> np.ndarray:
    """World->clip matrix matching the raymarcher's ray construction.

    Vertical FOV, clip +y = image top, WebGPU depth in [0, 1].
    """
    forward, right, up = camera.basis()
    w, h = size
    aspect = w / h
    f = 1.0 / np.tan(np.deg2rad(camera.fov) * 0.5)

    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
    view[:3, 3] = -view[:3, :3] @ origin

    proj = np.zeros((4, 4))
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = FAR / (NEAR - FAR)
    proj[2, 3] = NEAR * FAR / (NEAR - FAR)
    proj[3, 2] = -1.0
    return proj @ view


def _bird_rotation(heading_deg: float, pitch_deg: float,
                   bank_deg: float) -> np.ndarray:
    """Local (x right, y forward, z up) -> world rotation matrix (3x3).

    heading is a met azimuth (0 = N, 90 = E); positive bank rolls right
    (into a rightward turn); positive pitch climbs.
    """
    fwd = direction_from_azimuth_elevation(heading_deg, pitch_deg)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    nrm = np.linalg.norm(right)
    if nrm < 1e-6:  # pitched vertical; pick any horizontal right
        right = np.array([1.0, 0.0, 0.0])
        nrm = 1.0
    right /= nrm
    up = np.cross(right, fwd)
    b = radians(bank_deg)
    right_b = cos(b) * right - sin(b) * up
    up_b = cos(b) * up + sin(b) * right
    return np.column_stack([right_b, fwd, up_b])


class Bird:
    """GPU resources + animation state for the flying subject.

    Shares the device and the resident sigma texture/sampler with an
    :class:`~cloudyview.soar.engine.InteractiveRenderer`; draws as a second,
    tiny raster pass (own depth buffer for self-occlusion, alpha-blended
    over the finished volume frame).
    """

    def __init__(self, renderer):
        self.renderer = renderer
        device = renderer.device

        data, self.n_vertices = _build_mesh()
        self._vbuf = device.create_buffer_with_data(
            label="bird-mesh",
            data=data.tobytes(),
            usage=wgpu.BufferUsage.VERTEX,
        )
        self._ubuf = device.create_buffer(
            label="bird-uniforms",
            size=_UNIFORM_NBYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self._bind_group_layout = device.create_bind_group_layout(entries=[
            {
                "binding": 0,
                "visibility": (wgpu.ShaderStage.VERTEX
                               | wgpu.ShaderStage.FRAGMENT),
                "buffer": {"type": "uniform"},
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {"sample_type": "float", "view_dimension": "3d"},
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": "filtering"},
            },
        ])
        self._bind_group = device.create_bind_group(
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._ubuf,
                                            "offset": 0,
                                            "size": _UNIFORM_NBYTES}},
                {"binding": 1, "resource": renderer._texture.create_view()},
                {"binding": 2, "resource": renderer._sampler},
            ],
        )
        self._pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self._bind_group_layout]
        )
        self._shader = device.create_shader_module(
            label="bird", code=SHADER_PATH.read_text()
        )
        self._pipelines = {}   # target format -> pipeline
        self._depth = None     # {"size": (w, h), "view": ...}

        # --- Animation state (world units / degrees / radians as noted) ---
        self.position = np.zeros(3)   # world metres
        self.heading = 0.0            # smoothed met azimuth, deg
        self.view_elevation = 0.0     # smoothed camera elevation, deg
        self.bank = 0.0               # deg, + rolls right
        self.pitch = 0.0              # deg, + climbs
        self.flap_phase = 0.0         # rad
        self.flap_amp = FLAP_AMPLITUDE
        self.flap_angle = REST_DIHEDRAL  # rad, current wing angle
        self._speed = 0.0             # smoothed m/s
        self._vz = 0.0                # smoothed vertical velocity m/s
        self._clock = 0.0
        self._prev_origin = None      # for velocity estimation

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def _place(self, origin: np.ndarray, bob: float) -> None:
        """Anchor the bird ahead of / below the smoothed view direction."""
        cam = Camera(azimuth=self.heading, elevation=self.view_elevation)
        forward, _right, up = cam.basis()
        self.position = (np.asarray(origin, dtype=np.float64)
                         + forward * DISTANCE - up * DROP
                         + np.array([0.0, 0.0, bob]))

    def _flap(self, dt: float) -> float:
        """Advance the wingbeat; returns the body bob (m)."""
        glide = _smoothstep(self._speed, GLIDE_LO, GLIDE_HI)
        amp_target = FLAP_AMPLITUDE * (1.0 - glide)
        k = 1.0 - exp(-dt / TAU_FLAP_AMP)
        self.flap_amp += (amp_target - self.flap_amp) * k

        v = min(self._speed / GLIDE_LO, 1.0)
        hz = FLAP_HZ_SLOW + (FLAP_HZ_FAST - FLAP_HZ_SLOW) * v
        self.flap_phase = (self.flap_phase + 2.0 * pi * hz * dt) % (2.0 * pi)

        center = REST_DIHEDRAL + (GLIDE_DIHEDRAL - REST_DIHEDRAL) * glide
        self.flap_angle = center + self.flap_amp * sin(self.flap_phase)

        amp_frac = self.flap_amp / FLAP_AMPLITUDE
        return (BOB_AMPLITUDE * amp_frac * sin(self.flap_phase - 1.2)
                + IDLE_BOB * sin(2.0 * pi * IDLE_BOB_HZ * self._clock))

    def update(self, dt: float, origin, azimuth: float,
               elevation: float) -> None:
        """Per-frame dynamics from the live camera (windowed app).

        Parameters: frame time (s), camera world origin (m), camera met
        azimuth and elevation (deg). Everything else — banking, pitch,
        glide/flap blend, bob — is derived and smoothed here.
        """
        origin = np.asarray(origin, dtype=np.float64)
        dt = min(max(dt, 1e-4), 0.1)
        self._clock += dt

        if self._prev_origin is None:
            # First frame: snap, no dynamics.
            self._prev_origin = origin.copy()
            self.heading = azimuth
            self.view_elevation = elevation
            self._place(origin, self._flap(dt))
            return

        # Velocity estimate (smoothed).
        vel = (origin - self._prev_origin) / dt
        self._prev_origin = origin.copy()
        ks = 1.0 - exp(-dt / TAU_SPEED)
        self._speed += (float(np.linalg.norm(vel)) - self._speed) * ks
        self._vz += (float(vel[2]) - self._vz) * ks

        # Heading/view lag: the bird swings through turns and settles.
        kh = 1.0 - exp(-dt / TAU_HEADING)
        daz = ((azimuth - self.heading + 180.0) % 360.0) - 180.0
        self.heading = (self.heading + daz * kh) % 360.0
        self.view_elevation += (elevation - self.view_elevation) * kh

        # Bank into turns: roll follows the smoothed heading rate.
        heading_rate = daz * kh / dt   # deg/s
        bank_target = float(np.clip(heading_rate * BANK_PER_DEG_S,
                                    -BANK_MAX, BANK_MAX))
        kb = 1.0 - exp(-dt / TAU_BANK)
        self.bank += (bank_target - self.bank) * kb

        # Pitch with climb/descent.
        h_speed = max(float(np.hypot(vel[0], vel[1])), 15.0)
        pitch_target = float(np.clip(degrees(atan2(self._vz, h_speed)),
                                     -PITCH_MAX, PITCH_MAX))
        kp = 1.0 - exp(-dt / TAU_PITCH)
        self.pitch += (pitch_target - self.pitch) * kp

        self._place(origin, self._flap(dt))

    def set_static(self, origin, camera: Camera, t: float = 0.0, *,
                   bank: Optional[float] = None,
                   pitch: Optional[float] = None,
                   flap_phase: Optional[float] = None) -> None:
        """Deterministic pose for offscreen rendering (no dynamics).

        The bird cruises along the view direction: steady wingbeat at phase
        2π·FLAP_HZ_FAST·t, climbing/descending with the camera elevation as
        if flying that trajectory, unless `bank`/`pitch` (deg) or
        `flap_phase` (rad) override it.
        """
        self.heading = camera.azimuth
        self.view_elevation = camera.elevation
        self.bank = 0.0 if bank is None else float(bank)
        if pitch is None:
            self.pitch = float(np.clip(0.6 * camera.elevation,
                                       -PITCH_MAX, PITCH_MAX))
        else:
            self.pitch = float(pitch)
        self.flap_amp = FLAP_AMPLITUDE
        if flap_phase is None:
            self.flap_phase = (2.0 * pi * FLAP_HZ_FAST * t) % (2.0 * pi)
        else:
            self.flap_phase = float(flap_phase)
        self.flap_angle = (REST_DIHEDRAL
                           + self.flap_amp * sin(self.flap_phase))
        bob = BOB_AMPLITUDE * sin(self.flap_phase - 1.2)
        self._place(origin, bob)

    # ------------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------------

    def write_uniforms(self, origin, camera: Camera, size: Tuple[int, int],
                       *, sun_azimuth: float, sun_elevation: float,
                       exposure: float, ambient_strength: float) -> None:
        """Pack matrices + params for the current pose and enqueue upload."""
        origin = np.asarray(origin, dtype=np.float64)
        vp = _perspective_vp(origin, camera, size)
        rot = _bird_rotation(self.heading, self.pitch + BODY_PITCH, self.bank)
        model = np.eye(4)
        model[:3, :3] = rot
        model[:3, 3] = self.position
        nrot = np.eye(4)
        nrot[:3, :3] = rot
        sun = direction_from_azimuth_elevation(sun_azimuth, sun_elevation)

        def col_major(m):
            return np.ascontiguousarray(m.T, dtype=np.float32)

        vecs = np.empty((4, 4), dtype=np.float32)
        vecs[0] = [*origin, exposure]
        vecs[1] = [*sun, ambient_strength]
        vecs[2] = [*self.renderer.bmin, self.flap_angle]
        vecs[3] = [*self.renderer.bmax, 0.0]
        blob = (col_major(vp).tobytes() + col_major(model).tobytes()
                + col_major(nrot).tobytes() + vecs.tobytes())
        assert len(blob) == _UNIFORM_NBYTES
        self.renderer.device.queue.write_buffer(self._ubuf, 0, blob)

    _DEPTH_FORMAT = "depth24plus"

    def _pipeline_for(self, target_format: str):
        if target_format not in self._pipelines:
            self._pipelines[target_format] = \
                self.renderer.device.create_render_pipeline(
                    label="bird",
                    layout=self._pipeline_layout,
                    vertex={
                        "module": self._shader,
                        "entry_point": "vs_main",
                        "buffers": [{
                            "array_stride": 7 * 4,
                            "attributes": [
                                {"format": "float32x3", "offset": 0,
                                 "shader_location": 0},
                                {"format": "float32x3", "offset": 12,
                                 "shader_location": 1},
                                {"format": "float32", "offset": 24,
                                 "shader_location": 2},
                            ],
                        }],
                    },
                    primitive={
                        "topology": "triangle-list",
                        # Thin shell viewed from both sides; the fragment
                        # shader flips normals toward the camera.
                        "cull_mode": "none",
                    },
                    depth_stencil={
                        "format": self._DEPTH_FORMAT,
                        "depth_write_enabled": True,
                        "depth_compare": "less",
                    },
                    fragment={
                        "module": self._shader,
                        "entry_point": "fs_main",
                        "targets": [{
                            "format": target_format,
                            "blend": {
                                "color": {
                                    "src_factor": "src-alpha",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                                "alpha": {
                                    "src_factor": "one",
                                    "dst_factor": "one-minus-src-alpha",
                                    "operation": "add",
                                },
                            },
                        }],
                    },
                )
        return self._pipelines[target_format]

    def _depth_view(self, size: Tuple[int, int]):
        """Tiny per-target depth buffer for bird self-occlusion (cached)."""
        if self._depth is None or self._depth["size"] != tuple(size):
            tex = self.renderer.device.create_texture(
                label="bird-depth",
                size=(size[0], size[1], 1),
                format=self._DEPTH_FORMAT,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self._depth = {"size": tuple(size), "view": tex.create_view()}
        return self._depth["view"]

    def encode_pass(self, command_encoder, target_view, target_format: str,
                    size: Tuple[int, int], timestamp_writes=None) -> None:
        """Encode the bird raster pass over an already-rendered frame."""
        desc = {
            "color_attachments": [{
                "view": target_view,
                "load_op": wgpu.LoadOp.load,
                "store_op": wgpu.StoreOp.store,
            }],
            "depth_stencil_attachment": {
                "view": self._depth_view(size),
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.discard,
                "depth_clear_value": 1.0,
            },
        }
        if timestamp_writes is not None:
            desc["timestamp_writes"] = timestamp_writes
        rpass = command_encoder.begin_render_pass(**desc)
        rpass.set_pipeline(self._pipeline_for(target_format))
        rpass.set_bind_group(0, self._bind_group)
        rpass.set_vertex_buffer(0, self._vbuf)
        rpass.draw(self.n_vertices)
        rpass.end()
