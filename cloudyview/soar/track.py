"""Flight-track recording and offline video re-rendering.

The in-app recorder (R key) captures only the *track* — per-frame
(t, camera) samples plus a scene header — as a small JSON file. The
offline pass resamples that track at an exact output frame rate
(non-uniform Catmull-Rom through the hand-flown samples, which also
smooths low-fps input) and renders each video frame with full converged
temporal accumulation, so the result has none of the in-flight motion
speckle. Frames stream straight into ffmpeg; no intermediate files.

Track schema: {"schema": "cloudyview.track.v1",
               "header": <render_metadata dict>,
               "samples": [[t, x, y, z, azimuth, elevation, fov], ...]}
with camera fields in the Camera relative-coordinate convention.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from ..camera import Camera

TRACK_SCHEMA = "cloudyview.track.v1"
# Relative x/y span the domain as [-1, 1]; a periodic flight-path wrap
# shows up as a near-full-span jump between consecutive samples.
_REL_PERIOD = 2.0
_WRAP_JUMP_THRESHOLD = 1.0


def save_track(path: str | Path, header: dict, samples: Sequence[Sequence[float]]) -> Path:
    """Write a recorded track; returns the path."""
    path = Path(path)
    payload = {
        "schema": TRACK_SCHEMA,
        "header": header,
        "samples": [[float(v) for v in s] for s in samples],
    }
    path.write_text(json.dumps(payload))
    return path


def load_track(path: str | Path) -> tuple[dict, np.ndarray]:
    """Read a track file -> (header, samples[n, 7])."""
    payload = json.loads(Path(path).read_text())
    schema = payload.get("schema")
    if schema != TRACK_SCHEMA:
        raise ValueError(
            f"{path}: expected schema {TRACK_SCHEMA!r}, got {schema!r}."
        )
    samples = np.asarray(payload["samples"], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 7:
        raise ValueError(
            f"{path}: samples must be [n, 7] (t, x, y, z, az, el, fov); "
            f"got shape {samples.shape}."
        )
    if samples.shape[0] < 2:
        raise ValueError(f"{path}: need at least 2 samples to resample.")
    return payload["header"], samples


def _unwrap_periodic(values: np.ndarray, period: float,
                     threshold: float) -> np.ndarray:
    """Make a wrapped coordinate continuous across period jumps."""
    out = values.astype(np.float64).copy()
    jumps = np.diff(out)
    correction = np.zeros_like(out)
    correction[1:] = np.cumsum(
        np.where(jumps > threshold, -period,
                 np.where(jumps < -threshold, period, 0.0))
    )
    return out + correction


def _catmull_rom(times: np.ndarray, values: np.ndarray,
                 t_out: np.ndarray) -> np.ndarray:
    """Non-uniform (time-parameterized) Catmull-Rom through all samples.

    Barry-Goldman formulation with the real sample times as knots, so
    irregular in-flight frame timing interpolates correctly. Endpoints are
    handled by clamping the outer control points.
    """
    n = len(times)
    idx = np.clip(np.searchsorted(times, t_out, side="right") - 1, 0, n - 2)
    out = np.empty_like(t_out)
    for k, (i, t) in enumerate(zip(idx, t_out)):
        i0, i1, i2, i3 = max(i - 1, 0), i, i + 1, min(i + 2, n - 1)
        t0, t1, t2, t3 = times[i0], times[i1], times[i2], times[i3]
        p0, p1, p2, p3 = values[i0], values[i1], values[i2], values[i3]
        # Degenerate knot spacing (clamped ends / duplicate stamps) falls
        # back to linear inside the segment.
        if t2 <= t1:
            out[k] = p1
            continue

        def lerp(pa, pb, ta, tb):
            if tb <= ta:
                return pa
            w = (t - ta) / (tb - ta)
            return pa + (pb - pa) * w

        a1 = lerp(p0, p1, t0, t1)
        a2 = lerp(p1, p2, t1, t2)
        a3 = lerp(p2, p3, t2, t3)
        b1 = lerp(a1, a2, t0, t2)
        b2 = lerp(a2, a3, t1, t3)
        out[k] = lerp(b1, b2, t1, t2)
    return out


def resample_track(samples: np.ndarray, fps: float, *,
                   periodic: bool = True) -> list[tuple[float, Camera]]:
    """Resample hand-flown samples at exact 1/fps steps -> [(t, Camera)].

    Azimuth is unwrapped before interpolation (359 -> 1 goes through 0);
    in periodic domains the x/y flight-path wrap is unwrapped the same way
    and re-wrapped into [-1, 1) afterwards.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive; got {fps!r}.")
    order = np.argsort(samples[:, 0], kind="stable")
    samples = samples[order]
    keep = np.ones(len(samples), dtype=bool)
    keep[1:] = np.diff(samples[:, 0]) > 0
    samples = samples[keep]
    if len(samples) < 2:
        raise ValueError("track collapses to <2 unique-time samples.")

    times = samples[:, 0]
    x = samples[:, 1]
    y = samples[:, 2]
    if periodic:
        x = _unwrap_periodic(x, _REL_PERIOD, _WRAP_JUMP_THRESHOLD)
        y = _unwrap_periodic(y, _REL_PERIOD, _WRAP_JUMP_THRESHOLD)
    az = np.deg2rad(samples[:, 4])
    az = np.rad2deg(np.unwrap(az))

    t_out = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    cols = {
        "x": _catmull_rom(times, x, t_out),
        "y": _catmull_rom(times, y, t_out),
        "z": _catmull_rom(times, samples[:, 3], t_out),
        "az": _catmull_rom(times, az, t_out),
        "el": np.clip(_catmull_rom(times, samples[:, 5], t_out), -90.0, 90.0),
        "fov": _catmull_rom(times, samples[:, 6], t_out),
    }
    if periodic:
        for key in ("x", "y"):
            cols[key] = (cols[key] + 1.0) % _REL_PERIOD - 1.0

    frames = []
    for k, t in enumerate(t_out):
        frames.append((
            float(t),
            Camera(
                position=(cols["x"][k], cols["y"][k], cols["z"][k]),
                azimuth=float(cols["az"][k] % 360.0),
                elevation=float(cols["el"][k]),
                fov=float(cols["fov"][k]),
            ),
        ))
    return frames


class TrackVideoRender:
    """A track -> mp4 render the caller drives one frame at a time.

    Foreground by design. The windowed app steps this from its draw loop
    rather than handing it to a thread: the renderer and its resident
    volume are not shareable across threads, and giving the GPU wholly to
    the encode is the point — a live progress bar costs one blit a frame.

    Each output frame is rendered offscreen with `accumulate_frames`
    jittered accumulation passes — the converged-still quality the app only
    reaches when the camera stops. Frames are piped to ffmpeg as raw RGB;
    requires the `ffmpeg` binary (no fallback encoder).

    renderer : InteractiveRenderer, optional
        Render with an already-resident volume instead of loading the field
        again. The app passes its own: a second copy of a gigavoxel field
        would not fit beside the first. Its quality tier is set to High for
        the render and restored on close.
    """

    def __init__(self, track_path, out_path, *, fps: float = 60.0,
                 size: tuple[int, int] = (1920, 1080),
                 accumulate_frames: int = 24, crf: int = 18,
                 renderer=None):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH — install it (e.g. `sudo dnf "
                "install ffmpeg`) to encode track videos. Refusing to fall "
                "back to writing frame files."
            )
        from ..cloudfield import load
        from .engine import InteractiveRenderer

        self.track_path = Path(track_path)
        self.out_path = Path(out_path)
        header, samples = load_track(self.track_path)

        source = header.get("source", {})
        liquid = source.get("path")
        if not liquid:
            raise ValueError(f"{self.track_path}: header has no source path.")
        render_opts = header.get("render", {})
        sun = header.get("sun", {})

        # yuv420p wants even dimensions.
        self.width, self.height = (int(size[0]) & ~1, int(size[1]) & ~1)
        self.fps = float(fps)
        self.accumulate_frames = int(accumulate_frames)

        self._restore_tier = None
        if renderer is None:
            print(f"track: loading {liquid}", flush=True)
            field = load(
                liquid,
                ice=source.get("ice_path"),
                liquid_water_var=source.get("liquid_var"),
                ice_water_var=source.get("ice_var"),
            )
            renderer = InteractiveRenderer(
                field,
                periodic=bool(render_opts.get("periodic", True)),
                extinction_multiplier=float(
                    render_opts.get("extinction_multiplier", 1.0)
                ),
                volume_fp16=bool(render_opts.get("volume_fp16", False)),
            )
        else:
            self._restore_tier = renderer.quality_tier
        renderer.set_quality_tier("high", camera_moving=False)
        self.renderer = renderer

        self._frames = resample_track(
            samples, self.fps, periodic=renderer.periodic
        )
        self.total = len(self._frames)
        self.frames_done = 0

        self._uniform_kwargs = {
            "sun_azimuth": float(sun.get("azimuth", 235.0)),
            "sun_elevation": float(sun.get("elevation", 25.0)),
        }
        for key in (
            "gradient_shading_strength", "deep_shadow_ms_suppression",
            "ambient_occlusion_strength", "bounce_depth_attenuation",
            "light_march_lod_degrees", "view_step_lod_degrees",
        ):
            if key in render_opts:
                self._uniform_kwargs[key] = float(render_opts[key])

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps}", "-i", "-",
            "-c:v", "libx264", "-crf", str(int(crf)), "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self.out_path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self._t_start = perf_counter()
        self._closed = False

    @property
    def done(self) -> bool:
        return self.frames_done >= self.total

    def step(self, n: int = 1) -> bool:
        """Render up to n frames. Returns True once every frame is encoded."""
        for _ in range(int(n)):
            if self.done:
                break
            _, camera = self._frames[self.frames_done]
            try:
                image = self.renderer.render(
                    camera, size=(self.width, self.height), jitter=True,
                    accumulate_frames=self.accumulate_frames,
                    **self._uniform_kwargs,
                )
                self._proc.stdin.write(
                    np.ascontiguousarray(image[:, :, :3]).tobytes()
                )
            except BaseException:
                self.abort()
                raise
            self.frames_done += 1
        return self.done

    def progress(self) -> dict:
        elapsed = perf_counter() - self._t_start
        rate = self.frames_done / max(elapsed, 1e-6)
        remaining = self.total - self.frames_done
        return {
            "frame": self.frames_done,
            "total": self.total,
            "percent": 100.0 * self.frames_done / max(self.total, 1),
            "fps": rate,
            "eta": remaining / rate if rate > 0 else None,
            "elapsed": elapsed,
        }

    def _restore(self) -> None:
        if self._restore_tier is not None:
            self.renderer.set_quality_tier(
                self._restore_tier, camera_moving=False
            )
            self._restore_tier = None

    def abort(self) -> None:
        """Stop encoding and leave no half-written video behind."""
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdin.close()
            self._proc.terminate()
        finally:
            self._restore()
            self.out_path.unlink(missing_ok=True)

    def close(self) -> Path:
        """Finalize the encode; raises if ffmpeg failed."""
        if self._closed:
            return self.out_path
        self._closed = True
        try:
            self._proc.stdin.close()
            code = self._proc.wait()
        finally:
            self._restore()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with status {code}.")
        duration = self.total / self.fps
        print(
            f"track: wrote {self.out_path} ({self.total} frames, "
            f"{duration:.1f}s at {self.fps:g} fps)",
            flush=True,
        )
        return self.out_path


def render_track(track_path: str | Path, out_path: str | Path, *,
                 fps: float = 60.0, size: tuple[int, int] = (1920, 1080),
                 accumulate_frames: int = 24,
                 crf: int = 18,
                 renderer=None,
                 progress_callback=None) -> Path:
    """Re-render a recorded track into a video (blocks; prints progress).

    The CLI entry point: drives :class:`TrackVideoRender` to completion.
    """
    render = TrackVideoRender(
        track_path, out_path, fps=fps, size=size,
        accumulate_frames=accumulate_frames, crf=crf, renderer=renderer,
    )
    while not render.step():
        progress = render.progress()
        if progress_callback is not None:
            progress_callback(progress)
        if render.frames_done % 30 == 0:
            eta = progress["eta"]
            print(
                f"track: frame {progress['frame']}/{progress['total']} "
                f"({progress['fps']:.1f} fps render, ETA "
                f"{(eta or 0.0) / 60.0:.1f} min)",
                flush=True,
            )
    if progress_callback is not None:
        progress_callback(render.progress())
    return render.close()
