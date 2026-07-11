#!/usr/bin/env python
"""Reproducible soar tier/fp16 benchmark matrix.

Run from the repository root via the project's interactive environment. The
GATE file is intentionally an argument because it is not checked into this
worktree.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import cloudyview as cv
from cloudyview import CloudField
from cloudyview.soar.engine import InteractiveRenderer, QUALITY_PRESETS


DEFAULT_TWPICE = Path("data/TWPICE_subvolume_256x256_5km.nc")
HOT_CAMERA = cv.Camera(
    position=(0.0, 0.0, 2.0),
    azimuth=90.0,
    elevation=-10.0,
    fov=100.0,
)


def _parse_size(value: str) -> tuple[int, int]:
    try:
        w, h = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from exc
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return w, h


def _coarsen(field: CloudField, stride: int) -> CloudField:
    if stride < 1:
        raise ValueError(f"gate stride must be >= 1; got {stride}.")
    sl = (slice(None, None, stride), slice(None, None, stride), slice(None))
    return CloudField(
        lwc=field.lwc[sl],
        iwc=None if field.iwc is None else field.iwc[sl],
        x=field.x[::stride],
        y=field.y[::stride],
        z=field.z,
        source=field.source,
        ice_source=field.ice_source,
        liquid_var=field.liquid_var,
        ice_var=field.ice_var,
    )


def _timing_ms(result: dict) -> float:
    key = "gpu_ms_mean" if result["timestamps_used"] else "wall_ms_mean"
    return float(result[key])


def _tier_matrix(
    renderer: InteractiveRenderer,
    camera: cv.Camera,
    size: tuple[int, int],
    *,
    n_warmup: int,
    n_frames: int,
) -> dict:
    results = {}
    for name in QUALITY_PRESETS:
        renderer.set_quality_tier(name, camera_moving=True)
        result = renderer.benchmark(
            camera,
            size=size,
            n_warmup=n_warmup,
            n_frames=n_frames,
            azimuth_step=0.4,
        )
        results[name] = {
            "ms": _timing_ms(result),
            "wall_ms": float(result["wall_ms_mean"]),
            "gpu_ms": result.get("gpu_ms_mean"),
            "timestamps_used": bool(result["timestamps_used"]),
        }
    return results


def _print_matrix(matrix: dict) -> None:
    scenarios = tuple(matrix)
    print("\ntier" + "".join(f" | {name:>18}" for name in scenarios))
    print("-" * (12 + 22 * len(scenarios)))
    for tier in QUALITY_PRESETS:
        cells = []
        for scenario in scenarios:
            timing = matrix[scenario][tier]["ms"]
            high = matrix[scenario]["high"]["ms"]
            cells.append(f"{timing:8.2f} ms {high / timing:5.2f}x")
        print(f"{tier:10}" + " | " + " | ".join(cells))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--gate-ice", type=Path)
    parser.add_argument("--twpice", type=Path, default=DEFAULT_TWPICE)
    parser.add_argument("--size", type=_parse_size, default=(960, 540))
    parser.add_argument("--visual-size", type=_parse_size, default=(480, 270))
    parser.add_argument("--gate-stride", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument(
        "--hot-position", type=float, nargs=3, default=HOT_CAMERA.position
    )
    parser.add_argument("--hot-azimuth", type=float, default=HOT_CAMERA.azimuth)
    parser.add_argument(
        "--hot-elevation", type=float, default=HOT_CAMERA.elevation
    )
    parser.add_argument("--hot-fov", type=float, default=HOT_CAMERA.fov)
    args = parser.parse_args(argv)

    hot_camera = cv.Camera(
        position=tuple(args.hot_position),
        azimuth=args.hot_azimuth,
        elevation=args.hot_elevation,
        fov=args.hot_fov,
    )
    twpice = cv.load(args.twpice)
    gate = cv.load(args.gate, ice=args.gate_ice)
    gate_coarse = _coarsen(gate, args.gate_stride)
    print(f"TWPICE: {twpice}")
    print(f"GATE coarse (stride {args.gate_stride}): {gate_coarse}")
    print(f"GATE hot/full: {gate}")
    print(f"hot camera: {hot_camera}")

    twpice32 = InteractiveRenderer(twpice, periodic=True, volume_fp16=False)
    matrix = {
        "TWPICE 256": _tier_matrix(
            twpice32,
            cv.Camera(),
            args.size,
            n_warmup=args.warmup,
            n_frames=args.frames,
        )
    }

    gate_coarse_renderer = InteractiveRenderer(
        gate_coarse,
        periodic=True,
        volume_fp16=False,
        fif_normals=twpice32.ocean_fif_normals,
        device=twpice32.device,
    )
    matrix["GATE coarse"] = _tier_matrix(
        gate_coarse_renderer,
        cv.Camera(),
        args.size,
        n_warmup=args.warmup,
        n_frames=args.frames,
    )
    del gate_coarse_renderer
    gc.collect()

    gate32 = InteractiveRenderer(
        gate,
        periodic=True,
        volume_fp16=False,
        fif_normals=twpice32.ocean_fif_normals,
        device=twpice32.device,
    )
    matrix["GATE hot/full"] = _tier_matrix(
        gate32,
        hot_camera,
        args.size,
        n_warmup=args.warmup,
        n_frames=args.frames,
    )
    _print_matrix(matrix)

    twpice16 = InteractiveRenderer(
        twpice,
        periodic=True,
        volume_fp16=True,
        fif_normals=twpice32.ocean_fif_normals,
        device=twpice32.device,
    )
    twpice32.set_quality_tier("high", camera_moving=False)
    twpice16.set_quality_tier("high", camera_moving=False)
    image32 = twpice32.render(
        cv.Camera(), args.visual_size, jitter=False, frame_index=0
    )
    image16 = twpice16.render(
        cv.Camera(), args.visual_size, jitter=False, frame_index=0
    )
    delta = np.abs(image16.astype(np.int16) - image32.astype(np.int16))
    visual = {
        "max_over_255": float(delta.max() / 255.0),
        "mean_over_255": float(delta.mean() / 255.0),
        "fp32_volume_bytes": twpice32.volume_nbytes,
        "fp16_volume_bytes": twpice16.volume_nbytes,
    }

    gate16 = InteractiveRenderer(
        gate,
        periodic=True,
        volume_fp16=True,
        fif_normals=twpice32.ocean_fif_normals,
        device=twpice32.device,
    )
    gate32.set_quality_tier("high", camera_moving=True)
    gate16.set_quality_tier("high", camera_moving=True)
    perf32 = gate32.benchmark(
        hot_camera,
        args.size,
        n_warmup=args.warmup,
        n_frames=args.frames,
    )
    perf16 = gate16.benchmark(
        hot_camera,
        args.size,
        n_warmup=args.warmup,
        n_frames=args.frames,
    )
    fp32_ms = _timing_ms(perf32)
    fp16_ms = _timing_ms(perf16)
    fp16_perf = {
        "fp32_ms": fp32_ms,
        "fp16_ms": fp16_ms,
        "speedup": fp32_ms / fp16_ms,
        "gain_percent": (fp32_ms / fp16_ms - 1.0) * 100.0,
    }
    report = {
        "size": args.size,
        "periodic": True,
        "hot_camera": {
            "position": hot_camera.position,
            "azimuth": hot_camera.azimuth,
            "elevation": hot_camera.elevation,
            "fov": hot_camera.fov,
        },
        "matrix": matrix,
        "fp16_visual": visual,
        "fp16_gate_hot_high": fp16_perf,
    }
    print("\nfp16 visual delta:", json.dumps(visual, indent=2))
    print("fp16 GATE hot/High:", json.dumps(fp16_perf, indent=2))
    print("\nJSON report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
