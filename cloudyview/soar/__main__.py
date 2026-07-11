"""CLI: python -m cloudyview.soar <file.nc> [--ice ice.nc]

Loads the initial cloud field, then opens the windowed fly-through. In-window
pause menus handle later file opens, loading progress, errors, and behold
progress without native file dialogs.
"""

import argparse
import os
from pathlib import Path
import sys


DEMO_FILENAME = "TWPICE_subvolume_256x256_5km.nc"


def demo_data_path() -> Path:
    """Locate the bundled demo, or the repository copy in a dev checkout."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root is not None:
        return Path(bundle_root) / "data" / DEMO_FILENAME
    return Path(__file__).resolve().parents[2] / "data" / DEMO_FILENAME


def run_offscreen_smoke(filepath: Path, *, size=(96, 54)) -> None:
    """Load ``filepath`` and render one small frame without opening a window."""
    import numpy as np

    from ..camera import Camera
    from ..cloudfield import load
    from .engine import InteractiveRenderer

    field = load(str(filepath))
    flat = np.zeros((4, 4), dtype=np.float32)
    up = np.ones((4, 4), dtype=np.float32)
    renderer = InteractiveRenderer(
        field,
        periodic=False,
        quality_tier="low",
        fif_normals=(flat, flat, up, 1.0),
    )
    # Deliberately exactly one call and one accumulated sample: this is the
    # packaging smoke frame, not a benchmark.
    image = renderer.render(
        Camera(), size=size, jitter=False, accumulate_frames=1
    )
    if image.shape != (size[1], size[0], 3) or image.dtype != np.uint8:
        raise RuntimeError(
            f"unexpected smoke frame: shape={image.shape}, dtype={image.dtype}"
        )
    print(
        "SOAR_SMOKE_OK "
        f"file={filepath.name} shape={image.shape} dtype={image.dtype} "
        f"min={int(image.min())} max={int(image.max())} "
        f"mean={float(image.mean()):.3f} std={float(image.std()):.3f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m cloudyview.soar",
        description="Interactive wgpu fly-through of a 3D cloud field.",
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        help="NetCDF file with the cloud field (default: bundled demo data)",
    )
    parser.add_argument("--ice", default=None,
                        help="separate NetCDF file with the ice variable "
                             "(SAM LPT split-file style)")
    parser.add_argument("--size", default="1280x720",
                        help="window size WxH (default 1280x720)")
    parser.add_argument("--extinction-multiplier", type=float, default=1.0)
    parser.add_argument("--max-fps", type=float, default=120.0,
                        help="frame-rate cap (default 120)")
    parser.add_argument(
        "--tier",
        choices=("high", "medium", "low", "potato", "auto"),
        default="auto",
        help="interactive performance tier (default auto benchmarks at startup)",
    )
    parser.add_argument(
        "--fp16-volume",
        action="store_true",
        help="store cloud extinction in a filterable fp16 texture (applies at load)",
    )
    parser.add_argument(
        "--no-periodic",
        action="store_true",
        help="disable horizontal domain tiling (periodic is on by default; "
             "SAM LES domains are doubly periodic in x/y). Use for "
             "subvolume cutouts, which are not physically periodic and "
             "would show seams at the wrap.",
    )
    parser.add_argument(
        "--camera-position",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="initial camera position in relative coords",
    )
    parser.add_argument("--camera-azimuth", type=float,
                        help="initial camera azimuth in degrees")
    parser.add_argument("--camera-elevation", type=float,
                        help="initial camera elevation in degrees")
    parser.add_argument("--fov", type=float,
                        help="initial vertical field of view in degrees")
    parser.add_argument(
        "--offscreen-smoke",
        action="store_true",
        help="load the selected/default data and render exactly one small "
             "offscreen validation frame, then exit",
    )
    parser.add_argument(
        "--validate-launch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    w, h = (int(v) for v in args.size.lower().split("x"))

    from ..cloudfield import load
    from ..camera import Camera
    from .app import CONTROL_SUMMARY, run_app

    using_demo = args.filepath is None
    filepath = demo_data_path() if using_demo else Path(args.filepath)
    if using_demo:
        print(f"Loading demo data: {filepath} ...")
    else:
        print(f"Loading {filepath} ...")
    if args.offscreen_smoke:
        run_offscreen_smoke(filepath)
        return
    field = load(str(filepath), ice=args.ice)
    camera = None
    if any(v is not None for v in (
        args.camera_position, args.camera_azimuth,
        args.camera_elevation, args.fov,
    )):
        defaults = Camera()
        camera = Camera(
            position=(
                tuple(args.camera_position)
                if args.camera_position is not None
                else defaults.position
            ),
            azimuth=(
                args.camera_azimuth
                if args.camera_azimuth is not None
                else defaults.azimuth
            ),
            elevation=(
                args.camera_elevation
                if args.camera_elevation is not None
                else defaults.elevation
            ),
            fov=args.fov if args.fov is not None else defaults.fov,
        )
    print(f"Loaded {field}")
    if args.validate_launch or os.environ.get("CLOUDYVIEW_SOAR_VALIDATE_LAUNCH"):
        print("SOAR_LAUNCH_OK (window suppressed)")
        return
    print(CONTROL_SUMMARY)
    run_app(field, size=(w, h),
            extinction_multiplier=args.extinction_multiplier,
            max_fps=args.max_fps,
            camera=camera,
            periodic=not args.no_periodic,
            tier=args.tier,
            volume_fp16=args.fp16_volume,
            startup_message=("cloudyview demo data" if using_demo else None))


if __name__ == "__main__":
    main()
