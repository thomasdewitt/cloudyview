"""CLI: python -m cloudyview.soar <file.nc> [--ice ice.nc]

Loads the initial cloud field, then opens the windowed fly-through. In-window
pause menus handle later file opens, loading progress, errors, and behold
progress without native file dialogs.
"""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m cloudyview.soar",
        description="Interactive wgpu fly-through of a 3D cloud field.",
    )
    parser.add_argument("filepath", help="NetCDF file with the cloud field")
    parser.add_argument("--ice", default=None,
                        help="separate NetCDF file with the ice variable "
                             "(SAM LPT split-file style)")
    parser.add_argument("--size", default="1280x720",
                        help="window size WxH (default 1280x720)")
    parser.add_argument("--extinction-multiplier", type=float, default=1.0)
    parser.add_argument("--max-fps", type=float, default=120.0,
                        help="frame-rate cap (default 120)")
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
    args = parser.parse_args(argv)

    w, h = (int(v) for v in args.size.lower().split("x"))

    from ..cloudfield import load
    from ..camera import Camera
    from .app import CONTROL_SUMMARY, run_app

    print(f"Loading {args.filepath} ...")
    field = load(args.filepath, ice=args.ice)
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
    print(CONTROL_SUMMARY)
    run_app(field, size=(w, h),
            extinction_multiplier=args.extinction_multiplier,
            max_fps=args.max_fps,
            camera=camera,
            periodic=not args.no_periodic)


if __name__ == "__main__":
    main()
