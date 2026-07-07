"""CLI: python -m cloudyview.soar <file.nc> [--ice ice.nc]

Opens the windowed fly-through for a cloud field. Naming of a proper
console entry point (like glimpse/witness/behold) is deferred until the
subpackage name settles.
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
    args = parser.parse_args(argv)

    w, h = (int(v) for v in args.size.lower().split("x"))

    from ..cloudfield import load
    from .app import run_app

    field = load(args.filepath, ice=args.ice)
    print(f"Loaded {field}")
    print("Controls: WASD move, Space up, LShift/C down, mouse look "
          "(Tab releases, click recaptures), scroll speed, "
          "J jitter toggle, ESC quit")
    run_app(field, size=(w, h),
            extinction_multiplier=args.extinction_multiplier,
            max_fps=args.max_fps)


if __name__ == "__main__":
    main()
