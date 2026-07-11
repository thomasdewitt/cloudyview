#!/usr/bin/env python3
"""Frozen Soar entrypoint and one-frame bundle smoke-test launcher.

PyInstaller freezes this file. In a bundle it imports the packaged app modules
and delegates to Soar's CLI. From a checkout it locates the built executable
and asks that executable to load its bundled demo and render one frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _run_frozen() -> None:
    # Explicit imports exercise the frozen window shell and rendering module,
    # even though the smoke path itself remains offscreen/headless.
    import cloudyview.soar.app  # noqa: F401
    import cloudyview.soar.engine  # noqa: F401
    from cloudyview.soar.__main__ import main

    main()


def _default_executable(dist: Path) -> Path:
    if sys.platform == "win32":
        return dist / "cloudyview-soar" / "cloudyview-soar.exe"
    if sys.platform == "darwin":
        return (
            dist
            / "CloudyView Soar.app"
            / "Contents"
            / "MacOS"
            / "cloudyview-soar"
        )
    return dist / "cloudyview-soar" / "cloudyview-soar"


def _launch_bundle(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Run exactly one offscreen frame in the frozen Soar bundle."
    )
    parser.add_argument(
        "--dist",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dist",
        help="PyInstaller dist directory (default: repository dist/)",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        help="explicit frozen executable path",
    )
    args = parser.parse_args(argv)
    executable = args.executable or _default_executable(args.dist.resolve())
    if not executable.is_file():
        raise SystemExit(f"bundled executable not found: {executable}")
    print(f"Running one frozen offscreen frame: {executable}", flush=True)
    subprocess.run([str(executable), "--offscreen-smoke"], check=True)


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        _run_frozen()
    else:
        _launch_bundle()
