#!/usr/bin/env python
"""
Generate reference images for the render regression tests.

Two independent sets of golden images live under tests/reference_images/:

  behold  Mitsuba path-traced views, one per (dataset, camera).
  soar    The witness/soar WGSL renderer's eight frozen judge views. These
          need a real GPU and are baked on one -- a software rasterizer is
          refused rather than silently used.

Usage:
    python tests/generate_references.py                       # Both sets
    python tests/generate_references.py --target soar         # One set
    python tests/generate_references.py --dataset TWPICE_256  # Single dataset
    python tests/generate_references.py --view ground_view    # Single view
    python tests/generate_references.py --overwrite           # Overwrite existing
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for cloudyview imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib

from conftest import (
    CAMERA_CONFIGS,
    REFERENCE_DIR,
    RENDER_SETTINGS,
    SOAR_REFERENCE_DIR,
    SOAR_RENDER_SETTINGS,
    SOAR_VIEWS,
    TEST_DATA_FILES,
    build_soar_level,
    render_soar_view,
    soar_gpu_adapter,
)


def generate_reference(
    dataset_name: str,
    view_name: str,
    overwrite: bool = False,
) -> bool:
    """
    Generate a single reference image.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., "TWPICE_256").
    view_name : str
        Name of the view (e.g., "ground_view").
    overwrite : bool
        If True, overwrite existing reference images.

    Returns
    -------
    bool
        True if the reference was generated, False if skipped.
    """
    data_file = TEST_DATA_FILES[dataset_name]
    camera_config = CAMERA_CONFIGS[view_name]
    output_path = REFERENCE_DIR / dataset_name / f"{view_name}.png"

    # Check if data file exists
    if not data_file.exists():
        print(f"  SKIP: Data file not found: {data_file}")
        return False

    # Check if reference already exists
    if output_path.exists() and not overwrite:
        print(f"  SKIP: {output_path} already exists (use --overwrite to replace)")
        return False

    print(f"  Rendering {dataset_name}/{view_name}...")

    # The package attribute `cloudyview.behold` is the public render function
    # (it shadows the submodule); import the CLI module itself explicitly.
    # Deferred so that --target soar works without Mitsuba installed.
    behold = importlib.import_module("cloudyview.behold")

    # Render to temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        behold.main(
            filename=str(data_file),
            backend=RENDER_SETTINGS["backend"],
            quality="custom",
            output=str(tmp_path),
            custom_spp=RENDER_SETTINGS["spp"],
            custom_size=RENDER_SETTINGS["resolution"],
            custom_max_depth=RENDER_SETTINGS["max_depth"],
            custom_rr_depth=RENDER_SETTINGS["rr_depth"],
            camera_position=camera_config["position"],
            camera_azimuth=camera_config["azimuth"],
            camera_elevation=camera_config["elevation"],
            camera_fov=camera_config["fov"],
            sun_azimuth=RENDER_SETTINGS["sun_azimuth"],
            sun_elevation=RENDER_SETTINGS["sun_elevation"],
            progress_interval=RENDER_SETTINGS["progress_interval"],
        )

        # behold outputs filename based on max_depth and rr_depth
        rendered_filename = (
            f"behold_ground_view_max_depth={RENDER_SETTINGS['max_depth']}"
            f"_rr_depth={RENDER_SETTINGS['rr_depth']}.png"
        )
        rendered_path = tmp_path / rendered_filename

        if not rendered_path.exists():
            print(f"  ERROR: Render output not found: {rendered_path}")
            return False

        # Copy to reference location
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(rendered_path, output_path)

    print(f"  SAVED: {output_path}")
    return True


def generate_soar_references(only_view: str = None, overwrite: bool = False):
    """Bake the witness/soar golden images.

    Refuses to run without real GPU hardware rather than falling back to a
    software rasterizer: a reference baked on llvmpipe would be a different
    image, and every later run would be measured against it.

    Returns
    -------
    tuple[int, int]
        (generated, skipped)
    """
    import cloudyview as cv

    adapter = soar_gpu_adapter()
    if adapter is None:
        raise RuntimeError(
            "No GPU adapter available. The witness/soar references must be "
            "baked on real hardware -- a software rasterizer (llvmpipe, "
            "SwiftShader) is refused, not used as a fallback."
        )
    info = adapter.info
    print(f"\nAdapter: {info.get('device')} "
          f"({info.get('adapter_type')}, {info.get('backend_type')}, "
          f"driver {info.get('description')})")
    print(f"Render settings: {SOAR_RENDER_SETTINGS['size'][0]}x"
          f"{SOAR_RENDER_SETTINGS['size'][1]}, "
          f"{SOAR_RENDER_SETTINGS['accumulate']} accumulated passes, "
          f"periodic={SOAR_RENDER_SETTINGS['periodic']}")

    names = [only_view] if only_view else list(SOAR_VIEWS)
    pending = [
        n for n in names
        if overwrite or not (SOAR_REFERENCE_DIR / f"{n}.png").exists()
    ]
    generated = 0
    skipped = len(names) - len(pending)
    for name in names:
        if name not in pending:
            print(f"  SKIP: {name} already exists (use --overwrite to replace)")

    if pending:
        SOAR_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        level = build_soar_level()
        for name in pending:
            print(f"  Rendering soar/{name}...")
            image = render_soar_view(level, SOAR_VIEWS[name])
            output_path = SOAR_REFERENCE_DIR / f"{name}.png"
            cv.save_image(image, str(output_path))
            print(f"  SAVED: {output_path}")
            generated += 1

    return generated, skipped


def generate_behold_references(dataset: str = None, only_view: str = None,
                               overwrite: bool = False):
    """Bake the Mitsuba path-traced golden images.

    Returns
    -------
    tuple[int, int]
        (generated, skipped)
    """
    # Determine which datasets and views to generate
    datasets = [dataset] if dataset else list(TEST_DATA_FILES.keys())
    views = [only_view] if only_view else list(CAMERA_CONFIGS.keys())

    print(f"\nRender settings:")
    print(f"  Resolution: {RENDER_SETTINGS['resolution']}")
    print(f"  SPP: {RENDER_SETTINGS['spp']}")
    print(f"  Max depth: {RENDER_SETTINGS['max_depth']}")
    print(f"  RR depth: {RENDER_SETTINGS['rr_depth']}")
    print(f"  Backend: {RENDER_SETTINGS['backend']}")
    print(f"  Sun: azimuth={RENDER_SETTINGS['sun_azimuth']}, "
          f"elevation={RENDER_SETTINGS['sun_elevation']}")
    print()

    generated = 0
    skipped = 0

    for dataset_name in datasets:
        print(f"\nDataset: {dataset_name}")
        for view_name in views:
            if generate_reference(dataset_name, view_name, overwrite):
                generated += 1
            else:
                skipped += 1

    return generated, skipped


def main():
    """Generate reference images."""
    parser = argparse.ArgumentParser(
        description="Generate reference images for the render regression tests"
    )
    parser.add_argument(
        "--target",
        choices=["all", "behold", "soar"],
        default="all",
        help="Which reference set to generate (default: all)",
    )
    parser.add_argument(
        "--dataset",
        choices=list(TEST_DATA_FILES.keys()),
        help="Generate only for this dataset (behold only)",
    )
    parser.add_argument(
        "--view",
        choices=list(CAMERA_CONFIGS.keys()) + list(SOAR_VIEWS.keys()),
        help="Generate only for this view",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reference images",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CloudyView Reference Image Generator")
    print("=" * 60)

    generated = skipped = 0

    if args.target in ("all", "soar"):
        soar_view = args.view if args.view in SOAR_VIEWS else None
        if args.view and soar_view is None and args.target == "soar":
            parser.error(f"--view {args.view} is not a soar view; "
                         f"choose from {list(SOAR_VIEWS)}")
        print("\nTarget: soar (witness/soar WGSL renderer)")
        g, s = generate_soar_references(soar_view, args.overwrite)
        generated += g
        skipped += s

    if args.target in ("all", "behold"):
        behold_view = args.view if args.view in CAMERA_CONFIGS else None
        if args.view and behold_view is None and args.target == "behold":
            parser.error(f"--view {args.view} is not a behold view; "
                         f"choose from {list(CAMERA_CONFIGS)}")
        print("\nTarget: behold (Mitsuba path tracer)")
        g, s = generate_behold_references(args.dataset, behold_view, args.overwrite)
        generated += g
        skipped += s

    print("\n" + "=" * 60)
    print(f"Complete: {generated} generated, {skipped} skipped")
    print("=" * 60)


if __name__ == "__main__":
    main()
