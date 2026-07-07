#!/usr/bin/env python
"""
Generate reference images for behold render tests.

This script renders all (dataset, view) combinations and saves them as
reference images for regression testing.

Usage:
    python tests/generate_references.py              # Generate all references
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

import imageio.v3 as iio

# The package attribute `cloudyview.behold` is the public render function
# (it shadows the submodule); import the CLI module itself explicitly.
behold = importlib.import_module("cloudyview.behold")

from conftest import (
    CAMERA_CONFIGS,
    REFERENCE_DIR,
    RENDER_SETTINGS,
    TEST_DATA_FILES,
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


def main():
    """Generate reference images."""
    parser = argparse.ArgumentParser(
        description="Generate reference images for behold render tests"
    )
    parser.add_argument(
        "--dataset",
        choices=list(TEST_DATA_FILES.keys()),
        help="Generate only for this dataset",
    )
    parser.add_argument(
        "--view",
        choices=list(CAMERA_CONFIGS.keys()),
        help="Generate only for this view",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reference images",
    )
    args = parser.parse_args()

    # Determine which datasets and views to generate
    datasets = [args.dataset] if args.dataset else list(TEST_DATA_FILES.keys())
    views = [args.view] if args.view else list(CAMERA_CONFIGS.keys())

    print("=" * 60)
    print("CloudyView Reference Image Generator")
    print("=" * 60)
    print(f"\nRender settings:")
    print(f"  Resolution: {RENDER_SETTINGS['resolution']}")
    print(f"  SPP: {RENDER_SETTINGS['spp']}")
    print(f"  Max depth: {RENDER_SETTINGS['max_depth']}")
    print(f"  RR depth: {RENDER_SETTINGS['rr_depth']}")
    print(f"  Backend: {RENDER_SETTINGS['backend']}")
    print(f"  Sun: azimuth={RENDER_SETTINGS['sun_azimuth']}, "
          f"elevation={RENDER_SETTINGS['sun_elevation']}")
    print()

    total = len(datasets) * len(views)
    generated = 0
    skipped = 0

    for dataset_name in datasets:
        print(f"\nDataset: {dataset_name}")
        for view_name in views:
            if generate_reference(dataset_name, view_name, args.overwrite):
                generated += 1
            else:
                skipped += 1

    print("\n" + "=" * 60)
    print(f"Complete: {generated} generated, {skipped} skipped (of {total} total)")
    print("=" * 60)


if __name__ == "__main__":
    main()
