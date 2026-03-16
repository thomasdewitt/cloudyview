"""Shared CLI helpers for CloudyView command entrypoints."""

import argparse
from textwrap import dedent


class CloudyViewHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
):
    """Preserve section formatting while showing parser defaults."""


DATA_SELECTION_HELP = dedent(
    """
    Input dataset selection:
      CloudyView normally auto-detects the liquid water array, optional ice water
      array, and the x/y/z coordinate arrays from common variable names.

      Use the override flags below when:
      - the variables use non-standard names
      - the arrays live inside one or more NetCDF groups
      - the dimensions are named something other than x/y/z, lon/lat/height, etc.
      - the coordinate arrays live in a different group from the cloud variables

      Group rules:
      - `--group` sets the default group for all lookups.
      - `--liquid-water-group`, `--ice-water-group`, and `--coords-group` override
        that default for one specific lookup.
      - Use `/` or omit the flag to refer to the root group.

      Coordinate rules:
      - CloudyView still requires physical x/y/z coordinates. Integer index axes are
        not enough because optical depth and rendering depend on grid spacing.
      - If the dimension names are unusual, pass `--x-dim`, `--y-dim`, and `--z-dim`.
      - If the coordinate variable names are unusual, pass `--x-coord`,
        `--y-coord`, and `--z-coord`.

      Example override patterns:
      - All arrays in one group:
          --group /physics/clouds --liquid-water-var qc_cloud --ice-water-var qi_cloud
      - Variables and coordinates split across groups:
          --liquid-water-group /state/liquid --ice-water-group /state/ice --coords-group /grid
      - Unusual dimension names:
          --x-dim ni --y-dim nj --z-dim nk --x-coord xh --y-coord yh --z-coord zh
    """
).strip()


def add_dataset_selection_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared dataset override flags to a parser."""
    parser.add_argument(
        "--group",
        help="Default NetCDF group for all dataset lookups. Use '/' for the root group.",
    )
    parser.add_argument(
        "--liquid-water-var",
        help="Explicit liquid water variable name. Overrides autodetection.",
    )
    parser.add_argument(
        "--liquid-water-group",
        help="NetCDF group containing the liquid water variable.",
    )
    parser.add_argument(
        "--ice-water-var",
        help="Explicit ice water variable name. Overrides autodetection.",
    )
    parser.add_argument(
        "--ice-water-group",
        help="NetCDF group containing the ice water variable.",
    )
    parser.add_argument(
        "--coords-group",
        help="NetCDF group containing the x/y/z coordinate arrays.",
    )
    parser.add_argument(
        "--x-coord",
        help="Explicit x-coordinate variable name.",
    )
    parser.add_argument(
        "--y-coord",
        help="Explicit y-coordinate variable name.",
    )
    parser.add_argument(
        "--z-coord",
        help="Explicit z-coordinate variable name.",
    )
    parser.add_argument(
        "--x-dim",
        help="Explicit name of the horizontal east-west dimension.",
    )
    parser.add_argument(
        "--y-dim",
        help="Explicit name of the horizontal north-south dimension.",
    )
    parser.add_argument(
        "--z-dim",
        help="Explicit name of the vertical dimension.",
    )


def dataset_selection_kwargs(args: argparse.Namespace) -> dict:
    """Build io.load_and_validate keyword arguments from parsed CLI args."""
    return {
        "liquid_water_var": args.liquid_water_var,
        "ice_water_var": args.ice_water_var,
        "dataset_group": args.group,
        "liquid_water_group": args.liquid_water_group,
        "ice_water_group": args.ice_water_group,
        "coords_group": args.coords_group,
        "x_coord_name": args.x_coord,
        "y_coord_name": args.y_coord,
        "z_coord_name": args.z_coord,
        "x_dim": args.x_dim,
        "y_dim": args.y_dim,
        "z_dim": args.z_dim,
    }
