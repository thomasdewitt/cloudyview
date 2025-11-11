"""Basic matplotlib 3D rendering for CloudyView."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import xarray as xr
from typing import Optional, Tuple


def plot_isosurface(
    data: xr.DataArray,
    threshold: float = 0.01,
    title: str = "Cloud Field Isosurface",
    figsize: Tuple[int, int] = (8, 6),
    output_path: Optional[str] = None,
    cmap: str = "viridis"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 3D isosurface of data at specified threshold using matplotlib.

    Parameters
    ----------
    data : xr.DataArray
        3D data array (or 4D with time dimension)
    threshold : float, default=0.01
        Isosurface threshold value
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height). Default (8, 6) for 13" MacBook
    output_path : str, optional
        If provided, save figure to this path (PNG)
    cmap : str
        Colormap name

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    # Remove time dimension if present
    if 'time' in data.dims:
        # Get first (and should be only) timestep
        data = data.isel({dim: 0 for dim in data.dims if 'time' in dim})

    # Get numpy array and coordinates
    data_array = data.values

    # Get coordinate arrays for mesh
    coords = {}
    dims = data.dims
    for i, dim in enumerate(dims):
        if dim in data.coords:
            coords[dim] = data.coords[dim].values
        else:
            # Create default indices if no coordinates
            coords[dim] = np.arange(data.sizes[dim])

    # Create meshgrid from coordinates
    coord_arrays = [coords[dim] for dim in dims]
    grids = np.meshgrid(*coord_arrays, indexing='ij')

    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Find isosurface points (simple approach: extract points above threshold)
    mask = data_array > threshold
    points = np.column_stack([grid[mask] for grid in grids])

    # Plot points if any exist
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=data_array[mask], cmap=cmap, s=1, alpha=0.6)

    # Labels and title
    ax.set_xlabel(dims[0])
    ax.set_ylabel(dims[1])
    ax.set_zlabel(dims[2])
    ax.set_title(title)

    # Set equal aspect ratio for better visualization
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    return fig, ax


def plot_slices(
    data: xr.DataArray,
    title: str = "Cloud Field Slices",
    figsize: Tuple[int, int] = (8, 6),
    output_path: Optional[str] = None,
    slices: Optional[dict] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot 2D slices of 3D data.

    Parameters
    ----------
    data : xr.DataArray
        3D data array (or 4D with time dimension)
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    output_path : str, optional
        If provided, save figure to this path (PNG)
    slices : dict, optional
        Dictionary specifying which slices to plot.
        E.g., {'z': 0} plots xy slice at z=0

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array
    """
    # Remove time dimension if present
    if 'time' in data.dims:
        data = data.isel({dim: 0 for dim in data.dims if 'time' in dim})

    if slices is None:
        # Default: middle slices along each axis
        slices = {}
        for dim in data.dims:
            slices[dim] = data.sizes[dim] // 2

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_count = 0
    for ax, (dim, idx) in zip(axes, slices.items()):
        if plot_count >= 3:
            break

        # Select slice
        slice_data = data.isel({dim: idx})

        # Plot
        im = ax.imshow(slice_data.values, cmap='viridis', origin='lower')
        ax.set_title(f"Slice at {dim}={idx}")
        ax.set_xlabel(f"{slice_data.dims[1]}")
        ax.set_ylabel(f"{slice_data.dims[0]}")
        plt.colorbar(im, ax=ax, label=data.name or "Value")

        plot_count += 1

    fig.suptitle(title)
    plt.tight_layout()

    # Save if requested
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    return fig, axes
