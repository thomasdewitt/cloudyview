"""Basic matplotlib 3D rendering for CloudyView."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import xarray as xr
from typing import Optional, Tuple

# Cloud color scheme for optical depth visualization
sky_blue = '#3A4AA6'
cloud_colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    'cloud_colors',
    [(0, sky_blue), (1, '#FFFFFF')]
)


def plot_isosurface(
    data: xr.DataArray,
    threshold: float = 0.01,
    output_path: Optional[str] = None,
    color: Optional[str] = None,
    color_by_opacity: bool = False,
    opacity_field: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 3D isosurface of data at specified threshold using matplotlib.
    Minimal rendering with sky blue background.

    Parameters
    ----------
    data : xr.DataArray
        3D data array (or 4D with time dimension)
    threshold : float, default=0.01
        Isosurface threshold value
    output_path : str, optional
        Path to save figure (PNG). Required - no display.
    color : str, optional
        Point color (default: white if color_by_opacity=False)
    color_by_opacity : bool, optional
        If True, color points by opacity above them (black to white)
    opacity_field : ndarray, optional
        3D opacity field for coloring (required if color_by_opacity=True)

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

    # Create figure and 3D axis (2048x2048 px = 13.653 inches at 150 DPI)
    dpi = 150
    figsize = (2048/dpi, 2048/dpi)
    sky_blue = '#3A4AA6'
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=sky_blue)
    ax = fig.add_subplot(111, projection='3d', facecolor=sky_blue)

    # Find all points above threshold for scatter plot
    above_threshold = data_array > threshold
    indices = np.argwhere(above_threshold)

    if len(indices) > 0:
        # Map indices to actual coordinates
        points = np.zeros((len(indices), 3))
        for i, idx in enumerate(indices):
            points[i, 0] = coords[dims[0]][idx[0]]
            points[i, 1] = coords[dims[1]][idx[1]]
            points[i, 2] = coords[dims[2]][idx[2]]

        # Calculate voxel size from coordinate spacing
        # Compute mean spacing in each dimension
        spacings = []
        for dim in dims:
            coord = coords[dim]
            if len(coord) > 1:
                # Mean spacing between consecutive points
                spacing = np.mean(np.diff(coord))
                spacings.append(abs(spacing))
            else:
                spacings.append(1.0)

        # Use geometric mean of spacings to get a characteristic voxel size
        voxel_size = np.exp(np.mean(np.log(spacings)))

        # Scale voxel size to marker size (in points^2)
        # This is a heuristic: convert data units to screen points
        # Typical marker sizes range from 1-1000, we'll use voxel_size as a base
        marker_size = max(2, min(50, voxel_size * 10))

        # Get colors and opacity from opacity field
        if color_by_opacity and opacity_field is not None:
            # Get opacity values at each point
            point_opacity = opacity_field[indices[:, 0], indices[:, 1], indices[:, 2]]

            # Map brightness to transparency above: white = transparent (opacity=0), black = opaque (opacity=1)
            brightness = 1.0 - point_opacity

            # Create RGBA colors where alpha matches the actual opacity
            colors = np.zeros((len(indices), 4))
            colors[:, 0] = brightness  # Red channel
            colors[:, 1] = brightness  # Green channel
            colors[:, 2] = brightness  # Blue channel
            colors[:, 3] = point_opacity  # Alpha = actual opacity

            # Scatter plot with per-point colors and alpha
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=colors,
                s=marker_size,
                alpha=1.0,  # Alpha is already in the color array
                edgecolors='none'
            )
        else:
            # Single color scatter plot
            if color is None:
                color = "white"
            ax.scatter(
                points[:, 0], points[:, 1], points[:, 2],
                c=color,
                s=marker_size,
                alpha=0.8,
                edgecolors='none'
            )

    # Calculate physical extents from coordinates
    x_extent = coords['x'].max() - coords['x'].min() if len(coords['x']) > 1 else 1.0
    y_extent = coords['y'].max() - coords['y'].min() if len(coords['y']) > 1 else 1.0
    z_extent = coords['z'].max() - coords['z'].min() if len(coords['z']) > 1 else 1.0

    # Normalize aspect ratio to physical proportions
    extents = np.array([x_extent, y_extent, z_extent])
    min_extent = extents.min()
    aspect_ratio = (extents / min_extent).tolist()

    # Remove ALL visual elements including axis spines and lines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_title('')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    # Remove axis lines
    ax.xaxis.line.set_linewidth(0)
    ax.yaxis.line.set_linewidth(0)
    ax.zaxis.line.set_linewidth(0)
    # Remove spines
    for spine in ax.spines.values():
        spine.set_edgecolor('none')

    # Set aspect ratio to match physical proportions
    ax.set_box_aspect(aspect_ratio)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save (required)
    if not output_path:
        raise ValueError("output_path is required - no display mode")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0,
                facecolor='black')
    print(f"  ✓ Saved {output_path}")
    plt.close(fig)

    return fig, ax


def plot_optical_depth(
    optical_depth_2d: np.ndarray,
    output_path: Optional[str] = None,
    cmap=None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D column optical depth from above.

    Parameters
    ----------
    optical_depth_2d : ndarray (nx, ny)
        2D optical depth field
    output_path : str, optional
        Path to save figure (PNG). Required - no display.
    cmap : matplotlib.colors.Colormap, optional
        Colormap (default: cloud_colors, sky blue to white)

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    if cmap is None:
        cmap = cloud_colors

    # Create figure (2048x2048 px = 13.653 inches at 150 DPI)
    dpi = 150
    figsize = (2048/dpi, 2048/dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)

    # Plot with no axes/labels
    im = ax.imshow(optical_depth_2d, cmap=cmap, origin='lower', interpolation='nearest')
    ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save (required)
    if not output_path:
        raise ValueError("output_path is required - no display mode")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    print(f"  ✓ Saved {output_path}")
    plt.close(fig)

    return fig, ax
