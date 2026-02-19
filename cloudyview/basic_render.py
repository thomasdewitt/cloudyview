"""Basic matplotlib 3D rendering for CloudyView."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import Optional, Tuple

# Cloud color scheme for optical depth visualization
sky_blue = '#3A4AA6'
cloud_colors = matplotlib.colors.LinearSegmentedColormap.from_list(
    'cloud_colors',
    [(0, sky_blue), (1, '#FFFFFF')]
)


def plot_optical_depth(
    optical_depth_2d: np.ndarray,
    output_path: Optional[str] = None,
    cmap=None,
    label_dirs: bool = False,
    print_save: bool = True,
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
    label_dirs : bool, optional
        If True, label N/S/W/E sections of domain (default: False)
    print_save : bool, optional
        If True, print saved filepath (default: True)

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
    im = ax.imshow(optical_depth_2d, cmap=cmap, origin='lower', interpolation='nearest', vmin=0, vmax=1)
    ax.axis('off')

    # Add directional labels if requested
    if label_dirs:
        ny, nx = optical_depth_2d.shape
        # Font size scaled to image size (roughly 10% of domain width)
        fontsize = max(12, int(nx / 10))
        bbox_props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', linewidth=1.5)

        # East (+x, right) - center right, very close to edge
        ax.text(nx - 5, ny / 2, 'E', fontsize=fontsize, color='black',
                ha='right', va='center', bbox=bbox_props, weight='bold')

        # West (-x, left) - center left, very close to edge
        ax.text(5, ny / 2, 'W', fontsize=fontsize, color='black',
                ha='left', va='center', bbox=bbox_props, weight='bold')

        # North (+y, top) - top center, very close to edge
        ax.text(nx / 2, ny - 5, 'N', fontsize=fontsize, color='black',
                ha='center', va='top', bbox=bbox_props, weight='bold')

        # South (-y, bottom) - bottom center, very close to edge
        ax.text(nx / 2, 5, 'S', fontsize=fontsize, color='black',
                ha='center', va='bottom', bbox=bbox_props, weight='bold')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save (required)
    if not output_path:
        raise ValueError("output_path is required - no display mode")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    if print_save:
        print(f"  ✓ Saved {output_path}")
    plt.close(fig)

    return fig, ax
