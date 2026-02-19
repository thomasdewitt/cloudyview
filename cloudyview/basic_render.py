"""Basic matplotlib 3D rendering for CloudyView."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import Optional, Tuple, Dict, Any

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
    camera_overlay: Optional[Dict[str, Any]] = None,
    print_save: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D column optical depth from above.

    Parameters
    ----------
    optical_depth_2d : ndarray (ny, nx)
        2D optical depth field
    output_path : str, optional
        Path to save figure (PNG). Required - no display.
    cmap : matplotlib.colors.Colormap, optional
        Colormap (default: cloud_colors, sky blue to white)
    label_dirs : bool, optional
        If True, label N/S/W/E sections of domain (default: False)
    camera_overlay : dict, optional
        Optional camera/FOV annotation with keys:
        - 'camera_xy': (x, y) camera position in image pixel coordinates
        - 'fov_endpoints': list of (x, y) endpoints for FOV rays
        - 'circle_radius': optional radius in pixels; if provided, draws
          a circle instead of FOV rays/dot (for zenith/nadir ambiguity)
    print_save : bool, optional
        If True, print saved filepath (default: True)

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects
    """
    if cmap is None:
        cmap = cloud_colors

    ny, nx = optical_depth_2d.shape

    # Create figure with domain aspect ratio preserved.
    # Keep the longer image side at 2048 px for consistent output quality.
    dpi = 150
    long_side_px = 2048
    if nx >= ny:
        width_px = long_side_px
        height_px = max(1, int(round(long_side_px * ny / nx)))
    else:
        height_px = long_side_px
        width_px = max(1, int(round(long_side_px * nx / ny)))

    figsize = (width_px / dpi, height_px / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    # Plot with no axes/labels
    ax.imshow(
        optical_depth_2d,
        cmap=cmap,
        origin='lower',
        interpolation='nearest',
        vmin=0,
        vmax=1,
        aspect='auto',
    )
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_autoscale_on(False)
    ax.axis('off')

    # Add directional labels if requested
    if label_dirs:
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

    # Optional camera and field-of-view overlay
    if camera_overlay is not None:
        cam_x, cam_y = camera_overlay['camera_xy']
        fov_endpoints = camera_overlay.get('fov_endpoints', [])
        circle_radius = camera_overlay.get('circle_radius')
        linewidth = max(1.5, max(nx, ny) / 900)

        if circle_radius is not None:
            circle = plt.Circle(
                (cam_x, cam_y),
                float(circle_radius),
                fill=False,
                edgecolor='red',
                linewidth=linewidth,
                alpha=0.95,
                clip_on=True,
                zorder=4,
            )
            ax.add_patch(circle)
        else:
            marker_size = max(24.0, max(nx, ny) / 20)

            for end_x, end_y in fov_endpoints:
                ax.plot(
                    [cam_x, end_x],
                    [cam_y, end_y],
                    color='red',
                    linewidth=linewidth,
                    alpha=0.95,
                    solid_capstyle='round',
                    clip_on=True,
                    zorder=4,
                )

            ax.scatter(
                [cam_x],
                [cam_y],
                s=marker_size,
                c='red',
                edgecolors='white',
                linewidths=max(1.0, linewidth * 0.5),
                clip_on=True,
                zorder=5,
            )

    # Save (required)
    if not output_path:
        raise ValueError("output_path is required - no display mode")
    plt.savefig(output_path, dpi=dpi, pad_inches=0)
    if print_save:
        print(f"  ✓ Saved {output_path}")
    plt.close(fig)

    return fig, ax
