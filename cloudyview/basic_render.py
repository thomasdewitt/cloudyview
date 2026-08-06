"""Basic plotting and image-saving helpers for CloudyView."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from typing import Optional, Tuple, Dict, Any


def quantize_uint8(
    image: np.ndarray,
    dither: bool = True,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Encode a float RGB image (values in [0, 1]) to 8-bit, with dither.

    The renderers accumulate in float and their output is smooth: a sky
    gradient, a stretch of open water, the flank of a cloud. Rounding such a
    ramp to 8 bits draws its iso-contours — the quantization error is a
    *deterministic* function of the value, so it is constant along a contour
    and jumps by a level across it, which the eye reads as Mach banding. No
    amount of renderer work removes that; it is made at the encode.

    The cure is to decorrelate the error from the signal by adding ~1 LSB of
    zero-mean noise before rounding (TPDF: the sum of two uniforms, which is
    what makes the error's *variance* signal-independent as well as its mean,
    so the ramp carries an even grain instead of a modulated one). The
    amplitude is below perception on its own — the render's own Monte Carlo
    grain is an order of magnitude larger — and the mean is preserved exactly,
    so nothing about the image's brightness or look changes.

    Two details that matter:

    - The rounding is round-to-nearest, not the truncation this used to do.
      Truncation is a uniform -0.5 LSB bias, i.e. every image the toolkit has
      ever written was half a level dark; with a symmetric dither it would
      also defeat the dither's zero mean.
    - The dither tapers to zero within one level of the 0 and 255 rails, so a
      clipped highlight or a true black stays exactly clipped rather than
      picking up speckle. Clipping the dithered value instead would reintroduce
      a bias exactly where there is no quantization error to hide.

    Parameters
    ----------
    image : ndarray
        Float image; values outside [0, 1] are clipped.
    dither : bool, optional
        Set False for a bit-exact, noise-free encode (still round-to-nearest).
    seed : int or None, optional
        Seed for the dither. Fixed by default so renders stay reproducible;
        pass None to draw from OS entropy.

    Returns
    -------
    ndarray of uint8, same shape as `image`.
    """
    # float32 throughout: 24 bits of mantissa over a 0..255 range is eight
    # orders of magnitude more than an 8-bit encode can use, and it halves the
    # cost of drawing one dither sample per subpixel.
    v = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0) * np.float32(255)
    if dither:
        rng = np.random.default_rng(seed)
        tpdf = rng.random(v.shape, dtype=np.float32)
        tpdf -= rng.random(v.shape, dtype=np.float32)
        # Full amplitude everywhere except within one level of the rails,
        # where a clipped highlight or a true black must stay exact.
        v = v + tpdf * np.clip(np.minimum(v, np.float32(255) - v), 0.0, 1.0)
    return np.clip(np.rint(v), 0.0, 255.0).astype(np.uint8)


def save_image(
    image: np.ndarray,
    output_path: str,
    dither: bool = True,
    seed: Optional[int] = 0,
) -> None:
    """Save a float RGB image (values in [0, 1]) as an 8-bit PNG.

    Companion to the library render functions (`cv.witness`, `cv.behold`),
    which return arrays and never write files themselves. See
    :func:`quantize_uint8` for the encode, which is dithered.
    """
    from PIL import Image as PILImage

    PILImage.fromarray(quantize_uint8(image, dither=dither, seed=seed)).save(
        str(output_path))

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
