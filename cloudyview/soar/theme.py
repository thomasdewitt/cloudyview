"""Visual theme for soar's in-window UI.

One place for the palette, typography, ImGui style geometry, and the small
set of styled widgets (menu buttons with key-cap hints, overline/title
headers, themed progress bars) that the pause menus and overlays share.

Design intent: smoked-glass panels floating over the sky — a dark,
slightly blue glass with a single sky-cyan accent, generous padding,
consistent rounding, and a real UI font instead of ImGui's debug bitmap
font. Numeric readouts use a monospaced face so digits do not jitter.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Fonts. Resolve installed faces through fontconfig rather than assuming a
# distro-specific directory layout. Adwaita Sans is an Inter-derived UI face;
# the fallbacks are common permissively-licensed system fonts. We fail loudly
# if fontconfig or a usable TTF is missing — silently falling back to
# ProggyClean would defeat the theme.
# ---------------------------------------------------------------------------

FONT_BODY_FAMILIES = (
    "Adwaita Sans",
    "Inter",
    "Roboto",
    "Carlito",
    "Droid Sans",
    "DejaVu Sans",
)

FONT_MONO_FAMILIES = (
    "Adwaita Mono",
    "JetBrains Mono",
    "Droid Sans Mono",
    "DejaVu Sans Mono",
)


def find_font(families) -> str:
    """Resolve the first preferred regular TTF using ``fc-list``.

    ImGui copies the face into its own atlas, so this is a startup-only lookup.
    Restricting the result to a real ``.ttf`` keeps the font backend behavior
    predictable across Linux distributions.
    """
    command = (
        "fc-list",
        "--format=%{file}\t%{family}\t%{style}\n",
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "soar theme: fontconfig 'fc-list' is required to locate an "
            "embedded UI font. Install fontconfig and a supported TTF face."
        ) from exc

    records = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        path_text, family_text, style_text = parts
        path = Path(path_text)
        family_names = {name.strip().casefold() for name in family_text.split(",")}
        style_names = {name.strip().casefold() for name in style_text.split(",")}
        if (
            path.suffix.casefold() == ".ttf"
            and path.is_file()
            and style_names.intersection({"regular", "book", "roman"})
        ):
            records.append((path, family_names))

    for family in families:
        wanted = family.casefold()
        for path, family_names in records:
            if wanted in family_names:
                return str(path)
    raise RuntimeError(
        "soar theme: fc-list found no usable regular TTF for:\n  "
        + "\n  ".join(families)
        + "\nInstall one of these families (e.g. Adwaita or DejaVu)."
    )


def _rgba(hex_rgb: str, alpha: float) -> tuple:
    value = int(hex_rgb.lstrip("#"), 16)
    return (
        ((value >> 16) & 0xFF) / 255.0,
        ((value >> 8) & 0xFF) / 255.0,
        (value & 0xFF) / 255.0,
        float(alpha),
    )


# ---------------------------------------------------------------------------
# Palette — one accent, everything else is glass and cool greys.
# ---------------------------------------------------------------------------

ACCENT_HEX = "#6EC1F2"          # sky cyan, the one accent
INK_HEX = "#0D141D"             # panel glass
ERROR_HEX = "#F0958F"           # desaturated coral, error headers only

TEXT = _rgba("#E8EEF4", 1.00)
TEXT_MUTED = _rgba("#9AA9BC", 1.00)
TEXT_FAINT = _rgba("#66778C", 1.00)
ACCENT = _rgba(ACCENT_HEX, 1.00)
ACCENT_DIM = _rgba(ACCENT_HEX, 0.72)
ERROR = _rgba(ERROR_HEX, 1.00)

PANEL_BG = _rgba(INK_HEX, 0.82)
CHILD_BG = _rgba("#080D13", 0.28)
BORDER = _rgba("#FFFFFF", 0.09)

BUTTON = _rgba("#FFFFFF", 0.055)
BUTTON_HOVERED = _rgba(ACCENT_HEX, 0.24)
BUTTON_ACTIVE = _rgba(ACCENT_HEX, 0.32)

PROGRESS_TRACK = _rgba("#FFFFFF", 0.08)
PROGRESS_FILL = _rgba(ACCENT_HEX, 0.95)

KEYCAP_BG = _rgba("#FFFFFF", 0.06)
KEYCAP_BORDER = _rgba("#FFFFFF", 0.16)

SCROLL_GRAB = _rgba("#FFFFFF", 0.14)
SCROLL_GRAB_HOVERED = _rgba("#FFFFFF", 0.22)
SCROLL_GRAB_ACTIVE = _rgba(ACCENT_HEX, 0.50)

# Typography scale (logical px; ImGui 1.92 sizes fonts dynamically).
SIZE_BODY = 17.0
SIZE_TITLE = 26.0
SIZE_OVERLINE = 13.0
SIZE_CAPTION = 13.5
SIZE_MONO = 15.0
SIZE_MONO_SMALL = 13.0
SIZE_KEYCAP = 12.0

BUTTON_HEIGHT = 44.0
OVERLINE_TRACKING = 2.4         # extra advance px for letterspaced labels


class SoarTheme:
    """Fonts + style + styled widget helpers for the soar ImGui layer."""

    def __init__(self, imgui):
        self.imgui = imgui
        io = imgui.get_io()
        atlas = io.fonts

        body_path = find_font(FONT_BODY_FAMILIES)
        mono_path = find_font(FONT_MONO_FAMILIES)
        self.body_font_path = body_path
        self.mono_font_path = mono_path

        # First font added becomes the ImGui default (body text).
        self.font_body = atlas.add_font_from_file_ttf(body_path, SIZE_BODY)
        tracked_cfg = imgui.ImFontConfig()
        tracked_cfg.glyph_extra_advance_x = OVERLINE_TRACKING
        self.font_tracked = atlas.add_font_from_file_ttf(
            body_path, SIZE_OVERLINE, tracked_cfg
        )
        self.font_mono = atlas.add_font_from_file_ttf(mono_path, SIZE_MONO)

        self.apply_style()

    # -- style ----------------------------------------------------------

    def apply_style(self) -> None:
        imgui = self.imgui
        style = imgui.get_style()

        style.window_rounding = 14.0
        style.child_rounding = 10.0
        style.popup_rounding = 10.0
        style.frame_rounding = 9.0
        style.grab_rounding = 9.0
        style.scrollbar_rounding = 12.0
        style.window_border_size = 1.0
        style.child_border_size = 0.0
        style.popup_border_size = 1.0
        style.frame_border_size = 0.0
        style.window_padding = (30.0, 26.0)
        style.frame_padding = (16.0, 11.0)
        style.item_spacing = (12.0, 9.0)
        style.item_inner_spacing = (8.0, 6.0)
        style.scrollbar_size = 12.0
        style.grab_min_size = 12.0
        style.button_text_align = (0.0, 0.5)
        style.separator_text_align = (0.0, 0.5)
        style.window_title_align = (0.0, 0.5)

        col = imgui.Col_
        set_color = style.set_color_
        set_color(col.text, TEXT)
        set_color(col.text_disabled, TEXT_FAINT)
        set_color(col.window_bg, PANEL_BG)
        set_color(col.child_bg, CHILD_BG)
        set_color(col.popup_bg, PANEL_BG)
        set_color(col.border, BORDER)
        set_color(col.border_shadow, (0.0, 0.0, 0.0, 0.0))
        set_color(col.frame_bg, BUTTON)
        set_color(col.frame_bg_hovered, BUTTON_HOVERED)
        set_color(col.frame_bg_active, BUTTON_ACTIVE)
        set_color(col.title_bg, PANEL_BG)
        set_color(col.title_bg_active, PANEL_BG)
        set_color(col.title_bg_collapsed, PANEL_BG)
        set_color(col.scrollbar_bg, (0.0, 0.0, 0.0, 0.0))
        set_color(col.scrollbar_grab, SCROLL_GRAB)
        set_color(col.scrollbar_grab_hovered, SCROLL_GRAB_HOVERED)
        set_color(col.scrollbar_grab_active, SCROLL_GRAB_ACTIVE)
        set_color(col.check_mark, ACCENT)
        set_color(col.slider_grab, ACCENT_DIM)
        set_color(col.slider_grab_active, ACCENT)
        set_color(col.button, BUTTON)
        set_color(col.button_hovered, BUTTON_HOVERED)
        set_color(col.button_active, BUTTON_ACTIVE)
        set_color(col.header, BUTTON)
        set_color(col.header_hovered, BUTTON_HOVERED)
        set_color(col.header_active, BUTTON_ACTIVE)
        set_color(col.separator, BORDER)
        set_color(col.separator_hovered, ACCENT_DIM)
        set_color(col.separator_active, ACCENT)
        set_color(col.plot_histogram, PROGRESS_FILL)
        set_color(col.plot_histogram_hovered, PROGRESS_FILL)
        set_color(col.nav_cursor, _rgba(ACCENT_HEX, 0.55))
        set_color(col.text_selected_bg, _rgba(ACCENT_HEX, 0.30))
        set_color(col.modal_window_dim_bg, (0.0, 0.0, 0.0, 0.35))

    # -- small drawing utilities -----------------------------------------

    def _u32(self, rgba) -> int:
        return self.imgui.color_convert_float4_to_u32(rgba)

    def push_font(self, font, size: float) -> None:
        self.imgui.push_font(font, size)

    def pop_font(self) -> None:
        self.imgui.pop_font()

    # -- typography -------------------------------------------------------

    def overline(self, text: str, color=None) -> None:
        """Letterspaced uppercase kicker above a title."""
        imgui = self.imgui
        self.push_font(self.font_tracked, SIZE_OVERLINE)
        imgui.push_style_color(imgui.Col_.text, color or ACCENT_DIM)
        imgui.text(text.upper())
        imgui.pop_style_color()
        self.pop_font()

    def title(self, text: str) -> None:
        imgui = self.imgui
        self.push_font(self.font_body, SIZE_TITLE)
        imgui.text(text)
        self.pop_font()

    def header(self, kicker: str, title_text: str | None = None,
               *, kicker_color=None) -> None:
        """Standard panel header: overline kicker, optional title, rule."""
        self.overline(kicker, color=kicker_color)
        if title_text is not None:
            self.title(title_text)
        self.accent_rule(color=kicker_color)
        self.imgui.dummy((1.0, 6.0))

    def accent_rule(self, width: float = 46.0, color=None) -> None:
        """Short accent underline used beneath panel titles."""
        imgui = self.imgui
        pos = imgui.get_cursor_screen_pos()
        draw_list = imgui.get_window_draw_list()
        draw_list.add_rect_filled(
            (pos.x, pos.y + 4.0), (pos.x + width, pos.y + 7.0),
            self._u32(color or ACCENT), 2.0,
        )
        imgui.dummy((width, 8.0))

    def body_text(self, text: str, color=None, *, wrapped: bool = False) -> None:
        imgui = self.imgui
        if color is not None:
            imgui.push_style_color(imgui.Col_.text, color)
        if wrapped:
            imgui.text_wrapped(text)
        else:
            imgui.text(text)
        if color is not None:
            imgui.pop_style_color()

    def caption(self, text: str, color=None, *, wrapped: bool = False) -> None:
        self.push_font(self.font_body, SIZE_CAPTION)
        self.body_text(text, color or TEXT_MUTED, wrapped=wrapped)
        self.pop_font()

    def mono_text(self, text: str, color=None,
                  size: float = SIZE_MONO) -> None:
        self.push_font(self.font_mono, size)
        self.body_text(text, color or TEXT_MUTED)
        self.pop_font()

    # -- widgets ----------------------------------------------------------

    def _draw_right_slot(self, rect_min, rect_max, hint: str | None,
                         right_text: str | None) -> None:
        """Key-cap hint or plain right-aligned mono text inside an item."""
        imgui = self.imgui
        draw_list = imgui.get_window_draw_list()
        pad_r = 14.0
        if hint:
            self.push_font(self.font_mono, SIZE_KEYCAP)
            text_size = imgui.calc_text_size(hint)
            cap_h = text_size.y + 8.0
            cap_w = max(text_size.x + 14.0, cap_h)
            x1 = rect_max.x - pad_r
            x0 = x1 - cap_w
            y0 = (rect_min.y + rect_max.y - cap_h) * 0.5
            y1 = y0 + cap_h
            draw_list.add_rect_filled(
                (x0, y0), (x1, y1), self._u32(KEYCAP_BG), 5.0)
            draw_list.add_rect(
                (x0, y0), (x1, y1), self._u32(KEYCAP_BORDER), 5.0, 1.0)
            draw_list.add_text(
                ((x0 + x1 - text_size.x) * 0.5, (y0 + y1 - text_size.y) * 0.5),
                self._u32(TEXT_MUTED), hint,
            )
            self.pop_font()
        elif right_text:
            self.push_font(self.font_mono, SIZE_MONO_SMALL)
            text_size = imgui.calc_text_size(right_text)
            draw_list.add_text(
                (rect_max.x - pad_r - text_size.x,
                 (rect_min.y + rect_max.y - text_size.y) * 0.5),
                self._u32(TEXT_FAINT), right_text,
            )
            self.pop_font()

    def menu_button(self, label: str, hint: str | None = None, *,
                    width: float = 0.0, height: float = BUTTON_HEIGHT,
                    right_text: str | None = None, sublabel: str | None = None,
                    text_color=None) -> bool:
        """Full-width themed button: label left, key-cap hint right.

        ``sublabel`` draws a faint secondary note just after the label
        (e.g. a duration estimate on a quality preset).
        """
        imgui = self.imgui
        if width <= 0.0:
            width = imgui.get_content_region_avail().x
        if text_color is not None:
            imgui.push_style_color(imgui.Col_.text, text_color)
        label_size = imgui.calc_text_size(label)
        clicked = imgui.button(label, (width, height))
        if text_color is not None:
            imgui.pop_style_color()
        rect_min = imgui.get_item_rect_min()
        rect_max = imgui.get_item_rect_max()
        if sublabel:
            pad_x = imgui.get_style().frame_padding.x
            self.push_font(self.font_body, SIZE_CAPTION)
            sub_size = imgui.calc_text_size(sublabel)
            imgui.get_window_draw_list().add_text(
                (rect_min.x + pad_x + label_size.x + 12.0,
                 (rect_min.y + rect_max.y - sub_size.y) * 0.5),
                self._u32(TEXT_FAINT), sublabel,
            )
            self.pop_font()
        self._draw_right_slot(rect_min, rect_max, hint, right_text)
        return bool(clicked)

    def progress_bar(self, fraction: float | None, *,
                     width: float = 0.0, height: float = 8.0) -> None:
        """Rounded accent progress bar; ``None`` draws an indeterminate sweep."""
        imgui = self.imgui
        if width <= 0.0:
            width = imgui.get_content_region_avail().x
        pos = imgui.get_cursor_screen_pos()
        imgui.dummy((width, height))
        draw_list = imgui.get_window_draw_list()
        rounding = height * 0.5
        draw_list.add_rect_filled(
            (pos.x, pos.y), (pos.x + width, pos.y + height),
            self._u32(PROGRESS_TRACK), rounding,
        )
        if fraction is None:
            # Calm indeterminate sweep: a soft segment gliding back and forth.
            t = float(imgui.get_time()) * 0.45 % 1.0
            phase = 2.0 * t if t < 0.5 else 2.0 - 2.0 * t   # 0->1->0
            seg = width * 0.30
            x0 = pos.x + (width - seg) * phase
            draw_list.add_rect_filled(
                (x0, pos.y), (x0 + seg, pos.y + height),
                self._u32(_rgba(ACCENT_HEX, 0.75)), rounding,
            )
            return
        fraction = min(1.0, max(0.0, float(fraction)))
        if fraction > 0.0:
            fill = max(height, width * fraction)   # keep the pill round
            draw_list.add_rect_filled(
                (pos.x, pos.y), (pos.x + fill, pos.y + height),
                self._u32(PROGRESS_FILL), rounding,
            )

    def hint_row(self, pairs, *, color=None) -> None:
        """Muted caption row of ``key  action`` hints: (key, action) pairs."""
        imgui = self.imgui
        first = True
        for key, action in pairs:
            if not first:
                imgui.same_line(0.0, 10.0)
                self.push_font(self.font_body, SIZE_CAPTION)
                self.body_text("·", TEXT_FAINT)
                self.pop_font()
                imgui.same_line(0.0, 10.0)
            first = False
            self.push_font(self.font_mono, SIZE_MONO_SMALL)
            self.body_text(key, color or TEXT_MUTED)
            self.pop_font()
            imgui.same_line(0.0, 6.0)
            self.push_font(self.font_body, SIZE_CAPTION)
            self.body_text(action, TEXT_FAINT)
            self.pop_font()
