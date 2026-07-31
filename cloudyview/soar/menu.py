"""Pure menu and file-browser state for the soar app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ACTION_PAUSE = "pause"
ACTION_RESUME = "resume"
ACTION_QUIT = "quit"
ACTION_TOGGLE_FULLSCREEN = "toggle_fullscreen"
ACTION_OPEN_FILE = "open_file"
ACTION_REMOVE_NEST = "remove_nest"
ACTION_OPEN_ICE_YES = "open_ice_yes"
ACTION_OPEN_ICE_NO = "open_ice_no"
ACTION_SELECT_GROUP = "select_group"
ACTION_SELECT_BOTH_GROUPS = "select_both_groups"
ACTION_SELECT_UNITS = "select_units"
ACTION_RENDER_MENU = "render_menu"
ACTION_SELECT_BEHOLD_QUALITY = "select_behold_quality"
ACTION_COPY_BEHOLD_COMMAND = "copy_behold_command"
ACTION_MENU_BACK = "menu_back"
ACTION_SCREENSHOT = "screenshot"
ACTION_TOGGLE_PERIODIC = "toggle_periodic"
ACTION_QUALITY_MENU = "quality_menu"
ACTION_SELECT_TIER = "select_tier"
ACTION_CONTROLS_MENU = "controls_menu"
ACTION_SUN_MENU = "sun_menu"
ACTION_SELECT_SUN_PRESET = "select_sun_preset"
ACTION_TRACK_SAVE = "track_save"
ACTION_TRACK_DISCARD = "track_discard"
ACTION_SCREENSHOT_WITH_OVERLAYS = "screenshot_with_overlays"
ACTION_SCREENSHOT_CLOUDS_ONLY = "screenshot_clouds_only"
ACTION_CLOSE_PREVIEW = "close_preview"

MENU_MAIN = "main"
MENU_FILE_BROWSER_LIQUID = "file_browser_liquid"
MENU_OPEN_GROUP_PROMPT = "open_group_prompt"
MENU_OPEN_UNITS_PROMPT = "open_units_prompt"
MENU_OPEN_ICE_PROMPT = "open_ice_prompt"
MENU_FILE_BROWSER_ICE = "file_browser_ice"
MENU_RENDER_QUALITY = "render_quality"
MENU_QUALITY = "quality"
MENU_CONTROLS = "controls"
MENU_TRACK_SAVE = "track_save"
MENU_SUN = "sun"
MENU_SCREENSHOT = "screenshot"
MENU_SCREENSHOT_PREVIEW = "screenshot_preview"
MENU_ERROR = "error"

BEHOLD_QUALITIES_BY_KEY = {
    "1": "min",
    "2": "low",
    "3": "medium",
    "4": "high",
    "5": "max",   # pre-2026-07-07 'high': 1200x800, 2048 spp — overnight tier
}

QUALITY_TIERS_BY_KEY = {
    "1": "high",
    "2": "medium",
    "3": "low",
    "4": "potato",
}

# Group picker: number keys mirror the on-screen list order.
GROUP_INDEX_BY_KEY = {str(n): n - 1 for n in range(1, 10)}

# Time-of-day presets: (azimuth, elevation) in met degrees. Elevation is
# what drives the look — the spectral sun/sky package fades with air mass —
# so these are chosen by solar elevation, with an azimuth that puts the sun
# where it belongs at that time of day in the northern hemisphere. Sunset
# stays a hair above the horizon: a periodic domain's light march exits
# through the domain top and needs the sun above it.
SUN_PRESETS = {
    "midday": (180.0, 75.0),
    "golden hour": (255.0, 12.0),
    "sunset": (270.0, 0.5),
}
SUN_PRESET_BY_KEY = {
    str(index + 1): name for index, name in enumerate(SUN_PRESETS)
}

# The slider's lower bound on solar elevation, for the same reason.
MIN_SUN_ELEVATION_DEG = 0.5

CONDENSATE_UNITS_BY_KEY = {
    "g": "g/kg",
    "k": "kg/kg",
}


def _normalized_key(key: str) -> str:
    return key.lower() if len(key) == 1 else key


@dataclass(frozen=True)
class MenuTransition:
    action: str | None
    next_state: str | None = None
    quality: str | None = None
    tier: str | None = None
    group_index: int | None = None
    units: str | None = None
    sun_preset: str | None = None


def menu_transition(
    paused: bool, menu_state: str, key: str
) -> MenuTransition:
    """Pure key state machine for flight, pause menu, and submenus."""
    normalized = _normalized_key(key)
    if not paused:
        if key == "Escape":
            return MenuTransition(ACTION_PAUSE, MENU_MAIN)
        if key == "F12":
            # Pauses into the screenshot prompt rather than shooting
            # immediately: what belongs in a saved frame (the bird and the
            # location map, or clouds alone) is a per-shot decision.
            return MenuTransition(ACTION_SCREENSHOT, MENU_SCREENSHOT)
        if key in ("F1", "?"):
            # Pause straight into the controls reference.
            return MenuTransition(ACTION_PAUSE, MENU_CONTROLS)
        if normalized == "f":
            return MenuTransition(ACTION_TOGGLE_FULLSCREEN)
        return MenuTransition(None)

    if menu_state == MENU_MAIN:
        if key == "Escape" or normalized == "r":
            return MenuTransition(ACTION_RESUME, MENU_MAIN)
        if normalized == "q":
            return MenuTransition(ACTION_QUIT, MENU_MAIN)
        if normalized == "f":
            return MenuTransition(ACTION_TOGGLE_FULLSCREEN, MENU_MAIN)
        if normalized == "c":
            return MenuTransition(ACTION_CONTROLS_MENU, MENU_CONTROLS)
        if normalized == "o":
            return MenuTransition(ACTION_OPEN_FILE, MENU_FILE_BROWSER_LIQUID)
        if normalized == "n":
            # Drop a loaded nest. There is no "add" counterpart: a nest comes
            # from the file it lives in (the group picker's "Use both,
            # nested"), or from --nest at launch. The app ignores this when
            # its renderer has no nest.
            return MenuTransition(ACTION_REMOVE_NEST, MENU_MAIN)
        if normalized == "g":
            return MenuTransition(ACTION_RENDER_MENU, MENU_RENDER_QUALITY)
        if normalized == "s":
            return MenuTransition(ACTION_QUALITY_MENU, MENU_QUALITY)
        if normalized == "t":
            return MenuTransition(ACTION_SUN_MENU, MENU_SUN)
        if normalized == "p":
            return MenuTransition(ACTION_TOGGLE_PERIODIC, MENU_MAIN)
        return MenuTransition(None, MENU_MAIN)

    if menu_state == MENU_FILE_BROWSER_LIQUID:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        return MenuTransition(None, MENU_FILE_BROWSER_LIQUID)

    if menu_state == MENU_OPEN_GROUP_PROMPT:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        if normalized == "b":
            # Only offered when the app found a nestable pair; it ignores
            # the action otherwise.
            return MenuTransition(
                ACTION_SELECT_BOTH_GROUPS, MENU_OPEN_GROUP_PROMPT
            )
        group_index = GROUP_INDEX_BY_KEY.get(normalized)
        if group_index is not None:
            # The app rejects an index past the end of its own list.
            return MenuTransition(
                ACTION_SELECT_GROUP, MENU_OPEN_GROUP_PROMPT,
                group_index=group_index,
            )
        return MenuTransition(None, MENU_OPEN_GROUP_PROMPT)

    if menu_state == MENU_OPEN_UNITS_PROMPT:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        units = CONDENSATE_UNITS_BY_KEY.get(normalized)
        if units is not None:
            return MenuTransition(
                ACTION_SELECT_UNITS, MENU_OPEN_UNITS_PROMPT, units=units,
            )
        return MenuTransition(None, MENU_OPEN_UNITS_PROMPT)

    if menu_state == MENU_OPEN_ICE_PROMPT:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        if normalized == "y":
            return MenuTransition(ACTION_OPEN_ICE_YES, MENU_FILE_BROWSER_ICE)
        if normalized == "n":
            return MenuTransition(ACTION_OPEN_ICE_NO, MENU_OPEN_ICE_PROMPT)
        return MenuTransition(None, MENU_OPEN_ICE_PROMPT)

    if menu_state == MENU_FILE_BROWSER_ICE:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_OPEN_ICE_PROMPT)
        return MenuTransition(None, MENU_FILE_BROWSER_ICE)

    if menu_state == MENU_RENDER_QUALITY:
        # This menu hands over a command; nothing renders here. The number
        # keys choose which quality the command names.
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        if normalized == "c" or key == "Enter":
            return MenuTransition(
                ACTION_COPY_BEHOLD_COMMAND, MENU_RENDER_QUALITY
            )
        quality = BEHOLD_QUALITIES_BY_KEY.get(normalized)
        if quality is not None:
            return MenuTransition(
                ACTION_SELECT_BEHOLD_QUALITY, MENU_RENDER_QUALITY, quality
            )
        return MenuTransition(None, MENU_RENDER_QUALITY)

    if menu_state == MENU_QUALITY:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        tier = QUALITY_TIERS_BY_KEY.get(normalized)
        if tier is not None:
            return MenuTransition(
                ACTION_SELECT_TIER, MENU_QUALITY, tier=tier
            )
        return MenuTransition(None, MENU_QUALITY)

    if menu_state == MENU_SUN:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        preset = SUN_PRESET_BY_KEY.get(normalized)
        if preset is not None:
            return MenuTransition(
                ACTION_SELECT_SUN_PRESET, MENU_SUN, sun_preset=preset
            )
        return MenuTransition(None, MENU_SUN)

    if menu_state == MENU_CONTROLS:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        return MenuTransition(None, MENU_CONTROLS)

    if menu_state == MENU_TRACK_SAVE:
        # A recording just stopped: explicit save/discard only. R must not
        # resume here (it would silently drop the take).
        if normalized == "s" or key == "Enter":
            return MenuTransition(ACTION_TRACK_SAVE, MENU_MAIN)
        if normalized == "d" or key == "Escape":
            return MenuTransition(ACTION_TRACK_DISCARD, MENU_MAIN)
        return MenuTransition(None, MENU_TRACK_SAVE)

    if menu_state == MENU_SCREENSHOT:
        if normalized == "w" or key == "Enter" or normalized == "1":
            return MenuTransition(
                ACTION_SCREENSHOT_WITH_OVERLAYS, MENU_MAIN
            )
        if normalized == "c" or normalized == "2":
            return MenuTransition(ACTION_SCREENSHOT_CLOUDS_ONLY, MENU_MAIN)
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        return MenuTransition(None, MENU_SCREENSHOT)

    if menu_state == MENU_SCREENSHOT_PREVIEW:
        # Any way out is the same way out; the shot is already on disk.
        if key in ("Escape", "Enter") or normalized in ("c", "r"):
            return MenuTransition(ACTION_CLOSE_PREVIEW, MENU_MAIN)
        return MenuTransition(None, MENU_SCREENSHOT_PREVIEW)

    if menu_state == MENU_ERROR:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        return MenuTransition(None, MENU_ERROR)

    return MenuTransition(None, menu_state)


def control_action_for_key(
    paused: bool, key: str, menu_state: str = MENU_MAIN
) -> str | None:
    """Backward-compatible action-only view of the menu state machine."""
    return menu_transition(paused, menu_state, key).action


@dataclass(frozen=True)
class FileEntry:
    path: Path
    name: str
    is_dir: bool
    size_bytes: int | None = None

    @property
    def display_size(self) -> str:
        if self.is_dir:
            return ""
        return format_file_size(self.size_bytes or 0)


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size for the in-window browser."""
    size = float(max(0, int(size_bytes)))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def list_netcdf_entries(directory: str | Path) -> list[FileEntry]:
    """Return directories and ``*.nc`` files for the in-window browser."""
    root = Path(directory).expanduser()
    entries: list[FileEntry] = []
    for child in root.iterdir():
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            entries.append(
                FileEntry(path=child, name=child.name, is_dir=True)
            )
            continue
        if child.suffix.lower() != ".nc":
            continue
        try:
            size = child.stat().st_size
        except OSError:
            size = 0
        entries.append(
            FileEntry(path=child, name=child.name, is_dir=False, size_bytes=size)
        )

    return sorted(entries, key=lambda e: (not e.is_dir, e.name.lower()))
