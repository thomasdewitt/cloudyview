"""Pure menu and file-browser state for the soar app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ACTION_PAUSE = "pause"
ACTION_RESUME = "resume"
ACTION_QUIT = "quit"
ACTION_TOGGLE_FULLSCREEN = "toggle_fullscreen"
ACTION_OPEN_FILE = "open_file"
ACTION_OPEN_ICE_YES = "open_ice_yes"
ACTION_OPEN_ICE_NO = "open_ice_no"
ACTION_RENDER_MENU = "render_menu"
ACTION_RENDER_BEHOLD = "render_behold"
ACTION_MENU_BACK = "menu_back"
ACTION_SCREENSHOT = "screenshot"

MENU_MAIN = "main"
MENU_FILE_BROWSER_LIQUID = "file_browser_liquid"
MENU_OPEN_ICE_PROMPT = "open_ice_prompt"
MENU_FILE_BROWSER_ICE = "file_browser_ice"
MENU_RENDER_QUALITY = "render_quality"
MENU_ERROR = "error"

BEHOLD_QUALITIES_BY_KEY = {
    "1": "min",
    "2": "low",
    "3": "medium",
    "4": "high",
}


def _normalized_key(key: str) -> str:
    return key.lower() if len(key) == 1 else key


@dataclass(frozen=True)
class MenuTransition:
    action: str | None
    next_state: str | None = None
    quality: str | None = None


def menu_transition(
    paused: bool, menu_state: str, key: str
) -> MenuTransition:
    """Pure key state machine for flight, pause menu, and submenus."""
    normalized = _normalized_key(key)
    if not paused:
        if key == "Escape":
            return MenuTransition(ACTION_PAUSE, MENU_MAIN)
        if key == "F12":
            return MenuTransition(ACTION_SCREENSHOT)
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
        if normalized == "o":
            return MenuTransition(ACTION_OPEN_FILE, MENU_FILE_BROWSER_LIQUID)
        if normalized == "g":
            return MenuTransition(ACTION_RENDER_MENU, MENU_RENDER_QUALITY)
        return MenuTransition(None, MENU_MAIN)

    if menu_state == MENU_FILE_BROWSER_LIQUID:
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        return MenuTransition(None, MENU_FILE_BROWSER_LIQUID)

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
        if key == "Escape":
            return MenuTransition(ACTION_MENU_BACK, MENU_MAIN)
        quality = BEHOLD_QUALITIES_BY_KEY.get(normalized)
        if quality is not None:
            return MenuTransition(
                ACTION_RENDER_BEHOLD, MENU_RENDER_QUALITY, quality
            )
        return MenuTransition(None, MENU_RENDER_QUALITY)

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
