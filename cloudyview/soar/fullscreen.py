"""Fullscreen monitor selection helpers for GLFW-backed soar windows."""

from __future__ import annotations


def _safe_call(func, *args):
    try:
        return func(*args)
    except Exception:
        return None


def _valid_monitor(handle):
    """Normalize GLFW monitor handles: NULL ctypes pointers are falsy, not None."""
    try:
        return handle if handle else None
    except Exception:
        return None


def video_mode_fields(mode):
    """(width, height, refresh_rate) from a GLFWvidmode across pyGLFW layouts.

    pyGLFW exposes dimensions as mode.size.width/.height; some fakes/tests
    and older bindings use flat mode.width/.height.
    """
    size = getattr(mode, "size", None)
    if size is not None and hasattr(size, "width"):
        return int(size.width), int(size.height), int(getattr(mode, "refresh_rate", 0))
    return int(mode.width), int(mode.height), int(getattr(mode, "refresh_rate", 0))


def safe_windowed_bounds(glfw, window) -> tuple[int, int, int, int]:
    """Return restorable window bounds even when position APIs are unavailable."""
    size = _safe_call(glfw.get_window_size, window) or (1280, 720)
    pos = _safe_call(glfw.get_window_pos, window) or (100, 100)
    return (int(pos[0]), int(pos[1]), int(size[0]), int(size[1]))


def choose_fullscreen_monitor(glfw, window):
    """Choose a monitor without ever asking GLFW for a NULL video mode.

    Wayland can make window positions unusable, which breaks overlap-based
    monitor detection. In that case, or when detection yields no valid monitor,
    the primary monitor is used.
    """
    current = _valid_monitor(_safe_call(glfw.get_window_monitor, window))
    if current is not None:
        return current

    monitors = [m for m in (_safe_call(glfw.get_monitors) or [])
                if _valid_monitor(m) is not None]
    if not monitors:
        return _valid_monitor(_safe_call(glfw.get_primary_monitor))

    pos = _safe_call(glfw.get_window_pos, window)
    size = _safe_call(glfw.get_window_size, window)
    if pos is None or size is None:
        return _valid_monitor(_safe_call(glfw.get_primary_monitor))

    wx, wy = pos
    ww, wh = size
    wcx, wcy = wx + ww * 0.5, wy + wh * 0.5

    best = None
    best_overlap = -1
    best_distance = float("inf")
    for candidate in monitors:
        if candidate is None:
            continue
        mode = _safe_call(glfw.get_video_mode, candidate)
        if mode is None:
            continue
        monitor_pos = _safe_call(glfw.get_monitor_pos, candidate)
        if monitor_pos is None:
            continue
        mx, my = monitor_pos
        mw, mh, _ = video_mode_fields(mode)
        overlap_w = max(0, min(wx + ww, mx + mw) - max(wx, mx))
        overlap_h = max(0, min(wy + wh, my + mh) - max(wy, my))
        overlap = overlap_w * overlap_h
        mcx, mcy = mx + mw * 0.5, my + mh * 0.5
        distance = (wcx - mcx) ** 2 + (wcy - mcy) ** 2
        if overlap > best_overlap or (
            overlap == best_overlap and distance < best_distance
        ):
            best = candidate
            best_overlap = overlap
            best_distance = distance

    if best is not None:
        return best
    return _valid_monitor(_safe_call(glfw.get_primary_monitor))


def fullscreen_video_mode(glfw, monitor):
    """Return a non-null ``(monitor, mode)`` pair or raise a clear error."""
    monitor = _valid_monitor(monitor)
    if monitor is None:
        raise RuntimeError("Cannot enter fullscreen: GLFW found no monitors.")
    mode = _safe_call(glfw.get_video_mode, monitor)
    if mode is None:
        primary = _valid_monitor(_safe_call(glfw.get_primary_monitor))
        if primary is not None and primary is not monitor:
            primary_mode = _safe_call(glfw.get_video_mode, primary)
            if primary_mode is not None:
                return primary, primary_mode
        raise RuntimeError("Cannot enter fullscreen: no video mode found.")
    return monitor, mode
