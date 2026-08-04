"""The desktop's own file chooser, over xdg-desktop-portal.

Soar draws its menus with imgui inside the GPU window, and it used to
draw its file browser that way too. A hand-rolled browser is fine for
"the file is right there", but cloud fields live all over the disk and
the system chooser already has search, Recent, bookmarks, and a path
box. This module asks the desktop for that dialog over D-Bus
(``org.freedesktop.portal.FileChooser``), so nothing GTK or Qt has to be
imported into — or frozen alongside — the app.

The portal is asynchronous by design: ``OpenFile`` returns a *request*
object path immediately and the answer arrives later as a ``Response``
signal. `FileChooserRequest.run` waits for that signal, so it belongs on
a background thread (see `jobs.BackgroundJob`); `FileChooserRequest.close`
can be called from the render thread to withdraw the dialog.

No portal, no dialog: `portal_available` returns False and the caller
falls back to the in-window browser. That is a real fallback rather than
a silent one — the app says which chooser it is using and why.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from urllib.parse import unquote, urlparse

PORTAL_BUS_NAME = "org.freedesktop.portal.Desktop"
PORTAL_OBJECT_PATH = "/org/freedesktop/portal/desktop"
FILE_CHOOSER_INTERFACE = "org.freedesktop.portal.FileChooser"
REQUEST_INTERFACE = "org.freedesktop.portal.Request"

# Portal response codes (org.freedesktop.portal.Request::Response).
RESPONSE_SUCCESS = 0
RESPONSE_CANCELLED = 1

# How often the waiting thread looks up to see whether the app withdrew
# the dialog. Idle cost is one deque check per interval.
CLOSE_POLL_SECONDS = 0.2

NETCDF_FILTERS = [
    ("NetCDF (*.nc)", [(0, "*.nc"), (0, "*.nc4"), (0, "*.cdf")]),
    ("All files", [(0, "*")]),
]

_token_lock = Lock()
_token_counter = 0


def _next_handle_token() -> str:
    """A per-request token; the portal builds the request path from it."""
    global _token_counter
    with _token_lock:
        _token_counter += 1
        return f"cloudyview_soar_{_token_counter}"


def _request_path(unique_name: str, token: str) -> str:
    """Where the portal will put this request's object.

    Documented in the portal API: the caller can compute the path before
    the call returns, which is what makes it safe to start listening for
    the Response signal *first*. Without that, a fast answer can arrive
    before the match rule is installed and the wait never ends.
    """
    sender = unique_name.lstrip(":").replace(".", "_")
    return f"{PORTAL_OBJECT_PATH}/request/{sender}/{token}"


def portal_available() -> bool:
    """True when a desktop portal is reachable on the session bus."""
    try:
        from jeepney import message_bus
        from jeepney.io.blocking import open_dbus_connection, Proxy
    except Exception:
        return False

    try:
        connection = open_dbus_connection(bus="SESSION")
    except Exception:
        # No session bus at all: headless, ssh, a bare tty.
        return False
    try:
        bus = Proxy(message_bus, connection)
        if bus.NameHasOwner(PORTAL_BUS_NAME)[0]:
            return True
        # Portals are D-Bus activatable, so "not running yet" is not "absent".
        return PORTAL_BUS_NAME in bus.ListActivatableNames()[0]
    except Exception:
        return False
    finally:
        connection.close()


class FileChooserRequest:
    """One portal Open-File dialog: `run` waits for it, `close` withdraws it."""

    def __init__(
        self,
        title: str,
        *,
        current_folder: str | Path | None = None,
        filters=NETCDF_FILTERS,
        accept_label: str | None = None,
    ):
        self.title = title
        self.current_folder = current_folder
        self.filters = filters
        self.accept_label = accept_label
        self._token = _next_handle_token()
        self._path_lock = Lock()
        self._request_object_path: str | None = None
        self._closed = False

    def run(self) -> Path | None:
        """Show the dialog and block until it is answered.

        Returns the chosen path, or None if the user cancelled (or the
        app withdrew the dialog with `close`). Raises when the portal
        itself fails — an unreachable chooser is worth surfacing, not
        worth papering over with an empty selection.
        """
        from jeepney import DBusAddress, MatchRule, message_bus, new_method_call
        from jeepney.io.blocking import open_dbus_connection, Proxy

        portal = DBusAddress(
            object_path=PORTAL_OBJECT_PATH,
            bus_name=PORTAL_BUS_NAME,
            interface=FILE_CHOOSER_INTERFACE,
        )
        connection = open_dbus_connection(bus="SESSION")
        try:
            expected_path = _request_path(connection.unique_name, self._token)
            with self._path_lock:
                if self._closed:
                    return None
                self._request_object_path = expected_path

            rule = MatchRule(
                type="signal",
                interface=REQUEST_INTERFACE,
                member="Response",
                path=expected_path,
            )
            with connection.filter(rule) as responses:
                Proxy(message_bus, connection).AddMatch(rule)
                reply = connection.send_and_get_reply(
                    new_method_call(
                        portal, "OpenFile", "ssa{sv}",
                        # An empty parent window: GLFW exposes no handle a
                        # Wayland compositor would accept (xdg_foreign), so
                        # the caller drops exclusive fullscreen instead of
                        # parenting the dialog to a window it cannot name.
                        ("", self.title, self._options()),
                    )
                )
                actual_path = reply.body[0]
                if actual_path != expected_path:
                    # Only pre-0.9 portals did this. Listening on the wrong
                    # path would hang forever, so say so instead.
                    raise RuntimeError(
                        "Portal returned an unexpected request path "
                        f"{actual_path!r} (expected {expected_path!r}); "
                        "this needs xdg-desktop-portal 0.9 or newer."
                    )
                signal = self._wait_for_response(connection, responses)
                if signal is None:
                    return None
        finally:
            connection.close()
            with self._path_lock:
                self._request_object_path = None

        response_code, results = signal.body
        if response_code != RESPONSE_SUCCESS:
            return None
        uris = results.get("uris")
        paths = _paths_from_uris(uris[1] if uris else [])
        return paths[0] if paths else None

    def _wait_for_response(self, connection, responses):
        """Wait for the portal's Response signal, or for `close`.

        Withdrawing a request is explicitly documented to end the user
        interaction *without* emitting a Response, so waiting outright
        would park this thread forever on a dialog the app already took
        down. Short polls keep `close` effective; nothing else is racing
        for this connection.
        """
        while True:
            try:
                return connection.recv_until_filtered(
                    responses, timeout=CLOSE_POLL_SECONDS
                )
            except TimeoutError:
                with self._path_lock:
                    if self._closed:
                        return None

    def close(self) -> None:
        """Withdraw the dialog, from any thread. Safe to call twice.

        `run` is parked in a blocking read on its own connection, so this
        opens a second one to send Request.Close(); the portal answers the
        parked read with a cancelled Response.
        """
        with self._path_lock:
            self._closed = True
            path = self._request_object_path
        if path is None:
            return

        from jeepney import DBusAddress, new_method_call
        from jeepney.io.blocking import open_dbus_connection

        request = DBusAddress(
            object_path=path,
            bus_name=PORTAL_BUS_NAME,
            interface=REQUEST_INTERFACE,
        )
        connection = open_dbus_connection(bus="SESSION")
        try:
            connection.send_and_get_reply(new_method_call(request, "Close"))
        except Exception:
            # The dialog answered on its own between the two calls; the
            # Response signal run() is waiting on settles it either way.
            pass
        finally:
            connection.close()

    def _options(self) -> dict:
        options = {
            "handle_token": ("s", self._token),
            "modal": ("b", True),
            "multiple": ("b", False),
            "directory": ("b", False),
        }
        if self.accept_label:
            options["accept_label"] = ("s", self.accept_label)
        if self.filters:
            options["filters"] = ("a(sa(us))", list(self.filters))
        if self.current_folder is not None:
            folder = Path(self.current_folder).expanduser()
            # Portal wants a NUL-terminated byte string, not a str.
            options["current_folder"] = (
                "ay", os.fsencode(str(folder)) + b"\0"
            )
        return options


def _paths_from_uris(uris) -> list:
    """Local filesystem paths from the portal's file:// URIs.

    Anything non-local (a portal can hand back gvfs URIs for remote
    shares) is dropped: netCDF4 opens paths, not URLs.
    """
    paths = []
    for uri in uris:
        parsed = urlparse(str(uri))
        if parsed.scheme != "file":
            continue
        if parsed.netloc not in ("", "localhost"):
            continue
        paths.append(Path(unquote(parsed.path)))
    return paths
