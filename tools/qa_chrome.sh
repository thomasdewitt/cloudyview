#!/usr/bin/env bash
# The QA Chrome: hardware WebGPU, DevTools on a port, and no way to reach the
# desktop. tools/tint_check.mjs and tools/pinned_still.mjs drive it.
#
#   tools/qa_chrome.sh start [PORT]    # default 9333; refuses a port in use
#   tools/qa_chrome.sh stop  [PORT]
#
# Why headless and cut off: a windowed QA Chrome raised itself over whatever
# was in front each time a check opened a tab (Thomas, mid-game, 2026-11-08:
# "testing in chrome makes it pop up in front of game"), and a fresh profile
# on the session bus reaches the keyring and GNOME. So it runs with no
# DISPLAY, no Wayland socket (a private, empty XDG_RUNTIME_DIR) and its
# session bus pointed at a socket that does not exist: by construction it can
# open no window and reach nothing of the desktop session (the system bus is
# still there; it is not the desktop), while the GPU stays reachable through
# /dev/nvidia*.
#
# The flags that give the hardware adapter headless (checked 2026-11-08 on the
# RTX 5080, Chrome 157): --headless=new --enable-features=Vulkan
# --use-angle=vulkan. The windowed recipe's --ozone-platform=x11 gives NO
# adapter headless, and adding VulkanFromANGLE with --enable-unsafe-webgpu
# crashed the GPU process. Its stills hash byte for byte the same as the
# windowed Chrome's (pinned_still, city/day/desert at high/low/max), and
# soar_cdp.requireHardwareAdapter refuses the run if it ever falls back to
# software.
set -euo pipefail
cmd=${1:-}
PORT=${2:-9333}
# Checked before it reaches the profile and runtime paths or stop's regex.
if ! [[ $PORT =~ ^[0-9]{4,5}$ ]] \
    || (( 10#$PORT < 1024 || 10#$PORT > 65535 )); then
  echo "qa_chrome: PORT must be a number in 1024-65535; got '$PORT'" >&2
  exit 2
fi
QA=${QA_DIR:-$HOME/code-and-data/cloudyview/temp/qa}
PROFILE=$QA/chrome-headless-$PORT
RUNTIME=$QA/xdg-$PORT

running() { curl -sf "http://127.0.0.1:$PORT/json/version" >/dev/null; }

case "$cmd" in
  start)
    if running; then
      echo "qa_chrome: something already answers on :$PORT; not starting" >&2
      exit 1
    fi
    mkdir -p "$PROFILE" "$RUNTIME"
    chmod 700 "$RUNTIME"
    env -u DISPLAY -u WAYLAND_DISPLAY XDG_RUNTIME_DIR="$RUNTIME" \
        DBUS_SESSION_BUS_ADDRESS="unix:path=/nonexistent/qa-chrome-no-bus" \
        nohup /usr/bin/google-chrome --headless=new \
        --user-data-dir="$PROFILE" --remote-debugging-port="$PORT" \
        --no-first-run --no-default-browser-check --password-store=basic \
        --enable-features=Vulkan --use-angle=vulkan \
        --disable-background-timer-throttling \
        about:blank > "$QA/chrome-headless-$PORT.log" 2>&1 &
    for _ in $(seq 1 50); do running && break; sleep 0.2; done
    if ! running; then
      echo "qa_chrome: no DevTools on :$PORT after 10 s" >&2
      exit 1
    fi
    curl -s "http://127.0.0.1:$PORT/json/version" | head -3
    ;;
  stop)
    # By the profile path, which only this Chrome's processes carry (this
    # script's own command line does not).
    pkill -f -- "--user-data-dir=$PROFILE( |\$)" || true
    ;;
  *)
    echo "usage: tools/qa_chrome.sh start|stop [PORT]" >&2
    exit 2
    ;;
esac
