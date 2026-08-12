"""Development server for web/ with caching disabled.

Browsers heuristically cache modules served by plain http.server, so after
an edit a hard reload is needed — and a mixed old/new ES-module graph fails
with confusing missing-export errors. This serves every response with
Cache-Control: no-store, so a plain reload always gets the working tree.

Usage: python3 tools/dev_server.py [port]   (default 8765)
"""
import http.server
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    http.server.ThreadingHTTPServer.address_family
    server = http.server.ThreadingHTTPServer(
        ("", port),
        lambda *a, **kw: NoCacheHandler(*a, directory=str(WEB), **kw))
    print(f"serving {WEB} on http://localhost:{port}/soar/ (no-store)")
    server.serve_forever()


if __name__ == "__main__":
    main()
