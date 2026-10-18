"""Watch the camera Pi's live view on a laptop: what passive detection sees, with its boxes drawn.

    CAMERA_TOKEN=... python3 camera_view.py [--camera http://<faker's 100.x address>:8765] [--port 8766]

Then open http://127.0.0.1:8766. The camera Pi's /stream needs the X-Camera-Token header, which a browser
can't send, so this relays it from localhost only and adds the header; the token never goes in a URL.
Standard library only.
"""

import argparse
import os
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>LastScene camera</title><style>
body{margin:0;background:#111;color:#ddd;font:14px system-ui;display:flex;flex-direction:column;align-items:center}
img{max-width:100%;height:auto;margin-top:12px;border-radius:6px;background:#222;min-height:240px}
p{margin:8px 16px;color:#999}</style></head><body>
<img src="/stream" alt="camera stream">
<p>Boxes: what the detector saw in its last pass (green at 0.5 or more: raia reads it). Refresh if it stops.</p>
</body></html>"""


def make_handler(camera: str, token: str):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path != "/stream":
                self.send_error(404)
                return
            req = urllib.request.Request(f"{camera}/stream", headers={"X-Camera-Token": token})
            try:
                upstream = urllib.request.urlopen(req, timeout=15)
            except Exception as exc:  # noqa: BLE001
                self.send_error(502, f"camera: {exc}")
                return
            with upstream:
                self.send_response(200)
                self.send_header("Content-Type", upstream.headers.get("Content-Type", "multipart/x-mixed-replace"))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while chunk := upstream.read1(65536):
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default=os.environ.get("CAMERA_URL", "http://100.89.232.47:8765"))
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    token = os.environ.get("CAMERA_TOKEN", "")
    if len(token) < 16:
        print("Set CAMERA_TOKEN (the camera Pi's, 16+ characters).", file=sys.stderr)
        return 2
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.camera.rstrip("/"), token))
    print(f"Camera view of {args.camera}: http://127.0.0.1:{args.port}  (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
