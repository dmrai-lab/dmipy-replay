import functools
import http.server
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

DATA = Path(__file__).parent / "data"


class _Range(http.server.SimpleHTTPRequestHandler):
    """A static file server over ``tests/data`` that honours one byte range per request (what the Hub's CDN does)."""

    def do_GET(self):
        path = self.translate_path(self.path)
        data = Path(path).read_bytes()
        rng = self.headers.get("Range")
        if rng:
            a, b = (int(x) for x in rng.split("=")[1].split("-"))
            body, code = data[a:b + 1], 206
        else:
            body, code = data, 200
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Range, directory=str(DATA)))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
