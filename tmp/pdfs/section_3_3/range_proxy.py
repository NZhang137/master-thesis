from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen


BASE = "https://relay.fullyjustified.net"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        request = Request(BASE + self.path, headers={"User-Agent": "Tectonic"})
        if "Range" in self.headers:
            request.add_header("Range", self.headers["Range"])
        with urlopen(request) as response:
            self.send_response(response.status)
            for name in ("Content-Length", "Content-Range", "Accept-Ranges"):
                value = response.headers.get(name)
                if value:
                    self.send_header(name, value)
            self.end_headers()
            while chunk := response.read(1024 * 1024):
                self.wfile.write(chunk)

    def log_message(self, *_args):
        return


ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
