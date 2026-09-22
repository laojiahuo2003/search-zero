"""
Local Wikipedia Search HTTP Server.
Run this locally (with VPN for Wikipedia access).
Cloud eval script connects via SSH tunnel to get real wiki search results.
"""
import json
import hashlib
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    import wikipedia
    from wikipedia.exceptions import DisambiguationError, PageError
    HAS_WIKI = True
except ImportError:
    HAS_WIKI = False
    print("[FATAL] pip install wikipedia first!")


class WikiSearchHandler(BaseHTTPRequestHandler):
    """Handle /search?q=QUERY&top_k=3&sentences=3 requests."""

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            top_k = int(params.get("top_k", ["3"])[0])
            sentences = int(params.get("sentences", ["3"])[0])

            if not query:
                self._json_response({"error": "Missing q parameter"}, 400)
                return

            result = self._wiki_search(query, top_k, sentences)
            self._json_response(result)
        elif parsed.path == "/health":
            self._json_response({"status": "ok"})
        else:
            self._json_response({"error": "Use /search?q=QUERY"}, 404)

    def _wiki_search(self, query, top_k, sentences):
        if not HAS_WIKI:
            return {"error": "Wikipedia package not installed", "results": []}

        try:
            titles = wikipedia.search(query, results=top_k)
            if not titles:
                return {"query": query, "results": []}

            results = []
            for i, title in enumerate(titles[:top_k], 1):
                try:
                    summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
                    page = wikipedia.page(title, auto_suggest=False)
                    results.append({
                        "rank": i,
                        "title": title,
                        "summary": summary,
                        "url": page.url,
                    })
                except DisambiguationError as e:
                    for opt in e.options[:3]:
                        try:
                            summary = wikipedia.summary(opt, sentences=sentences, auto_suggest=False)
                            page = wikipedia.page(opt, auto_suggest=False)
                            results.append({"rank": i, "title": opt, "summary": summary, "url": page.url})
                            break
                        except Exception:
                            continue
                    else:
                        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        results.append({"rank": i, "title": title, "summary": "(disambiguation)", "url": url})
                except PageError:
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    results.append({"rank": i, "title": title, "summary": "(page not found)", "url": url})
                except Exception as e:
                    results.append({"rank": i, "title": title, "summary": f"(error: {e})", "url": ""})

            return {"query": query, "results": results}

        except Exception as e:
            return {"query": query, "error": str(e), "results": []}

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")


def main():
    host = "127.0.0.1"
    port = 18080
    print(f"Wiki Search Server: http://{host}:{port}/search?q=QUERY")
    print("Cloud connects via: ssh -R 18080:127.0.0.1:18080 root@CLOUD_IP -p PORT")
    server = HTTPServer((host, port), WikiSearchHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
