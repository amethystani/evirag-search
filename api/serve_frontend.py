import os, sys
os.chdir(os.path.join(os.path.dirname(__file__), "..", "evirag"))
port = int(os.environ.get("PORT", 3000))
import http.server, socketserver
handler = http.server.SimpleHTTPRequestHandler
with socketserver.TCPServer(("", port), handler) as httpd:
    print(f"Serving evirag/ on port {port}", flush=True)
    httpd.serve_forever()
