"""Local HTTP Server Launcher for Project P.E.T.E.R. Interactive Research Dashboard.

Usage:
    python serve_dashboard.py [--port 8000]
"""

import argparse
import http.server
import socketserver
import sys
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8000

def run_server(port: int = DEFAULT_PORT) -> None:
    project_root = Path(__file__).resolve().parent
    dashboard_dir = project_root / "dashboard"
    
    if not (dashboard_dir / "index.html").is_file():
        print(f"Error: Dashboard HTML not found at {dashboard_dir / 'index.html'}")
        sys.exit(1)

    class CustomHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(dashboard_dir), **kwargs)

    with socketserver.TCPServer(("", port), CustomHandler) as httpd:
        url = f"http://localhost:{port}/"
        print("=" * 70)
        print("   PROJECT P.E.T.E.R. INTERACTIVE RESEARCH DASHBOARD SERVER")
        print("=" * 70)
        print(f" * Dashboard serving locally at: {url}")
        print(f" * Serving directory: {dashboard_dir}")
        print(" * Press Ctrl+C to terminate.")
        print("=" * 70)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer terminated gracefully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve Project P.E.T.E.R. Research Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8000)")
    args = parser.parse_args()
    run_server(args.port)
