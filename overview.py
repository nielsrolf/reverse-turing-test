"""Minimal overview server for experiment results."""

import ast
import http.server
import os
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def get_experiments():
    """Scan experiments/ for exp*.py files, extract docstrings, check for reports."""
    experiments = []
    for py_file in sorted(EXPERIMENTS_DIR.glob("exp*.py")):
        # Extract module docstring
        try:
            tree = ast.parse(py_file.read_text())
            docstring = ast.get_docstring(tree) or ""
        except Exception:
            docstring = ""

        # Check for report (different experiments use different filenames)
        stem = py_file.stem
        report_path = None
        for report_name in ("viewer_static.html", "report.html"):
            candidate_path = EXPERIMENTS_DIR / stem / report_name
            if candidate_path.exists():
                report_path = report_name
                break

        experiments.append((stem, docstring, report_path))
    return experiments


def build_html():
    experiments = get_experiments()
    rows = ""
    for stem, docstring, report_path in experiments:
        desc = docstring.replace("\n", "<br>")
        link = (
            f'<a href="/experiments/{stem}/{report_path}">View Report</a>'
            if report_path
            else "<em>No report yet</em>"
        )
        rows += f"<tr><td><b>{stem}</b></td><td>{desc}</td><td>{link}</td></tr>\n"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Experiments Overview</title>
<style>
body {{ font-family: monospace; max-width: 1200px; margin: 40px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
</style></head><body>
<h1>Reverse Turing Test - Experiments</h1>
<table>
<tr><th>Experiment</th><th>Description</th><th>Report</th></tr>
{rows}
</table>
</body></html>"""


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            html = build_html().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(html))
            self.end_headers()
            self.wfile.write(html)
        else:
            # Serve static files (reports, etc.) from project root
            super().do_GET()


if __name__ == "__main__":
    os.chdir(Path(__file__).parent)
    port = 8000
    print(f"Serving at http://localhost:{port}")
    http.server.HTTPServer(("", port), Handler).serve_forever()
