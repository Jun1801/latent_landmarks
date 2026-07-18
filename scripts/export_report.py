#!/usr/bin/env python3
"""Export a Markdown report to a SELF-CONTAINED HTML file.

All referenced images are inlined as base64 data URIs, so the resulting single
.html file can be emailed / shared and opens in any browser with images intact
(no need to send the logs/ folder alongside it).

Usage:
    python scripts/export_report.py --md REPORT.md --out REPORT.html
"""

import argparse
import base64
import os
import re
import mimetypes

import markdown


def inline_images(md_text: str, base_dir: str) -> str:
    """Replace ![alt](relative/path) with ![alt](data:...;base64,...)."""
    def repl(m):
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http://", "https://", "data:")):
            return m.group(0)
        full = os.path.join(base_dir, path)
        if not os.path.isfile(full):
            return m.group(0)
        mime = mimetypes.guess_type(full)[0] or "image/png"
        with open(full, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"![{alt}](data:{mime};base64,{b64})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md_text)


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
max-width:900px;margin:2rem auto;padding:0 1.2rem;line-height:1.55;color:#1a1a1a}
h1,h2{border-bottom:1px solid #e2e2e2;padding-bottom:.3rem}
h1{font-size:1.7rem}h2{font-size:1.3rem;margin-top:2rem}
table{border-collapse:collapse;margin:1rem 0;font-size:.93rem}
th,td{border:1px solid #d0d0d0;padding:.4rem .6rem;text-align:left}
th{background:#f4f4f4}
code{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px;font-size:.9em}
pre{background:#f6f8fa;padding:.8rem 1rem;border-radius:6px;overflow-x:auto}
pre code{background:none;padding:0}
img{max-width:100%;height:auto;display:block;margin:1rem auto;border:1px solid #eee;border-radius:6px}
@media (prefers-color-scheme:dark){
 body{background:#0f1115;color:#e6e6e6}h1,h2{border-color:#333}
 th{background:#1c1f26}th,td{border-color:#333}code,pre{background:#1c1f26}
 img{border-color:#333}}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--md", default="REPORT.md")
    p.add_argument("--out", default="REPORT.html")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.md))
    with open(args.md) as f:
        md_text = f.read()
    md_text = inline_images(md_text, base_dir)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>L3P Report</title><style>{_CSS}</style></head>"
            f"<body>{body}</body></html>")
    with open(args.out, "w") as f:
        f.write(html)
    size_mb = os.path.getsize(args.out) / 1e6
    n_imgs = md_text.count("data:image")
    print(f"Saved {args.out} ({size_mb:.1f} MB, {n_imgs} ảnh nhúng base64)")


if __name__ == "__main__":
    main()
