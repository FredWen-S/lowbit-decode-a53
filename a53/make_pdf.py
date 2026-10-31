#!/usr/bin/env python3
"""summary.md → summary.html → summary.pdf(Chrome 无头打印,无加密)。
表格数字不换行、字号收紧,保证 Letter 纸两页内放下且表格不被截断。"""
import os, re, subprocess, markdown
HERE = os.path.dirname(os.path.abspath(__file__))
def prep(md):
    # Python-Markdown 需要列表前空一行、嵌套缩进 4 格
    out, prev = [], ""
    for line in md.split("\n"):
        line = re.sub(r"^  (- )", r"    \1", line)
        is_item = bool(re.match(r"\s*(- |\d+\. )", line))
        if is_item and prev.strip() and not re.match(r"\s*(- |\d+\. )", prev):
            out.append("")
        out.append(line); prev = line
    return "\n".join(out)
body = markdown.markdown(prep(open(f"{HERE}/summary.md").read()), extensions=["tables"])
css = """
@page { size: letter; margin: 11mm 14mm; }
body { font: 8.7pt/1.27 -apple-system, Helvetica, Arial, sans-serif; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4px; } h2 { font-size: 11pt; margin: 8px 0 2px; }
p, li { margin: 1.5px 0; } ul, ol { margin: 2px 0 4px; padding-left: 18px; }
code { font: 8.6pt Menlo, monospace; background: #f2f2f0; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 3px 0 8px; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #ddd; padding: 2px 6px; text-align: right; white-space: nowrap; }
th { border-bottom: 1.5px solid #888; font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
p:has(+ table) { break-after: avoid; } h2 { break-after: avoid; }
img { width: 44%; display: inline-block; vertical-align: top; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Why fewer bits stop paying off on a Cortex-A53</title><style>{css}</style></head><body>{body}</body></html>"
open(f"{HERE}/summary.html", "w").write(html)
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless", "--disable-gpu",
                "--no-pdf-header-footer", f"--print-to-pdf={HERE}/summary.pdf", f"file://{HERE}/summary.html"],
               check=True, capture_output=True)
