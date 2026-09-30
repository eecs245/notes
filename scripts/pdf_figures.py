"""Render saved Plotly output as PNGs in a disposable PDF-only source tree."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import re
from pathlib import Path

import kaleido
import numpy as np
import plotly.offline
from playwright.sync_api import sync_playwright

PLOTLY_MIME = "application/vnd.plotly.v1+json"


def decode_arrays(value):
    """Accept Plotly 6 typed arrays even with the repository's Plotly 5 renderer."""
    if isinstance(value, dict):
        if "bdata" in value and "dtype" in value:
            array = np.frombuffer(base64.b64decode(value["bdata"]), dtype=value["dtype"])
            if "shape" in value:
                array = array.reshape(tuple(map(int, value["shape"].split(","))))
            return array.tolist()
        return {key: decode_arrays(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_arrays(item) for item in value]
    return value


def saved_figure(data):
    if PLOTLY_MIME in data:
        figure = data[PLOTLY_MIME]
        return {key: figure[key] for key in ("data", "layout")}
    markup = "".join(data.get("text/html", []))
    # Some figures live inside escaped iframe srcdoc documents.
    if "srcdoc=" in markup:
        markup = html.unescape(markup)
    if not re.search(r'class=[\"\']plotly-graph-div[\"\']', markup):
        return None
    match = re.search(r"Plotly\.newPlot\s*\(", markup)
    if not match:
        return None
    decoder = json.JSONDecoder()
    rest = markup[match.end():].lstrip()
    args = []
    for _ in range(3):
        value, end = decoder.raw_decode(rest)
        args.append(value)
        rest = rest[end:].lstrip().removeprefix(",").lstrip()
    return {"data": args[1], "layout": args[2]}


def snapshot_notebooks(stage: Path, notebooks: list[Path], cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    plotly_js = plotly.offline.get_plotlyjs()
    renderer_hash = hashlib.sha256(
        (Path(__file__).read_text() + plotly_js).encode()
    ).hexdigest()
    # Kaleido already supplies an offline MathJax bundle in requirements.txt.
    mathjax = Path(kaleido.__file__).parent / "executable/etc/mathjax/MathJax.js"
    if not mathjax.exists():
        raise RuntimeError("The PDF renderer needs Kaleido 0.2.1's bundled MathJax")
    manifest = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1200, "height": 1000}, device_scale_factor=2)
        try:
            for relative in notebooks:
                path = stage / relative
                notebook = json.loads(path.read_text())
                changed = False
                for cell_index, cell in enumerate(notebook.get("cells", [])):
                    for output_index, output in enumerate(cell.get("outputs", [])):
                        figure = saved_figure(output.get("data", {}))
                        if figure is None:
                            continue
                        label = f"{relative}: cell {cell_index}, output {output_index}"
                        digest = hashlib.sha256((renderer_hash + json.dumps(figure, sort_keys=True)).encode()).hexdigest()
                        png = cache / f"{digest}.png"
                        if not png.exists():
                            print("Capturing", label, flush=True)
                            figure = decode_arrays(figure)
                            layout = figure["layout"]
                            width = int(layout.get("width") or 800)
                            height = int(layout.get("height") or 500)
                            page.set_viewport_size({"width": max(1200, width), "height": max(1000, height)})
                            # A local document lets MathJax load its own offline fonts/extensions.
                            shell = cache / "renderer.html"
                            shell.write_text('<html><head><style>body{margin:0;background:white}</style></head><body><div id="plot"></div></body></html>')
                            page.goto(shell.as_uri())
                            page.add_script_tag(url=mathjax.as_uri() + "?config=TeX-AMS-MML_SVG")
                            page.add_script_tag(content=plotly_js)
                            page.evaluate("""async ({figure, width, height}) => {
                                await new Promise(resolve => MathJax.Hub.Queue(resolve));
                                Object.assign(document.getElementById('plot').style,
                                    {width: `${width}px`, height: `${height}px`});
                                await Plotly.newPlot('plot', figure.data,
                                    {...figure.layout, width, height, autosize: false},
                                    {displayModeBar: false, responsive: false, plotGlPixelRatio: 2});
                                await document.fonts.ready;
                                await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                            }""", {"figure": figure, "width": width, "height": height})
                            page.locator("#plot").screenshot(path=str(png), animations="disabled", timeout=60000)
                        output["data"] = {"image/png": base64.b64encode(png.read_bytes()).decode()}
                        output["metadata"] = {}
                        changed = True
                        manifest.append({"source": label, "image": str(png)})
                if changed:
                    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
        finally:
            browser.close()
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest)} interactive figure snapshots", flush=True)
