"""Downsample the 600-DPI paper figures for embedding in the viva study guide.

The publication PNGs in paper_figures/results/ are ~4000-7000 px wide, which would
push the .docx past 15 MB. 1800 px is still well above what a printed page or a
projector resolves, so nothing visible is lost.

Run before tools/viva_docx/build_docx.js:
    python tools/viva_docx/prepare_figures.py
"""
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "paper_figures", "results")
DST = os.path.join(HERE, ".figcache")

TARGET_WIDTH = 1800

# Report figure number -> generator output. The report renumbers figures per
# chapter, so these do not match the figureNN_ script numbering.
FIGURES = [
    ("fig1_1", "figure01_environment.png"),          # Fig 1.1  deployment
    ("fig3_1", "figure02_method_overview.png"),      # Fig 3.1  framework overview
    ("fig3_2", "figure03_altitude_comparison.png"),  # Fig 3.2  altitude vs progress
    ("fig4_1", "figure05_qos_comparison.png"),       # Fig 4.1  per-class satisfaction
    ("fig4_2", "figure09_rate_cdf.png"),             # Fig 4.2  high-priority rate CDF
    ("fig4_3", "figure06_energy_comparison.png"),    # Fig 4.3  energy breakdown
    ("fig4_4", "figure10_altitude_distribution.png"),# Fig 4.4  hover-altitude density
    ("fig4_5", "figure07_trajectories_3d.png"),      # Fig 4.5  3D trajectories
]


def main():
    os.makedirs(DST, exist_ok=True)
    manifest = {}
    for key, name in FIGURES:
        src = os.path.join(SRC, name)
        if not os.path.exists(src):
            raise SystemExit(
                f"missing {src}\nRegenerate it with: python paper_figures/{name[:8]}_*.py"
            )
        im = Image.open(src).convert("RGB")
        w, h = im.size
        im = im.resize((TARGET_WIDTH, max(1, round(h * TARGET_WIDTH / w))), Image.LANCZOS)
        out = os.path.join(DST, key + ".png")
        im.save(out, optimize=True)
        # aspect ratio is stored so the builder can size the image without re-opening it
        manifest[key] = {"file": key + ".png", "ar": h / w, "orig": name}
        print(f"{key:7s} {name:38s} {w}x{h} -> {os.path.getsize(out) // 1024} KB")

    with open(os.path.join(DST, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nwrote {len(manifest)} figures to {DST}")


if __name__ == "__main__":
    main()
