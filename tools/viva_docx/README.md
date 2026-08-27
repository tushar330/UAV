# Viva study-guide DOCX builder

Renders `VIVA_STUDY_GUIDE.md` (repo root) into `BTP_Viva_Study_Guide.docx`
with all eight report figures embedded at their explanations.

## Build

```
pip install pillow
npm install --prefix tools/viva_docx

python tools/viva_docx/prepare_figures.py   # downsamples 600-DPI PNGs -> .figcache/
node tools/viva_docx/build_docx.js          # writes BTP_Viva_Study_Guide.docx
```

`prepare_figures.py` reads from `paper_figures/results/`. If a figure is
missing, regenerate it with its `paper_figures/figureNN_*.py` script first.

## Notes

- The markdown is the source of truth; edit it, not the `.docx`.
- The table of contents is written as a Word TOC field. Word populates the page
  numbers on open — right-click > Update Field if they show as blank.
- `build_docx.js` converts inline LaTeX to Unicode (`\sqrt{...}` -> `√(...)`,
  `\underbrace{X}_{lbl}` -> `X [lbl]`). It is deliberately narrow: it covers the
  commands this guide actually uses, not LaTeX in general.
- Figure numbers in the report are per-chapter and do **not** match the
  `figureNN_` script numbering. The mapping lives in `prepare_figures.py`.
