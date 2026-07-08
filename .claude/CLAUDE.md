# Repository guide for Claude Code

You are maintaining this repository.

## Always read before editing any file

The figure-generation specs live in `paper_figures/`. Read all of these
before touching any script:

- `paper_figures/PROJECT_SPEC.md`
- `paper_figures/FIGURE_LIST.md`
- `paper_figures/DATA_SPEC.md`
- `paper_figures/CODE_RULES.md`
- `paper_figures/PAPER_STORY.md`
- `paper_figures/FILE_DEPENDENCIES.md`

## Working rules

- Never modify an existing file unless requested.
- Prefer editing over rewriting.
- Never change `common_style.py` or `synthetic_city.py` unless requested.
- Every figure script must run independently (`python figureNN_*.py`).
- Always run the generated script.
- Fix runtime errors before stopping.
- Never fabricate experimental results. Load results from disk.
- Use placeholder data only when explicitly marked as placeholder
  (stamp `PLACEHOLDER — synthetic data` on the figure).
- Keep IEEE publication quality: 600 DPI, no seaborn, one figure per file,
  no subplots unless requested.
- Reuse helpers from `common_plot.py`; never duplicate plotting code.
- Keep figures faithful to the narrative in `PAPER_STORY.md`.
