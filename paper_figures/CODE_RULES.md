# CODE RULES

- **Never change `common_style.py`** unless requested.
- **Never modify `synthetic_city.py`** unless requested.
- Every figure must be **independent**.
- Each script must **run independently**
  (`python figureNN_*.py` with no arguments, from any cwd).

---

## Consequences of the rules

- All shared styling (IEEE rcParams, 600 DPI save helper, class/method
  colors, fonts) lives in `common_style.py`. Figure scripts *import* it;
  they never redefine styling inline.
- All shared constants live in `DATA_SPEC.md` / a constants module.
  No magic numbers in figure scripts.
- No figure script imports another figure script.
- Data is **loaded from disk**, never fabricated silently. Missing data →
  labelled placeholder, stamped `PLACEHOLDER — synthetic data`.
- One script → one figure → one `results/figureNN_*.png`. No subplots
  unless the figure list explicitly asks for them.
- Deterministic: seed any placeholder RNG so figures are reproducible.
