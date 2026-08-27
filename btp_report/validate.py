"""Structural validation for the report source. Run: python validate.py

Checks brace balance, environment balance, cross-reference targets, figure
files and citation keys. This is not a substitute for compiling, but it
catches the errors that would otherwise surface as a failed Overleaf build.
"""
import glob
import os
import pathlib
import re

files = sorted(glob.glob("chapters/*.tex")) + sorted(glob.glob("FrontPages/*.tex")) + ["main.tex"]
errors = 0
labels, refs, figs, cites = set(), [], set(), set()

for f in files:
    raw = pathlib.Path(f).read_text(encoding="utf-8")
    txt = re.sub(r"(?<!\\)%.*", "", raw)          # strip comments, keep \%

    if txt.count("{") != txt.count("}"):
        print(f"  BRACE MISMATCH  {f}: {txt.count('{')} open, {txt.count('}')} close")
        errors += 1

    for env in set(re.findall(r"\\begin\{(\w+\*?)\}", txt)):
        b = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", txt))
        e = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", txt))
        if b != e:
            print(f"  ENV MISMATCH    {f}: {env} begin={b} end={e}")
            errors += 1

    labels |= set(re.findall(r"\\label\{([^}]+)\}", txt))
    refs += [(f, r) for r in re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", txt)]
    figs |= set(re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", txt))
    for group in re.findall(r"\\cite\{([^}]+)\}", txt):
        cites |= {k.strip() for k in group.split(",")}

broken = [(f, r) for f, r in refs if r not in labels]
for f, r in broken:
    print(f"  BROKEN REF      {f}: {r}")
errors += len(broken)

for g in sorted(figs):
    candidates = [g, os.path.join("figures", g), g + ".pdf", os.path.join("figures", g + ".pdf")]
    if not any(os.path.exists(c) for c in candidates):
        print(f"  MISSING FIGURE  {g}")
        errors += 1

bib = set(re.findall(r"^@\w+\{([^,]+),", pathlib.Path("references.bib").read_text(encoding="utf-8"), re.M))
for key in sorted(cites - bib):
    print(f"  CITE NOT IN BIB {key}")
    errors += 1
uncited = sorted(bib - cites)
if uncited:
    print(f"  UNCITED ENTRIES {', '.join(uncited)}  (will not appear in the bibliography)")

print()
print(f"files checked      : {len(files)}")
print(f"labels defined     : {len(labels)}")
print(f"cross-references   : {len(refs)}")
print(f"figures referenced : {len(figs)}")
print(f"citation keys used : {len(cites)} of {len(bib)} bib entries")
print(f"ERRORS             : {errors}")
