# BTP Report — build and edit notes

LaTeX source for the final evaluation report, built on the supplied
`MTP_Report_Template_2022` class files (`pkmthesis.cls`, `extra_functions.sty`,
`pagesetup_pkm_middleside.tex` are copied unmodified from the template).

## Building

There is no LaTeX toolchain on this machine, so **this source has never been
compiled.** It passes structural validation only. Build it on Overleaf:

1. Zip this whole `btp_report/` directory and upload to Overleaf.
2. Set the main document to `main.tex`.
3. Set the compiler to **pdfLaTeX**.
4. Compile, then compile again (twice more if the table of contents,
   `minitoc`, or citations look wrong — this class needs several passes).

Locally, if you install MiKTeX or TeX Live:

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Before pushing to Overleaf, run the structural check:

```
python validate.py
```

It verifies brace and environment balance, that every `\ref` has a `\label`,
that every referenced figure file exists, and that every `\cite` key is in
`references.bib`. It currently reports zero errors. It is **not** a substitute
for compiling.

## Things you must edit before submission

| Where | What |
|---|---|
| `FrontPages/title_page.tex` | Supervisor name (`Dr. XX`), and confirm the author name and month. |
| `FrontPages/Certificate.tex` | Supervisor name and the project date range. |
| `FrontPages/Acknowledgement.tex` | Supervisor name; adjust the wording to your own. |

The base paper is cited correctly as `paperA` (Dong, Jiang and Peng, *IEEE
Transactions on Industrial Electronics*, vol. 72, no. 8, pp. 8463-8471, 2025,
doi 10.1109/TIE.2024.3525117), taken from the PDF itself. If you also cite the
second base paper (the 3D UAV-ISAC work) in the text, add it to `references.bib`
and cite it in Chapter 1 or 2.

## Page count

Target was 35–45 pages. The estimate is **42–48**, from a word count of roughly
9,400 in the chapters plus 6 figures, 8 tables, 12 equations and 1 algorithm,
over the template text block of 16 cm x 23.5 cm at 12 pt double spacing.

The estimate is uncertain because the source has not been compiled. If the
actual count falls outside the range, the dials in order of effect are:

- **Line spacing.** `main.tex` sets `\doublespacing` (from the template). Changing
  it to `\onehalfspacing` removes roughly a quarter of the pages. Largest single
  lever; check whether your department mandates double spacing first.
- **Figures.** Six are included. Dropping `figure10_altitude_distribution.pdf`
  from Chapter 6 or `figure07_trajectories_3d.pdf` costs least, since both
  support points made in the text rather than carrying a result.
- **Sections that compress without losing an argument.** Chapter 5 sections
  "Environment and Model Implementation" and "Experimental Campaign";
  Chapter 6 "Behavioural Interpretation".
- **If it comes out short**, Chapter 5 has the most room to expand: each of the
  discarded designs can be described in more detail.

Seven figures are available in `figures/`; `figure09_rate_cdf.pdf` is present but
not currently included, and can be added back to Chapter 6 if you need length.

## Where the numbers come from

Every figure in the report is read from the project results, not transcribed.
The source of record is `paper_figures/results_data/` in the parent repository,
and `labels.json` there records which procedure produced each curve. All
reported results are from the 20-seed campaign (city seeds 42–61) at a common
energy budget of 82.65 kJ.

If results are regenerated, refresh the figures with:

```
python -m atom_3d.experiments.export_figure_data --budget-frac-of-2d 0.65 --city-seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61 --skip-pareto --hover-cache results_data_cache/hovers.pkl
```

then re-run the figure scripts and copy the PDFs into `figures/`. The numeric
values in Chapter 6 tables would then need updating by hand — they are typed
into the LaTeX, not generated.

## Structure

```
main.tex                 document, front matter, chapter includes
references.bib           22 entries, IEEE style via IEEEtran
validate.py              structural checker
FrontPages/              title, declaration, acknowledgement, abstract, acronyms
chapters/1_introduction.tex        Introduction and motivation
chapters/2_literature_survey.tex   Literature survey and gap
chapters/3_objectives.tex          Objectives and deliverables
chapters/4_methodology.tex         System model and proposed method
chapters/5_tasks_completed.tex     Work carried out, including discarded designs
chapters/6_results.tex             Results, statistics and ablation
chapters/7_conclusion.tex          Conclusion, limitations, future scope
figures/                 figure PDFs
```
