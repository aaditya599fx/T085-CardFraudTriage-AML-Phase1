# IEEE report source

`main.tex` uses the IEEEtran conference class and includes the bibliography directly. The generated table rows are bracketed by comments. `figures/` contains all referenced plots.

To refresh tables and figures after a new experiment:

```bash
python scripts/build_report.py
```

Then upload the contents of this folder (or the provided Overleaf ZIP) to Overleaf and compile `main.tex` with pdfLaTeX. Alternatively, from the project root:

```bash
tectonic reports/ieee_report/main.tex
```

The delivered PDF was compiled with Tectonic. If changing results, regenerate and compile the report again; the existing narrative includes numerical findings and must be reviewed for consistency. Presentation charts are editable but do not automatically refresh from rerun CSV files.

After compilation, run `python scripts/package_report.py` from the repository root to update the submission PDF and matching Overleaf ZIP. Run `python scripts/verify_submission.py` after the final full audit and source manifest.
