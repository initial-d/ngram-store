# Technical report

**Deduplicate, Don't Compress: Storing and Serving Learned N-gram Memory**
([PDF](ngramstore-report.pdf))

When are distinct n-gram rows worth storing, what redundancy does training leave
in the table, and how should a frozen table be served? Every number in the report
comes from a directory under [`evidence/`](../../evidence/README.md).

Build:

```bash
python docs/report/make_figures.py          # needs matplotlib; reads evidence/
cd docs/report && latexmk -pdf main.tex     # needs a TeX distribution with newtx
```
