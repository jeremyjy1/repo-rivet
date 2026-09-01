# IncidentLens

IncidentLens is a small command-line tool that reads pipe-delimited incident logs and prints an
operations summary.

Input rows use this shape:

```text
timestamp|service|level|duration_ms|message
```

Run the current Markdown report with:

```bash
python -m incidentlens.cli fixtures/incidents.log
```

Run the test suite with:

```bash
python -m pytest -q
```
