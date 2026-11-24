# Tamweel risk desk (website branch)

A website built on the Day 5 outputs of the Tamweel project. It does **not** train or tune anything. It reads the notebook's output files and reproduces their results in the browser.

- **Applicant**: a form generated from `data/data_contract.json`. The request is scored with `final_model/model.json` and joins the challenge batch. The applicant sees only the status.
- **Lender desk** (Tamweel, a fictional lender): the 2,500 challenge requests, ranked, with a 6-step trace for each decision.
- **Model evidence**: 8 panels read from the Day 5 CSV/JSON files, each with a graph/table toggle and the source file named.
- **Files**: every file the site reads, its sha256, and a live check in your browser.

## How the site stays tied to the notebook

| Check | Where |
|---|---|
| `model.json` sha256 equals `model_manifest.json` | build script + browser |
| All 2,500 probabilities equal `submission.csv` (max diff ≤ 1e-12) | build script + browser + `tests/parity_test.js` |
| All 2,500 decisions (300 flags) equal `submission.csv` | build script + browser + test |
| Calibrated threshold = calibration(raw OOF threshold) | build script + browser |
| OOF operating point (TP 84, FP 161) equals `final_metrics.json` | test |
| Every copied file matches `files_manifest.json` | browser |

The build stops and writes nothing if any check fails.

## Rebuild after re-running Day 5

```bash
python scripts/build_site.py --bundle /content/tamweel --site docs
node tests/parity_test.js
```

`--bundle` is the project folder that contains `artifacts/`, `data/`, `submission/` and `reports/`.

## Folders

```
docs/            the website (GitHub Pages serves this folder)
  data/          byte-for-byte copies of the Day 5 files + files_manifest.json
  core.js        scoring and batch policy (same arithmetic as day5_final.py)
  app.js         pages, trace, charts
scripts/build_site.py
tests/parity_test.js
```

Educational project on fictional, synthetic data. A flag means "send for review", not "reject". Not for real credit decisions.
