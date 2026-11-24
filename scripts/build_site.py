"""Build the Tamweel website data folder from the Day 5 project bundle.

Nothing is re-modelled here. The script:
  1. verifies final_model/model.json against model_manifest.json (sha256),
  2. re-computes every challenge probability from model.json and requires it to
     match submission/submission.csv (max abs diff <= 1e-12),
  3. re-applies the frozen batch policy and requires the same 0/1 decisions,
  4. copies the source files byte-for-byte into docs/data/ and writes
     docs/data/files_manifest.json (path, bytes, sha256, which page uses it).
If any check fails, it stops and writes nothing.

Usage:  python scripts/build_site.py --bundle /content/tamweel --site docs
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

FILES = [  # (path inside the bundle, what uses it on the site)
    ("artifacts/final_model/model.json", "Scoring: medians, mean, scale, coef, intercept, sigmoid calibration"),
    ("artifacts/final_model/model_manifest.json", "Integrity: expected sha256 of model.json"),
    ("artifacts/final_policy.json", "Decision: calibrated threshold, 12% capacity, batch audit"),
    ("artifacts/final_metrics.json", "Evidence: gate decision, OOF operating point, calibration diagnostics"),
    ("data/data_contract.json", "Applicant form: features, ranges, nullable fields, leakage fields"),
    ("data/tamweel_challenge.csv", "Lender desk: the 2,500-request batch"),
    ("submission/submission.csv", "Integrity: probabilities and decisions the site must reproduce"),
    ("artifacts/ensemble_comparison.csv", "Evidence: candidate comparison"),
    ("artifacts/day5_fold_scores.csv", "Evidence: AP per fold"),
    ("artifacts/day5_threshold_sweep.csv", "Evidence: loss vs threshold"),
    ("artifacts/day5_period_capacity.csv", "Evidence: capacity per period"),
    ("artifacts/day5_cost_sensitivity.csv", "Evidence: FN cost 8 / 10 / 12"),
    ("artifacts/day5_region_audit.csv", "Evidence: false positive rate and recall by region"),
    ("artifacts/day5_calibration_fit_bins.csv", "Evidence: reliability bins, raw vs sigmoid"),
    ("artifacts/day5_oof_predictions.csv", "Evidence: ROC and precision-recall from labelled OOF rows"),
    ("reports/MODEL_CARD.md", "Files page: model card download"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fail(message: str) -> None:
    sys.exit(f"STOP: {message}. Nothing was written.")


def check(bundle: Path) -> dict:
    for rel, _ in FILES:
        if not (bundle / rel).is_file():
            fail(f"missing {rel}")
    model_path = bundle / "artifacts/final_model/model.json"
    manifest = json.loads((bundle / "artifacts/final_model/model_manifest.json").read_text())
    if sha256(model_path) != manifest.get("model.json"):
        fail("model.json does not match model_manifest.json")
    model = json.loads(model_path.read_text())
    if model["chosen"] != "Logistic" or set(model["models"]) != {"Logistic"}:
        fail("the site scorer supports the single Logistic model only")
    spec, cal = model["models"]["Logistic"], model["calibration"]

    challenge = pd.read_csv(bundle / "data/tamweel_challenge.csv", float_precision="round_trip")
    submission = pd.read_csv(bundle / "submission/submission.csv", float_precision="round_trip")
    X = challenge[model["features"]].to_numpy(float)
    X = np.where(np.isnan(X), np.asarray(spec["medians"]), X)
    z = ((X - np.asarray(spec["mean"])) / np.asarray(spec["scale"])) @ np.asarray(spec["coef"]) + spec["intercept"]
    raw = 1 / (1 + np.exp(-z))
    prob = 1 / (1 + np.exp(cal["a"] * raw + cal["b"]))  # expit(-(a*raw+b))
    expected = submission.set_index("application_id").loc[challenge.application_id]
    max_diff = float(np.abs(prob - expected.probability.to_numpy()).max())
    if max_diff > 1e-12:
        fail(f"probabilities differ from submission.csv (max {max_diff})")

    policy = json.loads((bundle / "artifacts/final_policy.json").read_text())
    thr, frac = policy["calibrated_threshold"], policy["cost_policy"]["max_flag_fraction"]
    eligible = prob >= thr
    cap = math.floor(len(prob) * frac)
    decision = eligible.copy()
    if eligible.sum() > cap:
        boundary = np.sort(prob[eligible])[::-1][cap - 1]
        decision = eligible & (prob >= boundary)
        if decision.sum() > cap:
            decision = eligible & (prob > boundary)
    if not (decision.astype(int) == expected.decision.to_numpy()).all():
        fail("batch decisions differ from submission.csv")
    transported = 1 / (1 + math.exp(cal["a"] * policy["raw_oof_threshold"] + cal["b"]))
    if abs(transported - thr) > 1e-12:
        fail("calibrated threshold is not the transported raw threshold")
    return {"rows": int(len(prob)), "max_abs_probability_diff": max_diff,
            "flagged": int(decision.sum()), "eligible": int(eligible.sum()), "capacity": cap,
            "decisions_match": True, "model_sha256_matches_manifest": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--site", required=True, type=Path)
    args = ap.parse_args()
    checks = check(args.bundle)
    data = args.site / "data"
    if data.exists():
        shutil.rmtree(data)
    entries = []
    for rel, used_by in FILES:
        target = data / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.bundle / rel, target)
        entries.append({"path": rel, "bytes": target.stat().st_size, "sha256": sha256(target), "used_by": used_by})
    (data / "files_manifest.json").write_text(json.dumps(
        {"source": "Day 5 project bundle", "python_checks": checks, "files": entries}, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"Copied {len(entries)} files into {data}")


if __name__ == "__main__":
    main()
