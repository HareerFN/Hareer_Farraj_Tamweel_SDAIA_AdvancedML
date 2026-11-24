// node tests/parity_test.js  — the browser code (docs/core.js) must reproduce the notebook outputs.
const fs = require("fs"), path = require("path");
require(path.join(__dirname, "../docs/core.js"));
const { parseCSV, score, batchPolicy, curves } = globalThis.TamweelCore;
const D = p => fs.readFileSync(path.join(__dirname, "../docs/data", p), "utf8");
const model = JSON.parse(D("artifacts/final_model/model.json"));
const policy = JSON.parse(D("artifacts/final_policy.json"));
const metrics = JSON.parse(D("artifacts/final_metrics.json"));
const challenge = parseCSV(D("data/tamweel_challenge.csv"));
const sub = new Map(parseCSV(D("submission/submission.csv")).map(r => [r.application_id, r]));
const probs = challenge.map(r => score(model, r).probability);
let maxDiff = 0; challenge.forEach((r, i) => { maxDiff = Math.max(maxDiff, Math.abs(probs[i] - sub.get(r.application_id).probability)); });
const b = batchPolicy(probs, policy.calibrated_threshold, policy.cost_policy.max_flag_fraction);
const decisionsMatch = challenge.every((r, i) => +b.decision[i] === sub.get(r.application_id).decision);
const oof = parseCSV(D("artifacts/day5_oof_predictions.csv"));
const t = metrics.oof_threshold_selection.threshold;
const tp = oof.filter(r => r.Logistic >= t && r.default_within_90d === 1).length, fp = oof.filter(r => r.Logistic >= t && r.default_within_90d === 0).length;
const c = curves(oof.map(r => r.default_within_90d), oof.map(r => r.Logistic));
const out = { rows: probs.length, maxDiff, decisionsMatch, flagged: b.flagged, eligible: b.eligibleCount, cap: b.cap,
  boundaryMatches: Math.abs(b.boundary - policy.batch_audit.boundary_score) < 1e-12, oofTP: tp, oofFP: fp,
  oofMatchesMetrics: tp === metrics.oof_threshold_selection.tp && fp === metrics.oof_threshold_selection.fp, pooledAP: c.ap, pooledAUC: c.auc };
console.log(out);
if (!(maxDiff < 1e-12 && decisionsMatch && out.boundaryMatches && out.oofMatchesMetrics)) { console.error("PARITY FAILED"); process.exit(1); }
console.log("PARITY OK");
