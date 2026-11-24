/* Tamweel core: the same arithmetic as scripts/day5_final.py predict_final and apply_batch_policy.
   Runs in the browser and in Node (for the parity test). No invented numbers live here. */
(function (root) {
  "use strict";
  function parseCSV(text) {
    const rows = []; let row = [], cell = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c; }
      else if (c === '"') q = true;
      else if (c === ",") { row.push(cell); cell = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
      else cell += c;
    }
    if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
    const head = rows.shift();
    return rows.filter(r => r.length === head.length).map(r => {
      const o = {};
      head.forEach((h, j) => { const v = r[j]; o[h] = v === "" ? null : (v === "True" ? true : v === "False" ? false : (isNaN(+v) ? v : +v)); });
      return o;
    });
  }
  const expit = x => 1 / (1 + Math.exp(-x));
  /* model.json -> per-feature trace and calibrated probability */
  function score(model, row) {
    const s = model.models.Logistic, cal = model.calibration;
    const parts = model.features.map((name, i) => {
      const given = row[name], imputed = given === null || given === undefined || Number.isNaN(given);
      const value = imputed ? s.medians[i] : +given;
      const z = (value - s.mean[i]) / s.scale[i];
      return { name, value, imputed, z, coef: s.coef[i], contribution: s.coef[i] * z };
    });
    let logit = s.intercept; for (const p of parts) logit += p.contribution;
    const raw = expit(logit);
    const probability = expit(-(cal.a * raw + cal.b));
    return { parts, intercept: s.intercept, logit, raw, probability };
  }
  /* final_policy: threshold first, then probability-descending cap; equal-score blocks kept or dropped whole */
  function batchPolicy(probs, threshold, fraction) {
    const n = probs.length, cap = Math.floor(n * fraction + 1e-9);
    const eligible = probs.map(p => p >= threshold);
    let decision = eligible.slice(), boundary = null;
    const elig = probs.filter((p, i) => eligible[i]).sort((a, b) => b - a);
    if (elig.length > cap) {
      if (cap === 0) decision = decision.map(() => false);
      else {
        boundary = elig[cap - 1];
        decision = probs.map((p, i) => eligible[i] && p >= boundary);
        if (decision.filter(Boolean).length > cap) decision = probs.map((p, i) => eligible[i] && p > boundary);
      }
    }
    const order = probs.map((p, i) => i).filter(i => eligible[i]).sort((a, b) => probs[b] - probs[a]);
    const rank = new Array(n).fill(null); order.forEach((idx, r) => { rank[idx] = r + 1; });
    return { n, cap, eligible, decision, boundary, rank, eligibleCount: order.length, flagged: decision.filter(Boolean).length };
  }
  /* ROC / PR from labelled rows; AP as sklearn average_precision_score (non-interpolated) */
  function curves(y, p) {
    const idx = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
    const P = y.filter(v => v === 1).length, N = y.length - P;
    let tp = 0, fp = 0, ap = 0, prevR = 0; const roc = [[0, 0]], pr = [];
    for (let k = 0; k < idx.length; k++) {
      const i = idx[k]; if (y[i] === 1) tp++; else fp++;
      if (k + 1 < idx.length && p[idx[k + 1]] === p[i]) continue;
      const rec = tp / P, prec = tp / (tp + fp);
      ap += (rec - prevR) * prec; prevR = rec;
      roc.push([fp / N, rec]); pr.push([rec, prec]);
    }
    let auc = 0; for (let k = 1; k < roc.length; k++) auc += (roc[k][0] - roc[k - 1][0]) * (roc[k][1] + roc[k - 1][1]) / 2;
    return { roc, pr, ap, auc, P, N };
  }
  root.TamweelCore = { parseCSV, score, batchPolicy, curves, expit };
})(typeof globalThis !== "undefined" ? globalThis : this);
