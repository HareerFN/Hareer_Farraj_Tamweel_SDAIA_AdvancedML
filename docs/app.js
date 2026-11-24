{
"use strict";
const C = globalThis.TamweelCore;
const $ = s => document.querySelector(s);
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const f = (v, d = 3) => (v === null || v === undefined || Number.isNaN(+v)) ? "—" : (+v).toFixed(d);
const pct = (v, d = 1) => (v === null || v === undefined) ? "—" : (100 * v).toFixed(d) + "%";
const int = v => Math.round(v).toLocaleString("en-US");
const src = p => `<a class="src" href="#files">${esc(p)}</a>`;
const S = { files: {}, sel: null, filter: "all", query: "", shown: 60, mine: [] };

async function sha(buf) { try { const h = await crypto.subtle.digest("SHA-256", buf); return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, "0")).join(""); } catch (e) { return null; } }
async function load() {
  const manifest = await (await fetch("data/files_manifest.json")).json();
  S.manifest = manifest;
  await Promise.all(manifest.files.map(async e => {
    const r = await fetch("data/" + e.path); if (!r.ok) throw new Error("Missing " + e.path);
    const buf = await r.arrayBuffer(); const text = new TextDecoder().decode(buf);
    S.files[e.path] = { text, sha: await sha(buf), entry: e };
  }));
  const J = p => JSON.parse(S.files[p].text), T = p => C.parseCSV(S.files[p].text);
  S.model = J("artifacts/final_model/model.json"); S.modelManifest = J("artifacts/final_model/model_manifest.json");
  S.policy = J("artifacts/final_policy.json"); S.metrics = J("artifacts/final_metrics.json"); S.contract = J("data/data_contract.json");
  S.challenge = T("data/tamweel_challenge.csv"); S.sub = new Map(T("submission/submission.csv").map(r => [r.application_id, r]));
  S.ens = T("artifacts/ensemble_comparison.csv"); S.folds = T("artifacts/day5_fold_scores.csv"); S.sweep = T("artifacts/day5_threshold_sweep.csv");
  S.periods = T("artifacts/day5_period_capacity.csv"); S.sens = T("artifacts/day5_cost_sensitivity.csv"); S.regions = T("artifacts/day5_region_audit.csv");
  S.bins = T("artifacts/day5_calibration_fit_bins.csv"); S.oof = T("artifacts/day5_oof_predictions.csv");
  S.thr = S.policy.calibrated_threshold; S.frac = S.policy.cost_policy.max_flag_fraction;
  S.fn = S.policy.cost_policy.false_negative_cost; S.fp = S.policy.cost_policy.false_positive_cost;
  S.base = S.challenge.map(r => ({ id: r.application_id, row: r, who: "file", s: C.score(S.model, r) }));
  verify();
  try { S.mine = JSON.parse(localStorage.getItem("tamweel_requests") || "[]").map(m => ({ ...m, s: C.score(S.model, m.row) })); } catch (e) { S.mine = []; }
  rebatch();
}

function verify() {
  const probs = S.base.map(b => b.s.probability);
  let maxDiff = 0; S.base.forEach(b => { maxDiff = Math.max(maxDiff, Math.abs(b.s.probability - S.sub.get(b.id).probability)); });
  const bp = C.batchPolicy(probs, S.thr, S.frac);
  const decOK = S.base.every((b, i) => +bp.decision[i] === S.sub.get(b.id).decision);
  const cal = S.model.calibration, transported = C.expit(-(cal.a * S.policy.raw_oof_threshold + cal.b));
  const shaAvail = Object.values(S.files).every(x => x.sha);
  const filesOK = Object.values(S.files).every(x => x.sha === x.entry.sha256);
  const modelOK = S.files["artifacts/final_model/model.json"].sha === S.modelManifest["model.json"];
  S.checks = [
    [shaAvail ? filesOK : null, `${S.manifest.files.length} files match files_manifest.json`],
    [shaAvail ? modelOK : null, "model.json matches model_manifest.json"],
    [maxDiff < 1e-12, `${int(probs.length)} probabilities reproduce submission.csv (max diff ${maxDiff.toExponential(1)})`],
    [decOK && bp.flagged === S.policy.batch_audit.flagged, `${bp.flagged} flags reproduce submission.csv`],
    [Math.abs(transported - S.thr) < 1e-12, `threshold ${f(S.thr, 4)} = calibration(${f(S.policy.raw_oof_threshold, 4)})`]];
  S.trusted = S.checks.every(c => c[0] !== false);
  $("#integrity").innerHTML = (S.trusted ? `<span class="ok">Verified in your browser</span>` : `<span class="bad">A check failed — results are not trusted</span>`) +
    S.checks.map(([ok, t]) => `<span class="${ok === false ? "bad" : ok ? "ok" : "note"}">${ok === false ? "✗" : ok ? "✓" : "–"} <span style="font-weight:400;color:var(--ink)">${esc(t)}</span></span>`).join("") + ` <a href="#files">details</a>`;
}

function rebatch() {
  S.all = S.base.concat(S.mine);
  S.batch = C.batchPolicy(S.all.map(a => a.s.probability), S.thr, S.frac);
  S.all.forEach((a, i) => { a.i = i; a.rank = S.batch.rank[i]; a.status = S.batch.decision[i] ? "flag" : S.batch.eligible[i] ? "cap" : "none"; });
  S.sorted = S.all.slice().sort((x, y) => y.s.probability - x.s.probability);
}
const STATUS = { flag: ["Flag for review", "t-flag"], cap: ["Removed by cap", "t-cap"], none: ["No flag", "t-none"] };
const tag = s => `<span class="tag ${STATUS[s][1]}">${STATUS[s][0]}</span>`;

/* ---------- applicant ---------- */
const GROUPS = [["About you", ["age", "months_employed", "residence_years", "@region", "@employment"]],
  ["Income and obligations", ["income_sar", "existing_obligations_sar", "savings_balance_sar", "salary_transfer"]],
  ["Request", ["loan_amount_sar", "tenor_months"]],
  ["Credit file", ["bureau_score", "recent_inquiries", "prior_defaults", "credit_history_months", "utilization_ratio", "num_open_accounts"]]];
function buildForm() {
  const spec = Object.fromEntries(S.contract.features.map(x => [x.name, x]));
  const field = n => {
    if (n === "@region") return `<label class="f"><b>Region</b><select id="f_region"><option value="">Choose…</option><option value="central">Central</option><option value="western">Western</option><option value="eastern">Eastern</option><option value="other">Other</option></select><span>region_central / western / eastern; other = all 0</span></label>`;
    if (n === "@employment") return `<label class="f"><b>Employment</b><select id="f_employment"><option value="">Choose…</option><option value="government">Government</option><option value="private">Private sector</option><option value="self_employed">Self-employed</option></select><span>exactly one employment flag</span></label>`;
    const s = spec[n];
    if (n === "salary_transfer") return `<label class="f"><b>${esc(s.description_en)}</b><select id="f_${n}"><option value="">Choose…</option><option value="1">Yes</option><option value="0">No</option></select><span><code>${n}</code></span></label>`;
    const step = s.dtype === "integer" ? 1 : "any";
    return `<label class="f"><b>${esc(s.description_en)}</b><input id="f_${n}" type="number" inputmode="decimal" step="${step}" min="${s.range[0]}" max="${s.range[1]}" placeholder="${s.range[0]}–${s.range[1]}"><span><code>${n}</code> · ${esc(s.unit)}${s.nullable ? " · optional: blank uses the training median" : ""}</span></label>`;
  };
  const placed = new Set(GROUPS.flatMap(g => g[1])), rest = S.contract.features.map(x => x.name).filter(n => !placed.has(n) && n !== "dti" && !n.startsWith("region_") && !n.startsWith("employment_"));
  $("#fieldsets").innerHTML = GROUPS.concat(rest.length ? [["Other fields", rest]] : []).map(([g, ns]) => `<fieldset><legend>${g}</legend><div class="fields">${ns.map(field).join("")}</div></fieldset>`).join("");
  receiptIntro();
}
function receiptIntro() {
  const leak = S.contract.leakage_features.map(x => `<code>${esc(x.name)}</code>`).join(" and ");
  $("#receipt").innerHTML = `<h3>What happens after you submit</h3><ol class="note" style="font-size:15px;padding-left:20px">
   <li>Your request is scored with the frozen model in <code>model.json</code>, exactly as <code>inference.predict</code> does.</li>
   <li>It joins today's batch of ${int(S.challenge.length)} challenge requests. The policy in <code>final_policy.json</code> flags requests at or above ${f(S.thr, 4)}, up to ${pct(S.frac, 0)} of the batch.</li>
   <li>You see only the status. The probability and reasons stay on the lender desk.</li></ol>
   <p class="note">Never collected: ${leak} (post-outcome, listed as leakage in the contract).</p>`;
}
function readForm() {
  const spec = Object.fromEntries(S.contract.features.map(x => [x.name, x])), row = {}, errs = [];
  for (const s of S.contract.features) {
    if (s.name.startsWith("region_") || s.name.startsWith("employment_") || s.name === "dti") continue;
    const el = $("#f_" + s.name), raw = el.value.trim();
    if (raw === "") { if (s.nullable) { row[s.name] = null; continue; } errs.push(s.description_en); continue; }
    const v = Number(raw);
    if (!Number.isFinite(v) || v < s.range[0] || v > s.range[1] || (s.dtype === "integer" && v % 1)) { errs.push(`${s.description_en} (${s.range[0]}–${s.range[1]}${s.dtype === "integer" ? ", whole number" : ""})`); continue; }
    row[s.name] = v;
  }
  const reg = $("#f_region").value, emp = $("#f_employment").value;
  if (!reg) errs.push("Region"); if (!emp) errs.push("Employment");
  for (const r of ["central", "western", "eastern"]) row["region_" + r] = +(reg === r);
  for (const e of ["government", "private", "self_employed"]) row["employment_" + e] = +(emp === e);
  if (row.income_sar && row.loan_amount_sar && row.tenor_months && row.existing_obligations_sar !== undefined) {
    row.dti = Math.round(((row.existing_obligations_sar + row.loan_amount_sar / row.tenor_months) / row.income_sar) * 1e4) / 1e4;
    const r = spec.dti.range; if (row.dti < r[0] || row.dti > r[1]) errs.push(`dti ${row.dti} is outside the trained range ${r[0]}–${r[1]}`);
  }
  return { row, errs };
}
function updDti() { const { row } = readForm(); $("#dtiOut").textContent = row.dti !== undefined ? row.dti.toFixed(4) : "—"; }
function submit(e) {
  e.preventDefault(); const { row, errs } = readForm();
  if (errs.length) { $("#err").textContent = "Please check: " + errs.join("; "); return; }
  $("#err").textContent = "";
  const id = "AP-" + String(S.mine.length + 1).padStart(4, "0");
  S.mine.push({ id, row: { application_id: id, ...row }, who: "you", s: C.score(S.model, row) });
  try { localStorage.setItem("tamweel_requests", JSON.stringify(S.mine.map(m => ({ id: m.id, row: m.row, who: "you" })))); } catch (err) {}
  rebatch();
  const a = S.all.find(x => x.id === id);
  const msg = { flag: "Sent for manual review", cap: "Passed the threshold, but today's review queue is full, so no review flag", none: "No review flag" }[a.status];
  $("#receipt").innerHTML = `<h3>Application ${esc(id)} received</h3><p style="font-size:18px;margin:10px 0">${tag(a.status)}</p><p>${msg}.</p>
   <p class="note">Scored with today's batch of ${int(S.all.length)} requests. The applicant view stops here.</p>
   <p><a href="#desk" data-open="${esc(id)}">Trace this decision as the lender →</a></p>`;
}

/* ---------- lender desk ---------- */
function renderDesk() {
  const B = S.batch, mine = S.mine.length;
  $("#deskLead").innerHTML = `Today's batch is the ${int(S.challenge.length)}-request challenge file${mine ? ` plus ${mine} request${mine > 1 ? "s" : ""} you submitted` : ""}. Rule from <code>final_policy.json</code>: flag when the calibrated probability ≥ ${f(S.thr, 4)}, then keep the highest ${pct(S.frac, 0)} of the batch.`;
  $("#kpis").innerHTML = [["Requests", int(B.n), mine ? `${mine} yours` : "challenge file"], ["Passed threshold", int(B.eligibleCount), `≥ ${f(S.thr, 4)}`],
   ["Review capacity", int(B.cap), `floor(${S.frac} × ${int(B.n)})`], ["Flagged", int(B.flagged), `${int(B.eligibleCount - B.flagged)} removed by cap`],
   ["Boundary score", B.boundary === null ? "—" : f(B.boundary, 4), "lowest flagged probability"]].map(k => `<div class="kpi"><small>${k[0]}</small><span class="num">${k[1]}</span><small>${k[2]}</small></div>`).join("");
  histogram();
  $("#filters").innerHTML = [["all", "All"], ["flag", "Flag for review"], ["cap", "Removed by cap"], ["none", "No flag"], ["you", "Yours"]].map(([k, l]) => `<button type="button" data-filter="${k}" aria-pressed="${S.filter === k}">${l}</button>`).join("");
  queue();
}
function histogram() {
  const p = S.all.map(a => a.s.probability), top = Math.min(1, Math.ceil(Math.max(...p) * 20) / 20), w = 0.01, nb = Math.round(top / w);
  const counts = new Array(nb).fill(0); p.forEach(v => counts[Math.min(nb - 1, Math.floor(v / w))]++);
  const ym = Math.max(...counts);
  const fr = frame({ xr: [0, top], yr: [0, ym * 1.08], xlab: "Calibrated probability", ylab: "Requests", xticks: ticks(0, top, 0.05), yticks: ticks(0, ym, ym > 400 ? 100 : 50), h: 250 });
  const bars = counts.map((c, i) => { const x0 = fr.sx(i * w), x1 = fr.sx((i + 1) * w), col = (i + 1) * w <= S.thr ? "var(--teal)" : "var(--amber)"; return `<rect x="${x0 + .5}" y="${fr.sy(c)}" width="${Math.max(0, x1 - x0 - 1)}" height="${fr.sy(0) - fr.sy(c)}" fill="${col}" opacity=".75"/>`; }).join("");
  const vl = (x, col, label, dy) => `<line x1="${fr.sx(x)}" x2="${fr.sx(x)}" y1="${fr.m.t}" y2="${fr.sy(0)}" stroke="${col}" stroke-width="2" stroke-dasharray="5 4"/><text x="${fr.sx(x) + 6}" y="${fr.m.t + dy}" class="lab">${label}</text>`;
  const g = fr.wrap(bars + vl(S.thr, "var(--amber)", `threshold ${f(S.thr, 4)}`, 12) + (S.batch.boundary ? vl(S.batch.boundary, "var(--coral)", `cap boundary ${f(S.batch.boundary, 4)}`, 30) : ""), "Histogram of calibrated probabilities");
  const t = `<div class="scroll"><table><thead><tr><th>Bin</th><th class="n">Requests</th></tr></thead><tbody>${counts.map((c, i) => c ? `<tr><td>${f(i * w, 2)}–${f((i + 1) * w, 2)}</td><td class="n">${c}</td></tr>` : "").join("")}</tbody></table></div>`;
  $("#histPanel").innerHTML = panelBody("Where today's batch falls", [src("submission.csv"), src("final_policy.json")], "Most requests sit far below the threshold; the review queue is filled from the right tail.", g, t);
}
function queue() {
  const q = S.query.trim().toUpperCase();
  const list = S.sorted.filter(a => (S.filter === "all" || (S.filter === "you" ? a.who === "you" : a.status === S.filter)) && (!q || a.id.toUpperCase().includes(q)));
  if (!S.sel || !S.all.find(a => a.id === S.sel)) S.sel = S.sorted[0].id;
  $("#queue").innerHTML = list.slice(0, S.shown).map(a => `<tr class="row${a.id === S.sel ? " sel" : ""}" data-id="${esc(a.id)}" tabindex="0"><td><b>${esc(a.id)}</b>${a.who === "you" ? ' <span class="src">yours</span>' : ""}${a.rank ? ` <span class="note">rank ${a.rank}</span>` : ""}</td><td class="n"><span class="pbar"><i style="width:${Math.min(100, a.s.probability / 0.5 * 100)}%"></i></span>${f(a.s.probability, 4)}</td><td>${tag(a.status)}</td></tr>`).join("") || `<tr><td colspan="3" class="note">No application matches.</td></tr>`;
  $("#qnote").textContent = `Showing ${Math.min(S.shown, list.length)} of ${int(list.length)}.`;
  $("#more").classList.toggle("hidden", list.length <= S.shown);
  trace();
}
function trace() {
  const a = S.all.find(x => x.id === S.sel), s = a.s, B = S.batch, spec = Object.fromEntries(S.contract.features.map(x => [x.name, x]));
  const parts = [...s.parts].sort((x, y) => Math.abs(y.contribution) - Math.abs(x.contribution)), top = parts.slice(0, 8), mx = Math.max(...top.map(p => Math.abs(p.contribution)), 0.3);
  const fileP = a.who === "file" ? S.sub.get(a.id) : null, cal = S.model.calibration;
  const scaleMax = Math.max(0.4, Math.ceil(s.probability * 12) / 10), X = v => Math.min(100, v / scaleMax * 100);
  const apC = s.probability * S.fn, flC = (1 - s.probability) * S.fp, be = S.fp / (S.fn + S.fp);
  const why = { flag: `Its probability ${f(s.probability, 4)} is at or above the threshold ${f(S.thr, 4)}, and it ranks ${a.rank} of ${B.eligibleCount} eligible requests, inside the ${B.cap} review slots.`,
    cap: `It passed the threshold but ranks ${a.rank} of ${B.eligibleCount}, below the cap boundary ${f(B.boundary, 4)}. Only ${B.cap} requests can be reviewed, so it is not flagged.`,
    none: `Its probability ${f(s.probability, 4)} is below the threshold ${f(S.thr, 4)}, so it is not flagged.` }[a.status];
  const val = p => p.name.endsWith("_sar") ? int(p.value) : (spec[p.name].dtype === "integer" ? String(p.value) : f(p.value, 4));
  $("#trace").innerHTML = `<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline"><h2>${esc(a.id)}</h2>${tag(a.status)}</div>
  <p class="note">Instance trace · ${a.who === "file" ? "row from <code>tamweel_challenge.csv</code>" : "submitted through the applicant form"}</p>
  <div class="step"><span class="dot">1</span><div><b>Inputs at application time</b><div class="srcs">${src(a.who === "file" ? "tamweel_challenge.csv" : "applicant form")}${src("model.json · medians")}</div>
   <div class="scroll" style="margin-top:8px"><table><tbody>${s.parts.map(p => `<tr><td><code>${p.name}</code></td><td class="n">${val(p)}${p.imputed ? ' <span class="src">blank → median</span>' : ""}</td></tr>`).join("")}</tbody></table></div></div></div>
  <div class="step"><span class="dot">2</span><div><b>Reason codes: coefficient × standardized value</b><div class="srcs">${src("model.json · mean, scale, coef")}</div>
   <p class="note">Contribution = coef × (value − mean) ÷ scale, in log-odds. For a linear model this is its exact SHAP value against the training mean.</p>
   ${top.map(p => { const w = Math.abs(p.contribution) / mx * 50; return `<div class="reason"><span><code>${p.name}</code></span><div class="rtrack"><i style="${p.contribution >= 0 ? `left:50%;width:${w}%;background:var(--coral)` : `right:50%;width:${w}%;background:var(--teal)`}"></i></div><span class="note n">${p.contribution >= 0 ? "+" : ""}${f(p.contribution, 3)}</span></div>`; }).join("")}
   <details><summary>All 22 terms</summary><div class="scroll"><table><thead><tr><th>Feature</th><th class="n">z</th><th class="n">coef</th><th class="n">contribution</th></tr></thead><tbody>${s.parts.map(p => `<tr><td><code>${p.name}</code></td><td class="n">${f(p.z, 3)}</td><td class="n">${f(p.coef, 4)}</td><td class="n">${f(p.contribution, 4)}</td></tr>`).join("")}</tbody></table></div></details></div></div>
  <div class="step"><span class="dot">3</span><div><b>From log-odds to calibrated probability</b><div class="srcs">${src("model.json · intercept, calibration a, b")}${fileP ? src("submission.csv") : ""}</div>
   <div class="eq">logit = intercept ${f(s.intercept, 4)} + Σ contributions ${f(s.logit - s.intercept, 4)} = <b>${f(s.logit, 4)}</b><br>raw = expit(logit) = ${f(s.raw, 6)}<br>probability = expit(−(${f(cal.a, 4)} × raw + ${f(cal.b, 4)})) = <b>${f(s.probability, 6)}</b>${fileP ? `<br>submission.csv: ${f(fileP.probability, 6)} · difference ${Math.abs(fileP.probability - s.probability).toExponential(1)}` : ""}</div></div></div>
  <div class="step"><span class="dot">4</span><div><b>Compare with the frozen threshold</b><div class="srcs">${src("final_policy.json · calibrated_threshold")}</div>
   <div class="rail"><div class="ln"></div><div class="mk" style="left:${X(S.thr)}%;background:var(--amber)"><span style="color:var(--amber)">threshold ${f(S.thr, 4)}</span></div>${B.boundary ? `<div class="mk" style="left:${X(B.boundary)}%;background:var(--coral)"><span style="color:var(--coral);top:38px">cap ${f(B.boundary, 4)}</span></div>` : ""}<div class="me" style="left:${X(s.probability)}%" title="${f(s.probability, 4)}"></div></div>
   <p class="note" style="margin-top:22px">Scale 0 to ${f(scaleMax, 1)}. Dot = this request.</p></div></div>
  <div class="step"><span class="dot">5</span><div><b>Review capacity</b><div class="srcs">${src("final_policy.json · max_flag_fraction")}</div><p class="note">${int(B.eligibleCount)} of ${int(B.n)} requests passed the threshold; capacity is floor(${S.frac} × ${int(B.n)}) = ${B.cap}. ${a.rank ? `This request ranks <b>${a.rank}</b>.` : "This request does not compete for a slot."}</p></div></div>
  <div class="step"><span class="dot">6</span><div><b>Cost of each action</b><div class="srcs">${src("final_policy.json · cost_policy")}</div>
   <div class="costs"><div class="${apC <= flC ? "win" : ""}"><small class="note">No flag: p × FN (${S.fn})</small><div class="num" style="font-size:28px">${f(apC, 3)}</div></div><div class="${flC < apC ? "win" : ""}"><small class="note">Flag: (1 − p) × FP (${S.fp})</small><div class="num" style="font-size:28px">${f(flC, 3)}</div></div></div>
   <p class="note">Cost alone would flag above ${f(be, 4)}. The frozen threshold is higher because it was chosen on OOF data under the ${pct(S.frac, 0)} capacity limit in every period.</p>
   <p style="margin-top:10px"><b>Decision:</b> ${why}</p></div></div>`;
}

/* ---------- charts ---------- */
function ticks(a, b, step) { const t = []; for (let v = a; v <= b + 1e-9; v += step) t.push(+v.toFixed(6)); return t; }
function frame(o) {
  const W = o.W || 640, H = o.h || 280, m = { l: o.ml || 54, r: 18, t: 30, b: 44 };
  const sx = v => m.l + (v - o.xr[0]) / (o.xr[1] - o.xr[0]) * (W - m.l - m.r), sy = v => H - m.b - (v - o.yr[0]) / (o.yr[1] - o.yr[0]) * (H - m.t - m.b);
  const fmt = o.fmt || (v => +v.toFixed(3));
  const grid = (o.yticks || []).map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--grid)"/><text x="${m.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmt(v)}</text>`).join("") +
    (o.xticks || []).map(v => `<text x="${sx(v)}" y="${H - m.b + 18}" text-anchor="middle">${fmt(v)}</text>`).join("") +
    `<line x1="${m.l}" x2="${W - m.r}" y1="${sy(o.yr[0])}" y2="${sy(o.yr[0])}" stroke="var(--mute)"/>` +
    `<text x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${esc(o.xlab || "")}</text><text x="${m.l - 8}" y="14" text-anchor="start">${esc(o.ylab || "")}</text>`;
  return { sx, sy, m, W, H, wrap: (inner, label) => `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">${grid}${inner}</svg>` };
}
const path = (pts, fr) => pts.map((p, i) => `${i ? "L" : "M"}${fr.sx(p[0]).toFixed(1)},${fr.sy(p[1]).toFixed(1)}`).join("");
function hbars(items, max, fmt, color) {
  return `<div>${items.map(it => `<div class="reason" style="grid-template-columns:minmax(0,130px) minmax(0,1fr) 64px"><span>${esc(it.label)}</span><div style="position:relative;height:18px;background:var(--line);border-radius:4px"><i style="position:absolute;inset:0 auto 0 0;width:${Math.min(100, it.value / max * 100)}%;border-radius:4px;background:${it.hl ? "var(--teal)" : color || "var(--mute)"}"></i>${it.err ? `<b style="position:absolute;top:-3px;bottom:-3px;left:${(it.value - it.err) / max * 100}%;width:${2 * it.err / max * 100}%;border-left:2px solid var(--ink);border-right:2px solid var(--ink);opacity:.5"></b>` : ""}</div><span class="note n">${fmt(it.value)}</span></div>`).join("")}</div>`;
}
const table = (cols, rows) => `<div class="scroll"><table class="stack"><thead><tr>${cols.map(c => `<th${c[2] ? ' class="n"' : ""}>${c[0]}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${cols.map(c => `<td${c[2] ? ' class="n"' : ""}>${c[1](r)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
function panelBody(title, sources, note, graph, tbl) {
  return `<header><div><h3>${title}</h3><div class="srcs">${sources.join("")}</div></div><div class="seg" role="group" aria-label="View"><button type="button" data-view="graph" aria-pressed="true">Graph</button><button type="button" data-view="table" aria-pressed="false">Table</button></div></header>
   <p class="note">${note}</p><div class="v-graph">${graph}</div><div class="v-table">${tbl}</div>`;
}
const panel = (title, sources, note, graph, tbl) => `<section class="card panel" data-view="graph">${panelBody(title, sources, note, graph, tbl)}</section>`;
const legend = items => `<div class="legend">${items.map(([c, l, dash]) => `<span><i style="background:${dash ? "transparent;border-top:2px dashed " + c + ";height:0;margin-bottom:4px" : c}"></i>${l}</span>`).join("")}</div>`;

/* ---------- evidence ---------- */
function renderEvidence() {
  const gate = S.metrics.gate, chosen = S.ens.find(r => r.candidate === gate.chosen), op = S.metrics.oof_threshold_selection;
  let h = `<h1>${esc(gate.chosen)} Regression kept single: ${esc(gate.decision)}</h1>
   <p class="lead">Every figure on this page is read from the files named under each panel. Switch any panel between graph and table.</p>
   <div class="srcs">${src("final_metrics.json · gate")}${src("ensemble_comparison.csv")}</div>
   <div class="kpis">${[["Mean AP, 3 forward folds", f(chosen.mean_ap, 3), `fold SD ${f(chosen.fold_sd, 3)}`], ["Brier · ECE", `${f(chosen.mean_brier, 4)} · ${f(chosen.mean_ece, 4)}`, "mean over folds"],
    ["Threshold", f(S.policy.calibrated_threshold, 4), `calibrated · raw ${f(S.policy.raw_oof_threshold, 4)}`], ["OOF recall · precision", `${pct(op.recall)} · ${pct(op.precision)}`, `${op.flagged} of ${int(op.rows)} flagged`],
    ["Challenge batch", `${S.policy.batch_audit.flagged} / ${int(S.policy.batch_audit.rows)}`, `${S.policy.batch_audit.removed_by_cap} removed by cap`]].map(k => `<div class="kpi"><small>${k[0]}</small><span class="num">${k[1]}</span><small>${k[2]}</small></div>`).join("")}</div>`;
  const apMax = Math.ceil(Math.max(...S.ens.map(r => r.mean_ap + r.fold_sd)) * 10) / 10;
  h += panel("1 · Candidate comparison", [src("ensemble_comparison.csv"), src("final_metrics.json · gate")],
    `Gate: ${esc(gate.gate)}. No candidate passed, so the best single model was kept. Whiskers = ± one fold SD (descriptive, not a confidence interval).`,
    hbars(S.ens.map(r => ({ label: r.candidate, value: r.mean_ap, err: r.fold_sd, hl: r.candidate === gate.chosen })), apMax, v => f(v, 3)) + `<p class="note">Mean AP, axis 0 to ${apMax}.</p>`,
    table([["Candidate", r => esc(r.candidate)], ["Mean AP", r => f(r.mean_ap, 4), 1], ["Fold SD", r => f(r.fold_sd, 4), 1], ["Brier", r => f(r.mean_brier, 4), 1], ["ECE", r => f(r.mean_ece, 4), 1], ["Lift vs single", r => f(r.lift_vs_single, 4), 1], ["Passes gate", r => r.passes_gate ? "yes" : "no"]], S.ens));
  const cands = [...new Set(S.folds.map(r => r.candidate))], fl = [...new Set(S.folds.map(r => r.fold))];
  const apLo = Math.floor(Math.min(...S.folds.map(r => r.average_precision)) * 20) / 20, apHi = Math.ceil(Math.max(...S.folds.map(r => r.average_precision)) * 20) / 20;
  const shapes = ["circle", "square", "diamond"], fr2 = frame({ xr: [apLo, apHi], yr: [0, cands.length], xticks: ticks(apLo, apHi, 0.05), xlab: "Average Precision per fold", h: 80 + cands.length * 34, ml: 96 });
  const yc = i => fr2.sy(cands.length - i - 0.5);
  const mark = (x, y, k) => k === 0 ? `<circle cx="${x}" cy="${y}" r="6" fill="var(--teal)"/>` : k === 1 ? `<rect x="${x - 5}" y="${y - 5}" width="10" height="10" fill="var(--amber)"/>` : `<path d="M${x},${y - 7}L${x + 7},${y}L${x},${y + 7}L${x - 7},${y}Z" fill="var(--coral)"/>`;
  const foldG = fr2.wrap(cands.map((c, i) => `<text x="${fr2.m.l - 8}" y="${yc(i) + 4}" text-anchor="end" class="lab">${esc(c)}</text><line x1="${fr2.m.l}" x2="${fr2.W - fr2.m.r}" y1="${yc(i)}" y2="${yc(i)}" stroke="var(--grid)"/>` + S.folds.filter(r => r.candidate === c).map(r => mark(fr2.sx(r.average_precision), yc(i), fl.indexOf(r.fold))).join("")).join(""), "AP per fold per candidate");
  h += panel("2 · AP in each forward fold", [src("day5_fold_scores.csv")], "The ranking moves between folds; differences are smaller than the fold-to-fold spread.",
    foldG + legend(fl.map((x, k) => [["var(--teal)", "var(--amber)", "var(--coral)"][k], `Fold ${x} (${shapes[k]})`])),
    table([["Candidate", r => esc(r.candidate)], ["Fold", r => r.fold, 1], ["Rows", r => int(r.rows), 1], ["Positives", r => r.positives, 1], ["ROC-AUC", r => f(r.roc_auc, 4), 1], ["AP", r => f(r.average_precision, 4), 1], ["Brier", r => f(r.brier, 4), 1], ["ECE", r => f(r.ece, 4), 1]], S.folds));
  const y = S.oof.map(r => r.default_within_90d), p = S.oof.map(r => r.Logistic), cv = C.curves(y, p), prev = cv.P / (cv.P + cv.N);
  const fr3 = frame({ xr: [0, 1], yr: [0, 1], xticks: ticks(0, 1, 0.2), yticks: ticks(0, 1, 0.2), xlab: "False positive rate", ylab: "Recall", h: 300, W: 400, h: 340 });
  const fr4 = frame({ xr: [0, 1], yr: [0, 1], xticks: ticks(0, 1, 0.2), yticks: ticks(0, 1, 0.2), xlab: "Recall", ylab: "Precision", h: 300, W: 400, h: 340 });
  const rocG = fr3.wrap(`<path d="${path([[0, 0], [1, 1]], fr3)}" stroke="var(--mute)" stroke-dasharray="4 4" fill="none"/><path d="${path(cv.roc, fr3)}" stroke="var(--teal)" stroke-width="2.5" fill="none"/><circle cx="${fr3.sx(op.false_positive_rate)}" cy="${fr3.sy(op.recall)}" r="6" fill="var(--amber)" stroke="var(--card)" stroke-width="2"/>`, "ROC curve");
  const prG = fr4.wrap(`<path d="${path([[0, prev], [1, prev]], fr4)}" stroke="var(--mute)" stroke-dasharray="4 4" fill="none"/><path d="${path(cv.pr, fr4)}" stroke="var(--teal)" stroke-width="2.5" fill="none"/><circle cx="${fr4.sx(op.recall)}" cy="${fr4.sy(op.precision)}" r="6" fill="var(--amber)" stroke="var(--card)" stroke-width="2"/>`, "Precision-recall curve");
  h += panel("3 · ROC and precision–recall on labelled OOF rows", [src("day5_oof_predictions.csv · Logistic"), src("final_metrics.json · oof_threshold_selection")],
    `Curves computed in your browser from ${int(y.length)} out-of-fold rows (${cv.P} defaults, prevalence ${pct(prev)}). Pooled AP ${f(cv.ap, 3)} differs from the fold mean ${f(chosen.mean_ap, 3)} because folds are pooled into one ranking. Amber dot = chosen operating point.`,
    `<div class="grid2" style="gap:12px"><div>${rocG}<p class="note">ROC-AUC ${f(cv.auc, 3)} · dashed = random</p></div><div>${prG}<p class="note">AP ${f(cv.ap, 3)} · dashed = prevalence ${pct(prev)}</p></div></div>`,
    table([["Measure", r => r[0]], ["Value", r => r[1], 1]], [["OOF rows", int(y.length)], ["Defaults", cv.P], ["Pooled ROC-AUC", f(cv.auc, 4)], ["Pooled AP", f(cv.ap, 4)], ["Threshold (raw)", f(op.threshold, 4)], ["TP · FP · FN · TN", `${op.tp} · ${op.fp} · ${op.fn} · ${op.tn}`], ["Recall · precision", `${pct(op.recall)} · ${pct(op.precision)}`], ["FPR", pct(op.false_positive_rate)]]));
  const sw = S.sweep, maxL = Math.max(...sw.map(r => r.loss_units)), feas = sw.filter(r => r.capacity_feasible);
  const unc = sw.reduce((a, b) => b.loss_units < a.loss_units ? b : a), d05 = sw.filter(r => r.threshold >= 0.5).sort((a, b) => a.threshold - b.threshold)[0];
  const fr5 = frame({ xr: [0, 1], yr: [0, maxL * 1.05], xticks: ticks(0, 1, 0.1), yticks: ticks(0, maxL, 500), xlab: "Raw threshold (flag when score ≥ threshold)", ylab: "Loss units (10 FN + FP)", h: 300, fmt: v => v >= 10 ? int(v) : +v.toFixed(1) });
  const pts = [[unc, "var(--mute)", "lowest loss, no capacity"], [d05, "var(--ink)", "default 0.5"], [op, "var(--amber)", "chosen, capacity-feasible"]];
  const sweepG = fr5.wrap(`<path d="${path(sw.map(r => [r.threshold, r.loss_units]), fr5)}" stroke="var(--mute)" stroke-width="1.5" fill="none"/>` + feas.map(r => `<circle cx="${fr5.sx(r.threshold).toFixed(1)}" cy="${fr5.sy(r.loss_units).toFixed(1)}" r="1.8" fill="var(--teal)"/>`).join("") +
    pts.map(([r, c]) => `<circle cx="${fr5.sx(r.threshold)}" cy="${fr5.sy(r.loss_units)}" r="7" fill="${c}" stroke="var(--card)" stroke-width="2"/>`).join(""), "Loss by threshold");
  h += panel("4 · Choosing the threshold", [src("day5_threshold_sweep.csv"), src("final_metrics.json")], `Each point is one threshold on the OOF rows. Teal points keep every period within ${pct(S.frac, 0)} capacity. The chosen threshold is the lowest loss among them.`,
    sweepG + legend([["var(--teal)", "capacity-feasible thresholds"], ...pts.map(([r, c, l]) => [c, `${l}: ${f(r.threshold, 3)} → ${int(r.loss_units)} units`])]),
    table([["Rule", r => r[0]], ["Threshold", r => f(r[1].threshold, 4), 1], ["TP", r => r[1].tp, 1], ["FP", r => r[1].fp, 1], ["FN", r => r[1].fn, 1], ["Flagged", r => pct(r[1].flag_fraction), 1], ["Recall", r => pct(r[1].recall), 1], ["Precision", r => pct(r[1].precision), 1], ["Loss", r => int(r[1].loss_units), 1], ["Feasible", r => r[1].capacity_feasible ? "yes" : "no"]],
      [["Default 0.5", d05], ["Lowest loss, no capacity limit", unc], ["Chosen (capacity in every period)", op]]));
  const pmax = Math.max(...S.periods.map(r => r.capacity));
  h += `<div class="grid2">` + panel("5 · Capacity in each period", [src("day5_period_capacity.csv")], "Flags at the chosen threshold vs the review slots of each OOF period.",
    S.periods.map(r => `<p class="note" style="margin:10px 0 0">Period ${r.fold} · ${r.rows} requests</p>` + hbars([{ label: "Capacity", value: r.capacity }, { label: "Flagged", value: r.flagged, hl: true }], pmax, v => int(v))).join(""),
    table([["Period", r => r.fold, 1], ["Rows", r => r.rows, 1], ["Capacity", r => r.capacity, 1], ["Flagged", r => r.flagged, 1], ["Flag share", r => pct(r.flag_fraction), 1], ["Within", r => r.within_capacity ? "yes" : "no"]], S.periods)) +
    panel("6 · Sensitivity to the cost of a missed default", [src("day5_cost_sensitivity.csv")], "Same frozen threshold, different FN costs.",
    hbars(S.sens.map(r => ({ label: `FN = ${r.false_negative_cost}`, value: r.loss_units, hl: r.false_negative_cost === S.fn })), Math.max(...S.sens.map(r => r.loss_units)) * 1.1, v => int(v)) + `<p class="note">Loss units on the OOF rows.</p>`,
    table([["FN cost", r => r.false_negative_cost, 1], ["FP cost", r => r.false_positive_cost, 1], ["Threshold", r => f(r.fixed_threshold, 4), 1], ["Loss units", r => int(r.loss_units), 1]], S.sens)) + `</div>`;
  const rmax = Math.max(...S.regions.map(r => r.false_positive_rate)) * 1.25;
  h += panel("7 · Regions: false alarms and recall", [src("day5_region_audit.csv"), src("final_metrics.json · regional_oof")], `Gap in false positive rate: ${f(S.metrics.regional_oof.fpr_gap_percentage_points, 1)} percentage points. ${esc(S.metrics.regional_oof.interpretation)}`,
    `<div class="grid2" style="gap:16px"><div><p class="note">False positive rate</p>${hbars(S.regions.map(r => ({ label: r.region, value: r.false_positive_rate })), rmax, v => pct(v), "var(--amber)")}</div><div><p class="note">Recall</p>${hbars(S.regions.map(r => ({ label: r.region, value: r.recall, hl: true })), 1, v => pct(v))}</div></div>`,
    table([["Region", r => esc(r.region)], ["Rows", r => r.rows, 1], ["Negatives", r => r.negatives, 1], ["Positives", r => r.positives, 1], ["Flagged", r => r.flagged, 1], ["FP", r => r.false_positives, 1], ["TP", r => r.true_positives, 1], ["FPR", r => pct(r.false_positive_rate), 1], ["Recall", r => pct(r.recall), 1]], S.regions));
  const cd = S.metrics.calibration_fit_diagnostics, series = [...new Set(S.bins.map(r => r.series))], cols = ["var(--mute)", "var(--teal)"];
  const cMax = Math.ceil(Math.max(...S.bins.filter(r => r.count > 0).flatMap(r => [r.mean_probability, r.observed_rate])) * 10) / 10;
  const fr6 = frame({ xr: [0, cMax], yr: [0, cMax], xticks: ticks(0, cMax, 0.1), yticks: ticks(0, cMax, 0.1), xlab: "Mean predicted probability in bin", ylab: "Observed default rate", h: 300 });
  const calG = fr6.wrap(`<path d="${path([[0, 0], [cMax, cMax]], fr6)}" stroke="var(--mute)" stroke-dasharray="4 4" fill="none"/>` + series.map((sr, k) => { const b = S.bins.filter(r => r.series === sr && r.count > 0); return `<path d="${path(b.map(r => [r.mean_probability, r.observed_rate]), fr6)}" stroke="${cols[k]}" stroke-width="2" fill="none"/>` + b.map(r => `<circle cx="${fr6.sx(r.mean_probability)}" cy="${fr6.sy(r.observed_rate)}" r="${3 + Math.sqrt(r.count) / 6}" fill="${cols[k]}" opacity=".8"/>`).join(""); }).join(""), "Reliability diagram");
  h += panel("8 · Calibration check", [src("day5_calibration_fit_bins.csv"), src("final_metrics.json · calibration_fit_diagnostics")],
    `Scored on the same ${cd.raw_fit_diagnostic.rows} rows the sigmoid was fitted on, so it is a diagnostic, not an independent test. Here the sigmoid did not help: Brier ${f(cd.raw_fit_diagnostic.brier, 4)} → ${f(cd.sigmoid_fit_diagnostic.brier, 4)}, ECE ${f(cd.raw_fit_diagnostic.ece, 4)} → ${f(cd.sigmoid_fit_diagnostic.ece, 4)}. Dot size = rows in bin.`,
    calG + legend(series.map((sr, k) => [cols[k], sr]).concat([["var(--mute)", "perfect calibration", 1]])),
    table([["Series", r => esc(r.series)], ["Bin", r => `${f(r.lower, 1)}–${f(r.upper, 1)}`], ["Rows", r => r.count, 1], ["Mean predicted", r => f(r.mean_probability, 4), 1], ["Observed", r => f(r.observed_rate, 4), 1], ["Low support", r => r.low_support ? "yes" : "no"]], S.bins));
  $("#evidence").innerHTML = h;
}

/* ---------- files ---------- */
function renderFiles() {
  $("#filesBody").innerHTML = S.manifest.files.map(e => { const x = S.files[e.path]; const ok = x.sha ? x.sha === e.sha256 : null;
    return `<tr><td><a href="data/${esc(e.path)}" target="_blank" rel="noopener"><code>${esc(e.path)}</code></a></td><td>${esc(e.used_by)}</td><td class="n">${(e.bytes / 1024).toFixed(1)} KB</td><td><code>${esc(e.sha256.slice(0, 12))}</code></td><td class="${ok === false ? "bad" : ok ? "ok" : "note"}">${ok === false ? "✗ changed" : ok ? "✓ sha256 matches" : "not available here"}</td></tr>`; }).join("");
}

/* ---------- routing and events ---------- */
const VIEWS = ["apply", "desk", "evidence", "files"];
let built = {};
function route() {
  const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "apply";
  VIEWS.forEach(k => $("#v-" + k).classList.toggle("hidden", k !== v));
  document.querySelectorAll("nav a").forEach(a => a.dataset.v === v ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  if (v === "desk") renderDesk();
  if (v === "evidence" && !built.e) { renderEvidence(); built.e = 1; }
  if (v === "files") renderFiles();
}
document.addEventListener("click", e => {
  const vb = e.target.closest(".seg [data-view]");
  if (vb) { const pn = vb.closest(".panel"); pn.dataset.view = vb.dataset.view; pn.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", b === vb)); return; }
  const fb = e.target.closest("[data-filter]"); if (fb) { S.filter = fb.dataset.filter; S.shown = 60; renderDesk(); return; }
  const row = e.target.closest("tr.row"); if (row) { S.sel = row.dataset.id; queue(); return; }
  const op = e.target.closest("[data-open]"); if (op) { S.sel = op.dataset.open; S.filter = "all"; S.query = ""; }
});
document.addEventListener("keydown", e => { const row = e.target.closest && e.target.closest("tr.row"); if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); S.sel = row.dataset.id; queue(); } });
$("#more").addEventListener("click", () => { S.shown += 60; queue(); });
$("#q").addEventListener("input", e => { S.query = e.target.value; S.shown = 60; queue(); });
$("#form").addEventListener("input", updDti);
$("#form").addEventListener("submit", submit);
window.addEventListener("hashchange", route);
load().then(() => { $("#loading").remove(); $("#q").placeholder = `Find an application ID, e.g. ${S.challenge[0].application_id}`; buildForm(); route(); }).catch(err => {
  $("#loading").textContent = "Could not load the data files: " + err.message + ". Open the site through a web server (GitHub Pages or `python -m http.server`), not as a local file.";
  $("#integrity").innerHTML = `<span class="bad">Files not loaded</span>`;
});
}
