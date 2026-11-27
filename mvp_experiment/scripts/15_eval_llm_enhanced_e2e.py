"""Step 15: Calibrate leak-free Corridor Gemini Flash priors on val60 and run head-to-head
end-to-end reconstruction evaluation on the fixed 100-day MVP test cohort.

Zero test leakage:
  - LLM temperature & class-bias calibration (tau, gamma_a) fit ONLY on val60 matched stops.
  - Fusion weights (w_llm, w_zone) and MILP weights theta* calibrated ONLY on val60.
  - Test100 predictions use ONLY AVI observables + leak-free corridor LLM JSON.
"""
from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
import math
import multiprocessing as mp
import pickle
import sys
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

sys.dont_write_bytecode = True
e2e = importlib.import_module("12_reconstruct_trip_chains")
features, prism, network = e2e.features, e2e.prism, e2e.network

CLASSES, CLASS_ID, N_CLASSES = e2e.CLASSES, e2e.CLASS_ID, e2e.N_CLASSES
RESULTS = network.MVP / "results"
BATCH_DIR = network.MVP / "data" / "corridor_llm_batches"
SUBSET_100 = network.MVP / "data" / "batches_100" / "ground_truth.json"

VAL_LLM_PATH = RESULTS / "corridor_gemini_val60.json"
TEST_LLM_PATH = RESULTS / "corridor_gemini_test100.json"

COMPARE_MODELS = (
    "M1_Decoupled_Physical",
    "M3_Joint_HAPP_NoLLM",
    "M3_LLM_Raw",
    "M3_LLM_ActOnly",
    "M3_LLM_Calibrated",
)


def norm_prob_vec(prob_dict: dict | None) -> np.ndarray:
    if not prob_dict:
        return np.full(N_CLASSES, 1.0 / N_CLASSES, dtype=np.float64)
    v = np.array([max(float(prob_dict.get(c, 0.0)), 1e-4) for c in CLASSES], dtype=np.float64)
    return v / v.sum()


def extract_llm_zone_and_act(pred: dict | None, zone_id: int, n_cands: int) -> tuple[float, np.ndarray]:
    """Return (zone_prob, act_probs_vec) for a specific candidate zone_id in a gap."""
    if pred is None:
        return 1.0 / max(n_cands, 1), np.full(N_CLASSES, 1.0 / N_CLASSES, dtype=np.float64)

    gap_act = norm_prob_vec(pred.get("gap_activity_probs"))
    top_zones = pred.get("top_zones") or []
    z_map = {}
    z_sum = 0.0
    for tz in top_zones:
        try:
            zid = int(tz["zone_id"])
            zp = max(float(tz.get("zone_prob", 0.0)), 1e-3)
        except Exception:
            continue
        z_map[zid] = (zp, norm_prob_vec(tz.get("activity_probs")))
        z_sum += zp

    # Smooth remaining mass across unlisted candidates
    rem_cands = max(n_cands - len(z_map), 1)
    rem_mass = max(1.0 - min(z_sum, 0.95), 0.05)
    if zone_id in z_map:
        zp, ap = z_map[zone_id]
        # Renormalize cleanly with residual floor
        scale = (1.0 - rem_mass) / max(z_sum, 1e-6)
        return max(zp * scale, 1e-4), ap
    return max(rem_mass / rem_cands, 1e-4), gap_act


def fit_llm_calibration(val_chains: list[dict], val_truth: dict, val_llm: dict) -> tuple[float, np.ndarray]:
    """Fit temperature tau and class bias gamma_a strictly on val60 true-stop matched gaps."""
    logits_list, y_list = [], []
    for chain in val_chains:
        pdid = chain["person_day_id"]
        real_by_gap = {}
        for s in val_truth[pdid]:
            if s["interval"] is not None:
                real_by_gap.setdefault(s["interval"], []).append(s)
        for gap in chain["intervals"]:
            gidx = gap["index"]
            if gidx not in real_by_gap or not gap["candidates"]:
                continue
            gk = f"{pdid}_{gidx}"
            pred = val_llm.get(gk)
            if pred is None:
                continue
            # Match ground-truth tract candidate if present, else top-1 candidate
            real_stop = real_by_gap[gidx][0]
            cand = next((c for c in gap["candidates"] if c["tract_id"] == real_stop["tract_id"]), gap["candidates"][0])
            _, ap = extract_llm_zone_and_act(pred, int(cand["zone_id"]), len(gap["candidates"]))
            logits_list.append(np.log(np.clip(ap, 1e-4, 1.0)))
            y_list.append(CLASS_ID[real_stop["activity"]])

    if len(y_list) < 20:
        return 1.0, np.zeros(N_CLASSES, dtype=np.float64)

    X = np.asarray(logits_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.int64)
    # Mild inverse-sqrt class weights during bias calibration so rare classes aren't suppressed
    cnt = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    cw = np.where(cnt > 0, np.sqrt(len(y) / (N_CLASSES * np.maximum(cnt, 1.0))), 1.0)
    sw = cw[y]
    sw /= sw.mean()

    def loss_fn(params):
        log_inv_t = params[0]
        inv_t = math.exp(np.clip(log_inv_t, -1.5, 1.5))
        bias = params[1:]
        z = inv_t * X + bias[None, :]
        z -= z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        nll = -np.mean(sw * (z[np.arange(len(y)), y] - lse))
        reg = 0.08 * (log_inv_t ** 2) + 0.06 * float(np.sum(bias ** 2))
        return float(nll + reg)

    res = minimize(loss_fn, np.zeros(1 + N_CLASSES), method="L-BFGS-B")
    inv_t = math.exp(float(np.clip(res.x[0], -1.5, 1.5)))
    tau = 1.0 / inv_t
    gamma = res.x[1:] - np.mean(res.x[1:])
    return float(tau), gamma.astype(np.float64)


def make_llm_potential(llm_preds: dict, tau: float, gamma: np.ndarray, w_llm: float, w_zone: float,
                       eta_llm: float = 0.0, class_prior: np.ndarray | None = None):
    """Build observable semantic_potential(chain, gap, candidate) combining ML + Corridor LLM."""
    def potential(chain, gap, candidate):
        ml_pot = e2e.psi_sem(chain, gap, candidate)
        gk = f"{chain['person_day_id']}_{gap['index']}"
        pred = llm_preds.get(gk)
        if pred is None:
            return ml_pot
        n_cands = len(gap["candidates"])
        zp, ap = extract_llm_zone_and_act(pred, int(candidate["zone_id"]), n_cands)
        # Calibrated log-probability vector
        z = (1.0 / tau) * np.log(np.clip(ap, 1e-4, 1.0)) + gamma
        z = z - math.log(float(np.sum(np.exp(z - z.max())))) - z.max()
        if eta_llm > 0.0 and class_prior is not None:
            z = z - eta_llm * np.log(np.clip(class_prior, 1e-5, 1.0))
        llm_act_score = z + math.log(N_CLASSES)
        # Zone log-lift relative to uniform 1/n_cands
        zone_lift = math.log(max(zp, 1e-4) * max(n_cands, 1))
        return (1.0 - w_llm) * ml_pot + w_llm * llm_act_score + w_zone * zone_lift
    return potential


def summarize_custom(rows: list[dict], centroids: dict, model_names: tuple[str, ...]) -> dict:
    """Reuse step 12's exact summary logic for any list of model names."""
    orig = e2e.MODEL_NAMES
    try:
        e2e.MODEL_NAMES = model_names
        # Compute standard metrics per model (skipping 3-model hardcoded coupling, compute custom coupling below)
        all_counts, found = collections.Counter(), collections.Counter()
        for row in rows:
            for stop in row["intermediate_truth"]:
                all_counts[stop["activity"]] += 1
                gap = stop["interval"]
                found[stop["activity"]] += bool(gap is not None and stop["tract_id"] in row["candidate_tracts"][gap])
        result = {"n_chains": len(rows), "n_true_intermediate_stops": sum(all_counts.values()), "models": {}}
        for name in model_names:
            stop_counts, exact, triple, order, all_path, withheld_path = [], [], [], [], [], []
            stop_diffs, pairs, sim_all, sim_valid, runtimes = [], [], [], [], []
            for row in rows:
                m = row["predictions"][name]
                pred, real = m["stops"], row["intermediate_truth"]
                pg = {p["interval"] for p in pred}
                rg = {t["interval"] for t in real if t["interval"] is not None}
                stop_counts.append((len(pg & rg), len(pg), len(rg)))
                stop_diffs.append(abs(len(pred) - len(real)))
                exact.append((e2e.count_matches(pred, real, ("interval", "tract_id")), len(pred), len(real)))
                triple.append((e2e.count_matches(pred, real, ("interval", "tract_id", "activity")), len(pred), len(real)))
                _, lcs = e2e.edit_stats([p["tract_id"] for p in pred], [t["tract_id"] for t in real])
                order.append((lcs, len(pred), len(real)))
                pairs.extend(e2e.pairs_for_schedule(pred, real))
                seq = m["activity_sequence"]
                truth_seq = [r["activity"] if r["activity"] is not None else "<missing>" for r in row["truth_chain"]]
                edits, _ = e2e.edit_stats(seq, truth_seq)
                ds = 1.0 - edits / max(len(seq), len(truth_seq))
                sim_all.append(ds)
                if row["truth_start"]["activity"] is not None and row["truth_end"]["activity"] is not None:
                    sim_valid.append(ds)
                pm = row["path_metrics"][name]
                all_path.append(pm["physical"])
                withheld_path.append(pm["withheld"])
                ms = m.get("milp_solve_ms")
                if ms:
                    runtimes.append(ms)
            dists = [e2e.distance_km(p["tract_id"], r["tract_id"], centroids) for p, r in pairs]
            dists = [d for d in dists if d is not None]
            result["models"][name] = {
                "stop_detection": e2e.sum_prf(stop_counts),
                "daily_stop_count_mae": e2e.mean(stop_diffs),
                "exact_interval_tract": e2e.sum_prf(exact),
                "chain_order_tract": e2e.sum_prf(order),
                "mean_tract_centroid_error_km": e2e.mean(dists),
                "matched_arrival_mae_min": e2e.mean([abs(p["arrival_min"] - r["arrival_min"]) for p, r in pairs if r["arrival_min"] is not None]),
                "matched_duration_mae_min": e2e.mean([abs(p["duration_min"] - r["duration_min"]) for p, r in pairs if r["duration_min"] is not None]),
                "triple_match": e2e.sum_prf(triple),
                "matched_stop_activity": e2e.activity_metrics(pairs),
                "daily_activity_sequence_similarity_valid": e2e.mean(sim_valid),
                "physical_edge_set": e2e.sum_prf(all_path),
                "withheld_physical_edge_set": e2e.sum_prf(withheld_path),
                "milp_solve_ms": prism.distribution(runtimes) if runtimes else None,
            }
        return result
    finally:
        e2e.MODEL_NAMES = orig


def audit_llm_delta(rows: list[dict], base_name: str, llm_name: str) -> dict:
    """Directly measure how LLM changes locations & activities vs No-LLM M3."""
    tract_changed = 0
    tract_corrected = 0
    tract_harmed = 0
    act_changed = 0
    act_corrected = 0
    act_harmed = 0
    triple_gained = 0
    triple_lost = 0
    gap_path_base, gap_path_llm = [], []

    for row in rows:
        b_stops = {s["interval"]: s for s in row["predictions"][base_name]["stops"]}
        l_stops = {s["interval"]: s for s in row["predictions"][llm_name]["stops"]}
        truth_by_gap = collections.defaultdict(list)
        for s in row["intermediate_truth"]:
            if s["interval"] is not None:
                truth_by_gap[s["interval"]].append(s)

        for g in sorted(set(b_stops) & set(l_stops)):
            bs, ls = b_stops[g], l_stops[g]
            tr = truth_by_gap.get(g, [])
            true_tracts = {t["tract_id"] for t in tr}
            true_acts = {t["activity"] for t in tr}
            true_triples = {(t["tract_id"], t["activity"]) for t in tr}

            b_tok = (bs["tract_id"] in true_tracts)
            l_tok = (ls["tract_id"] in true_tracts)
            if bs["tract_id"] != ls["tract_id"]:
                tract_changed += 1
                if l_tok and not b_tok:
                    tract_corrected += 1
                elif b_tok and not l_tok:
                    tract_harmed += 1
                bp = row["path_metrics"][base_name]["withheld_by_interval"].get(g)
                lp = row["path_metrics"][llm_name]["withheld_by_interval"].get(g)
                if bp is not None and lp is not None:
                    gap_path_base.append(bp)
                    gap_path_llm.append(lp)

            b_aok = (bs["activity"] in true_acts)
            l_aok = (ls["activity"] in true_acts)
            if bs["activity"] != ls["activity"]:
                act_changed += 1
                if l_aok and not b_aok:
                    act_corrected += 1
                elif b_aok and not l_aok:
                    act_harmed += 1

            b_3 = ((bs["tract_id"], bs["activity"]) in true_triples)
            l_3 = ((ls["tract_id"], ls["activity"]) in true_triples)
            if l_3 and not b_3:
                triple_gained += 1
            elif b_3 and not l_3:
                triple_lost += 1

    bp_prf = e2e.sum_prf(gap_path_base) if gap_path_base else {"f1": None}
    lp_prf = e2e.sum_prf(gap_path_llm) if gap_path_llm else {"f1": None}
    return {
        "tract_changed_when_both_stop": tract_changed,
        "tract_corrected": tract_corrected,
        "tract_harmed": tract_harmed,
        "tract_net": tract_corrected - tract_harmed,
        "changed_tract_withheld_path_f1_base": bp_prf["f1"],
        "changed_tract_withheld_path_f1_llm": lp_prf["f1"],
        "act_changed_when_both_stop": act_changed,
        "act_corrected": act_corrected,
        "act_harmed": act_harmed,
        "act_net": act_corrected - act_harmed,
        "triple_gained_shared_gaps": triple_gained,
        "triple_lost_shared_gaps": triple_lost,
        "triple_net_shared_gaps": triple_gained - triple_lost,
    }


def write_showcase(rows: list[dict], test_llm: dict, base_name: str, llm_name: str) -> str:
    """Build plain-language Chinese showcase of Gemini Flash's reasoning vs M3 No-LLM vs Truth."""
    scored_days = []
    for row in rows:
        pdid = row["person_day_id"]
        b_stops = {s["interval"]: s for s in row["predictions"][base_name]["stops"]}
        l_stops = {s["interval"]: s for s in row["predictions"][llm_name]["stops"]}
        truth_by_gap = collections.defaultdict(list)
        for s in row["intermediate_truth"]:
            if s["interval"] is not None:
                truth_by_gap[s["interval"]].append(s)

        gains, losses, loc_wins, tail_wins = [], [], [], []
        for g in sorted(set(truth_by_gap) | set(b_stops) | set(l_stops)):
            gk = f"{pdid}_{g}"
            if gk not in test_llm:
                continue
            tr = truth_by_gap.get(g, [])
            bs, ls = b_stops.get(g), l_stops.get(g)
            true_tracts = {t["tract_id"] for t in tr}
            true_acts = {t["activity"] for t in tr}
            true_triples = {(t["tract_id"], t["activity"]) for t in tr}
            b3 = bool(bs and (bs["tract_id"], bs["activity"]) in true_triples)
            l3 = bool(ls and (ls["tract_id"], ls["activity"]) in true_triples)
            b_loc = bool(bs and bs["tract_id"] in true_tracts)
            l_loc = bool(ls and ls["tract_id"] in true_tracts)
            b_act = bool(bs and bs["activity"] in true_acts)
            l_act = bool(ls and ls["activity"] in true_acts)
            if l_loc and not b_loc:
                loc_wins.append(g)
            if l3 and not b3:
                gains.append(g)
            if l_act and not b_act and any(t["activity"] in ("healthcare", "school", "meal", "escort") for t in tr):
                tail_wins.append(g)
            if b3 and not l3:
                losses.append(g)
        scored_days.append((row, len(loc_wins), len(tail_wins), len(gains), len(losses)))

    picked = []
    used = set()
    def pick_one(title, key_fn):
        cands = [x for x in scored_days if x[0]["person_day_id"] not in used]
        if not cands:
            return
        best = max(cands, key=key_fn)
        used.add(best[0]["person_day_id"])
        picked.append((title, best[0]))

    pick_one("案例 1：大模型帮着同时修对了“地点 + 活动”", lambda x: (x[1] > 0 and x[3] > 0, x[1], x[3], -x[4]))
    pick_one("案例 2：大模型攻克长尾活动（看病/上学/吃饭/接送）", lambda x: (x[2], x[3], -x[4]))
    pick_one("案例 3：大模型仅凭路边 POI 纠偏停靠片区", lambda x: (x[1], x[3], -x[4]))
    pick_one("案例 4：多停留综合出行链（LLM 链级上下文推理）", lambda x: (x[3] - x[4], len(x[0]["intermediate_truth"])))
    pick_one("案例 5：诚实反面教材（大模型想多了 / 误判案例）", lambda x: (x[4], -x[3]))

    lines = [
        "# Corridor Gemini Flash 推理案例展示（无泄露 100 链测试集）",
        "",
        "以下展示完全来自测试集真实车辆。大模型**绝对没看过任何真值**，只看到了稀疏卡口时段 + 顺路候选微区的真实 OSM 店名/医院/学校名称。",
    ]
    for title, row in picked:
        pdid = row["person_day_id"]
        obs = row["observables"]
        b_stops = {s["interval"]: s for s in row["predictions"][base_name]["stops"]}
        l_stops = {s["interval"]: s for s in row["predictions"][llm_name]["stops"]}
        truth_by_gap = collections.defaultdict(list)
        for s in row["intermediate_truth"]:
            if s["interval"] is not None:
                truth_by_gap[s["interval"]].append(s)
        bp = row["path_metrics"][base_name]["withheld"]
        lp = row["path_metrics"][llm_name]["withheld"]
        bf1 = e2e.prf(*bp)["f1"] * 100
        lf1 = e2e.prf(*lp)["f1"] * 100

        lines += [
            "",
            f"## {title}（车辆日 `{pdid}`，{'周末' if obs['travel_dow'] in (6,7) else '工作日'}，卡口抓拍 {len(obs['observed_edges'])} 次）",
            f"- 隐藏道路 F1：无 LLM（M3）`{bf1:.2f}%` → 接入 LLM（M3+LLM）`{lf1:.2f}%`",
            "",
            "| 间隙 | 真实停留（答案） | M3 无 LLM | M3 + Corridor LLM | Gemini Flash 真实判断理由（仅看可观测路边店名与时段） |",
            "|---|---|---|---|---|",
        ]
        all_g = sorted(set(truth_by_gap) | set(b_stops) | set(l_stops))
        for g in all_g:
            tr = truth_by_gap.get(g, [])
            tr_s = "<br>".join(f"**{t['activity']}** · `{t['tract_id']}` ({t['duration_min']:.0f}m)" for t in tr) if tr else "—"
            bs = b_stops.get(g)
            bs_s = f"{bs['activity']} · `{bs['tract_id']}` ({bs['duration_min']:.0f}m)" if bs else "—"
            ls = l_stops.get(g)
            ls_s = f"**{ls['activity']}** · `{ls['tract_id']}` ({ls['duration_min']:.0f}m)" if ls else "—"
            gk = f"{pdid}_{g}"
            pred = test_llm.get(gk)
            reason = pred["brief_reasoning"].replace("|", "／").replace("\n", " ") if pred else "（低停靠概率间隙，未调用 LLM）"
            lines.append(f"| Gap#{g} | {tr_s} | {bs_s} | {ls_s} | {reason} |")
    return "\n".join(lines) + "\n"


def evaluate_100(test_chains_list: list[dict], predictions: dict, physical: np.ndarray):
    """Exact replica of step 12's evaluate() scoped strictly to the 100 finalized MVP chains."""
    assert len(test_chains_list) == 100 and len(predictions) == 100
    test_chains = {c["person_day_id"]: c for c in test_chains_list}
    labels = features.label_rows("test")[0]
    tract_features = json.loads(network.TRACTS.read_text(encoding="utf-8"))["features"]
    centroids = {
        r["properties"]["GEOID"]: (float(r["properties"]["Lon_Centroid"]), float(r["properties"]["Lat_Centroid"]))
        for r in tract_features
    }
    rows = []
    for record in network.json_lines(network.MASKED_TEST):
        key = record["person_day_id"]
        if key not in test_chains:
            continue
        chain = test_chains[key]
        pred = predictions[key]
        surveyed = features.read_truth(record, labels[key])
        truth, alignment = surveyed["intermediate_stops"], surveyed["observed_positions"]
        targets = record["target_edge_sequence"]
        observed = set(chain["observables"]["observed_edges"])
        actual = {e for e in targets if physical[e]}
        if alignment is not None:
            boundaries = [-1, *alignment, len(targets)]
            gap_paths = [
                {t for t in targets[a + 1:b] if physical[t] and t not in observed}
                for a, b in zip(boundaries, boundaries[1:])
            ]
        else:
            gap_paths = [None for _ in chain["intervals"]]

        path_metrics = {}
        for name, m in pred.items():
            reconstructed = set(m["physical_edge_sequence"])
            by_iv = {}
            for gidx, g_truth in enumerate(gap_paths):
                if g_truth is not None:
                    g_pred = {e for e in m["interval_physical_edge_sequences"][gidx] if physical[e] and e not in observed}
                    by_iv[gidx] = e2e.route_overlap(g_pred, g_truth)
            path_metrics[name] = {
                "physical": e2e.route_overlap(reconstructed, actual),
                "withheld": e2e.route_overlap(reconstructed - observed, actual - observed),
                "withheld_by_interval": by_iv,
            }
        truth_start, truth_end = surveyed["truth_start"], surveyed["truth_end"]
        candidate_tracts = [{c["tract_id"] for c in gap["candidates"]} for gap in chain["intervals"]]
        rows.append({
            "person_day_id": key,
            "observables": chain["observables"],
            "truth_start": truth_start,
            "truth_end": truth_end,
            "truth_chain": [truth_start, *truth, truth_end],
            "intermediate_truth": truth,
            "n_dropped_intermediate_activities": surveyed["n_dropped_intermediate_activities"],
            "predictions": pred,
            "path_metrics": path_metrics,
            "candidate_tracts": candidate_tracts,
            "n_candidate_tracts": [len(c) for c in candidate_tracts],
            "interval_statuses": [gap["status"] for gap in chain["intervals"]],
            "withheld_truth_by_gap": gap_paths,
            "alignment_ok": alignment is not None,
        })
    assert len(rows) == 100
    return rows, centroids


def main():
    t0 = perf_counter()
    with features.MODEL_FILE.open("rb") as f:
        model = pickle.load(f)

    manifest = json.loads((BATCH_DIR / "manifest.json").read_text(encoding="utf-8"))
    val60_ids = set(manifest["splits"]["val60"]["selected_pdids"])
    test100_ids = set(json.loads(SUBSET_100.read_text(encoding="utf-8"))["selected_pdids"])

    val_llm = json.loads(VAL_LLM_PATH.read_text(encoding="utf-8"))["gap_predictions"]
    test_llm = json.loads(TEST_LLM_PATH.read_text(encoding="utf-8"))["gap_predictions"]
    print(f"[15] Loaded LLM predictions: val60={len(val_llm)} gaps, test100={len(test_llm)} gaps")

    val_all = list(network.json_lines(features.candidate_file("val")))
    val_labels, _ = features.label_rows("val")
    val_truth = features.supervised_truth("val", val_labels)
    val60_chains = [c for c in val_all if c["person_day_id"] in val60_ids]

    # 1. Exact baseline M1 threshold & M3 (No-LLM) theta from full val (matching step 12)
    threshold, _ = e2e.select_stop_threshold(val_all, val_truth)
    theta_nollm, cal_nollm = e2e.calibrate(
        [e2e.Day(c, model, e2e.psi_sem) for c in val_all],
        [val_truth[c["person_day_id"]] for c in val_all],
    )
    print(f"[15] Recomputed M3_NoLLM theta* on val_276: {e2e.weight_dict(theta_nollm)}")

    # 2. Strictly on val60: fit LLM temperature & class-bias calibration
    tau_llm, gamma_llm = fit_llm_calibration(val60_chains, val_truth, val_llm)
    print(f"[15] Val60 LLM calibration: tau={tau_llm:.3f}, gamma={dict(zip(CLASSES, np.round(gamma_llm, 3)))}")

    # 3. Strictly on val60: calibrate M3_LLM_Raw, M3_LLM_ActOnly, and M3_LLM_Calibrated (selecting fusion weights on val60!)
    pot_raw_val = make_llm_potential(val_llm, 1.0, np.zeros(N_CLASSES), w_llm=0.7, w_zone=0.8)
    theta_raw, _ = e2e.calibrate(
        [e2e.Day(c, model, pot_raw_val) for c in val60_chains],
        [val_truth[c["person_day_id"]] for c in val60_chains],
    )

    pot_actonly_val = make_llm_potential(val_llm, tau_llm, gamma_llm, w_llm=0.7, w_zone=0.0)
    theta_actonly, _ = e2e.calibrate(
        [e2e.Day(c, model, pot_actonly_val) for c in val60_chains],
        [val_truth[c["person_day_id"]] for c in val60_chains],
    )

    # Small clean val60 grid search over (w_llm, w_zone) strictly on val60
    best_cfg = None
    best_val_score = -1.0
    best_theta_llm = None
    best_cal_llm = None
    for w_llm in (0.5, 0.75, 1.0):
        for w_zone in (0.5, 1.0, 1.5):
            pot_v = make_llm_potential(val_llm, tau_llm, gamma_llm, w_llm=w_llm, w_zone=w_zone)
            days_v = [e2e.Day(c, model, pot_v) for c in val60_chains]
            th_v, cal_v = e2e.calibrate(days_v, [val_truth[c["person_day_id"]] for c in val60_chains])
            preds_v = [d.stops(*e2e.solve(d, th_v)[:2]) for d in days_v]
            macro_v, sf1_v, triple_v = e2e.calibration_metrics(preds_v, [val_truth[c["person_day_id"]] for c in val60_chains])
            sc = (macro_v + sf1_v + triple_v) / 3.0
            print(f"[15] val60 grid w_llm={w_llm}, w_zone={w_zone} -> stop={sf1_v:.3f}, macro={macro_v:.3f}, triple={triple_v:.3f}, score={sc:.4f}")
            if sc > best_val_score:
                best_val_score = sc
                best_cfg = (w_llm, w_zone)
                best_theta_llm = th_v
                best_cal_llm = cal_v

    w_llm_star, w_zone_star = best_cfg
    print(f"[15] Selected on val60: w_llm={w_llm_star}, w_zone={w_zone_star}, theta_llm*={e2e.weight_dict(best_theta_llm)}")

    # 4. Run all 5 models strictly on held-out test100
    test_all = list(network.json_lines(features.candidate_file("test")))
    test_100 = [c for c in test_all if c["person_day_id"] in test100_ids]
    assert len(test_100) == 100

    pot_raw_test = make_llm_potential(test_llm, 1.0, np.zeros(N_CLASSES), w_llm=0.7, w_zone=0.8)
    pot_actonly_test = make_llm_potential(test_llm, tau_llm, gamma_llm, w_llm=0.7, w_zone=0.0)
    pot_cal_test = make_llm_potential(test_llm, tau_llm, gamma_llm, w_llm=w_llm_star, w_zone=w_zone_star)

    decisions = []
    for chain in test_100:
        day_base = e2e.Day(chain, model, e2e.psi_sem)
        day_raw = e2e.Day(chain, model, pot_raw_test)
        day_act = e2e.Day(chain, model, pot_actonly_test)
        day_cal = e2e.Day(chain, model, pot_cal_test)

        models = {
            "M1_Decoupled_Physical": {"stops": e2e.decoupled(day_base, model, threshold), "milp_solve_ms": None},
        }
        for mname, d_obj, th in [
            ("M3_Joint_HAPP_NoLLM", day_base, theta_nollm),
            ("M3_LLM_Raw", day_raw, theta_raw),
            ("M3_LLM_ActOnly", day_act, theta_actonly),
            ("M3_LLM_Calibrated", day_cal, best_theta_llm),
        ]:
            sel, sched, ms = e2e.solve(d_obj, th)
            models[mname] = {"stops": d_obj.stops(sel, sched), "milp_solve_ms": ms}

        for pred in models.values():
            pred["activity_sequence"] = ["home", *(s["activity"] for s in pred["stops"]), "home"]
            pred["day_anchors"] = [
                {"activity": "home", "tract_id": chain["observables"][k], "arrival_min": None, "departure_min": None, "duration_min": None}
                for k in ("known_day_start_tract_id", "known_day_end_tract_id")
            ]
        decisions.append((chain, models))

    # 5. Recover exact physical paths on road network for all 5 models across test_100
    data, net_summary = network.load_cache()
    e2e.ROUTER = e2e.Router(data, net_summary)
    predictions = {}
    with ProcessPoolExecutor(max_workers=features.WORKERS, mp_context=mp.get_context("fork")) as pool:
        for (chain, _), models in zip(decisions, pool.map(e2e.recover_chain, decisions, chunksize=2), strict=True):
            predictions[chain["person_day_id"]] = models

    # 6. Evaluate strictly after predictions & paths are finalized
    rows, centroids = evaluate_100(test_100, predictions, data["vocab_is_physical"])
    summary = summarize_custom(rows, centroids, COMPARE_MODELS)
    llm_delta_vs_m3 = audit_llm_delta(rows, "M3_Joint_HAPP_NoLLM", "M3_LLM_Calibrated")
    llm_delta_vs_m1 = audit_llm_delta(rows, "M1_Decoupled_Physical", "M3_LLM_Calibrated")

    out_json = {
        "n_chains_test": len(rows),
        "val60_llm_gaps": len(val_llm),
        "test100_llm_gaps": len(test_llm),
        "llm_calibration": {
            "tau_llm": tau_llm,
            "gamma_llm": dict(zip(CLASSES, map(float, gamma_llm))),
            "selected_w_llm": w_llm_star,
            "selected_w_zone": w_zone_star,
            "theta_nollm": e2e.weight_dict(theta_nollm),
            "theta_llm_calibrated": e2e.weight_dict(best_theta_llm),
        },
        "summary": summary,
        "llm_delta_vs_m3_nollm": llm_delta_vs_m3,
        "llm_delta_vs_m1_physical": llm_delta_vs_m1,
        "runtime_seconds": perf_counter() - t0,
    }
    (RESULTS / "e2e_llm_comparison_100chains.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write markdown summary
    md = [
        "# End-to-End 100-Chain Comparison: No-LLM vs Corridor Gemini Flash",
        "",
        f"- 严格零泄露评估：LLM 温度与偏置校准（`tau={tau_llm:.3f}`）、融合系数（`w_llm={w_llm_star}, w_zone={w_zone_star}`）及逆优化权重均只在独立验证集 `val60` 上拟合；测试集 `test100`（100 辆纯小汽车，302 个中间真实停留）全程未接触真值。",
        f"- 总耗时：`{perf_counter() - t0:.1f} s`；LLM 覆盖：`val60` {len(val_llm)} 段、`test100` {len(test_llm)} 段。",
        "",
        "## 主对比指标（100 辆测试车，端到端从稀疏卡口重构）",
        "",
        "| 模型 | 停靠 F1 | 精确地点 F1 (空隙+片区) | 顺序片区 F1 | **三合一全对 F1 (空隙+片区+活动)** | 匹配非家 Acc | **匹配非家 Macro-F1** | 隐藏道路 F1 | 平均地点误差(km) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in COMPARE_MODELS:
        r = summary["models"][m]
        act = r["matched_stop_activity"]
        md.append(
            f"| {m} | {r['stop_detection']['f1']*100:.2f}% | {r['exact_interval_tract']['f1']*100:.2f}% | "
            f"{r['chain_order_tract']['f1']*100:.2f}% | **{r['triple_match']['f1']*100:.2f}%** | "
            f"{act['non_home_accuracy']*100:.2f}% | **{act['non_home_macro_f1']*100:.2f}%** | "
            f"{r['withheld_physical_edge_set']['f1']*100:.2f}% | {r['mean_tract_centroid_error_km']:.3f} |"
        )

    md += [
        "",
        "## 9 类活动细分 F1（匹配停留，看长尾提升）",
        "",
        "| 活动类别 | M1 (无LLM) | M3 (无LLM) | M3+LLM Raw | M3+LLM ActOnly | **M3+LLM Calibrated** |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c in CLASSES:
        vals = [summary["models"][m]["matched_stop_activity"]["per_class"][c]["f1"] * 100 for m in COMPARE_MODELS]
        md.append(f"| {c} | {vals[0]:.2f}% | {vals[1]:.2f}% | {vals[2]:.2f}% | {vals[3]:.2f}% | **{vals[4]:.2f}%** |")

    md += [
        "",
        "## 大模型究竟帮了多少？（M3+LLM Calibrated vs M3 No-LLM 直接逐段对拍）",
        "",
        f"- **地点纠偏**：两者都判停靠但选了不同片区的空隙共 `{llm_delta_vs_m3['tract_changed_when_both_stop']}` 处 → LLM **改对 {llm_delta_vs_m3['tract_corrected']} 处 vs 改错 {llm_delta_vs_m3['tract_harmed']} 处（净增 {llm_delta_vs_m3['tract_net']:+d} 处）**！",
        f"- **活动纠偏**：两者都判停靠但判了不同活动的空隙共 `{llm_delta_vs_m3['act_changed_when_both_stop']}` 处 → LLM **改对 {llm_delta_vs_m3['act_corrected']} 处 vs 改错 {llm_delta_vs_m3['act_harmed']} 处（净增 {llm_delta_vs_m3['act_net']:+d} 处）**！",
        f"- **三合一全对净增（共同停靠段）**：新增全对 `{llm_delta_vs_m3['triple_gained_shared_gaps']}` 处 vs 丢失 `{llm_delta_vs_m3['triple_lost_shared_gaps']}` 处（**净增 {llm_delta_vs_m3['triple_net_shared_gaps']:+d} 处**）。",
    ]
    if llm_delta_vs_m3["changed_tract_withheld_path_f1_base"] is not None:
        bf = llm_delta_vs_m3["changed_tract_withheld_path_f1_base"] * 100
        lf = llm_delta_vs_m3["changed_tract_withheld_path_f1_llm"] * 100
        md.append(f"- **改判地点段的隐藏道路 F1**：`{bf:.2f}%` → `{lf:.2f}%`（`{lf - bf:+.2f} pp`）。")

    (RESULTS / "e2e_llm_comparison_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    showcase_md = write_showcase(rows, test_llm, "M3_Joint_HAPP_NoLLM", "M3_LLM_Calibrated")
    (RESULTS / "e2e_llm_reasoning_showcase.md").write_text(showcase_md, encoding="utf-8")
    print("\n".join(md[:35]))
    print(f"[15] Saved comparison & showcase in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
