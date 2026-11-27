"""Step 5: Head-to-head comparison on the EXACT same 100 pure-auto test chains (499 stops, 281 non-home):
  - B1: POI+Time Rules
  - B2d: Multinomial Logit (Full)
  - B2e: Multinomial Logit (Full, balanced)
  - B2h: LightGBM (Full)
  - B2i: LightGBM (Full, balanced)
  - B3-Gemini: Gemini Flash zero-shot chain-level inference
  - B3-Codex: Codex zero-shot chain-level inference
  - B4-LogLinear Fusion (LightGBM + Gemini Prior)  [alpha=0.5, plus OOF-tuned check]
  - B4-LogLinear Fusion (LightGBM + Codex Prior)
  - B4-LogLinear Fusion (Logit + Codex/Gemini Prior)
  - B4-Chain DP/Viterbi Decoding (Markov chain transition + calibrated LLM+ML emission)

Outputs:
  mvp_experiment/results/head_to_head_100chains.json
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

ROOT = Path(__file__).resolve().parents[2]
MVP_DATA = ROOT / "mvp_experiment" / "data"
BATCH_DIR = MVP_DATA / "batches_100"
RES_DIR = ROOT / "mvp_experiment" / "results"

CLASSES = [
    "home", "work", "school", "shop", "meal",
    "social_recreational", "escort", "healthcare", "errand_other",
]
NON_HOME_CLASSES = [c for c in CLASSES if c != "home"]
C2I = {c: i for i, c in enumerate(CLASSES)}


def norm_probs(prob_dict: dict) -> np.ndarray:
    v = np.array([max(float(prob_dict.get(c, 0.0)), 1e-4) for c in CLASSES], dtype=float)
    return v / v.sum()


def evaluate_preds(y_true, y_pred, name, probs=None):
    overall_acc = accuracy_score(y_true, y_pred)
    overall_mf1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    nh = y_true != "home"
    nh_acc = accuracy_score(y_true[nh], y_pred[nh])
    nh_mf1 = f1_score(y_true[nh], y_pred[nh], labels=NON_HOME_CLASSES, average="macro", zero_division=0)
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    ll = None
    if probs is not None:
        sorted_classes = sorted(CLASSES)
        order = [CLASSES.index(c) for c in sorted_classes]
        ll = round(float(log_loss(y_true, np.clip(probs[:, order], 1e-6, 1.0), labels=sorted_classes)), 4)
    return {
        "model": name,
        "overall_acc": round(overall_acc, 4),
        "overall_macro_f1": round(overall_mf1, 4),
        "non_home_acc": round(nh_acc, 4),
        "non_home_macro_f1": round(nh_mf1, 4),
        "log_loss": ll,
        "escort_r": round(rep["escort"]["recall"], 4),
        "escort_f1": round(rep["escort"]["f1-score"], 4),
        "healthcare_r": round(rep["healthcare"]["recall"], 4),
        "healthcare_f1": round(rep["healthcare"]["f1-score"], 4),
        "school_r": round(rep["school"]["recall"], 4),
        "school_f1": round(rep["school"]["f1-score"], 4),
        "meal_f1": round(rep["meal"]["f1-score"], 4),
        "work_f1": round(rep["work"]["f1-score"], 4),
        "shop_f1": round(rep["shop"]["f1-score"], 4),
        "social_f1": round(rep["social_recreational"]["f1-score"], 4),
        "errand_f1": round(rep["errand_other"]["f1-score"], 4),
        "home_f1": round(rep["home"]["f1-score"], 4),
    }


def viterbi_decode(df_sub: pd.DataFrame, log_emit: np.ndarray, log_trans: np.ndarray, trans_weight: float = 0.5) -> np.ndarray:
    """Simple chain-level Viterbi decoding over each person-day chain."""
    preds = np.empty(len(df_sub), dtype=object)
    # Group indices by person_day_id preserving activity_index order
    for pdid, grp in df_sub.groupby("person_day_id", sort=False):
        idxs = grp.sort_values("activity_index").index.to_numpy()
        T = len(idxs)
        K = len(CLASSES)
        dp = np.full((T, K), -1e9)
        bp = np.zeros((T, K), dtype=int)
        dp[0] = log_emit[idxs[0]]
        for t in range(1, T):
            scores = dp[t - 1][:, None] + trans_weight * log_trans  # (K_prev, K_cur)
            bp[t] = scores.argmax(axis=0)
            dp[t] = scores.max(axis=0) + log_emit[idxs[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = dp[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = bp[t + 1, path[t + 1]]
        for t, orig_i in enumerate(idxs):
            preds[orig_i] = CLASSES[path[t]]
    return preds


def main():
    # Inline import helpers from 03_run_baselines.py cleanly
    import importlib.util
    spec = importlib.util.spec_from_file_location("b03", Path(__file__).parent / "03_run_baselines.py")
    b03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b03)

    acts = pd.read_csv(MVP_DATA / "mvp_auto_activities.csv", dtype={"tract_id": str, "person_day_id": str})
    pois = pd.read_csv(MVP_DATA / "tract_poi_features.csv", dtype={"tract_id": str})
    df = acts.merge(pois, on="tract_id", how="left")
    df["dominant_poi_group"] = df["dominant_poi_group"].fillna("none")
    df = b03.add_cyclical_and_derived_features(df)
    df["sample_key"] = df["person_day_id"].astype(str) + "_" + df["activity_index"].astype(str)

    gt_info = json.loads((BATCH_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    sel_pdids = set(gt_info["selected_pdids"])

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    tr_full = pd.concat([train_df, val_df], ignore_index=True)
    test100 = df[(df["split"] == "test") & (df["person_day_id"].isin(sel_pdids))].sort_values(["person_day_id", "activity_index"]).reset_index(drop=True)

    # Estimate empirical activity transition matrix on train+val
    trans = np.ones((len(CLASSES), len(CLASSES)), dtype=float) * 0.5  # Laplace smoothing
    for _, grp in tr_full.groupby("person_day_id"):
        seq = grp.sort_values("activity_index")["label"].map(C2I).to_numpy()
        for a, b in zip(seq[:-1], seq[1:]):
            trans[a, b] += 1.0
    trans = trans / trans.sum(axis=1, keepdims=True)
    log_trans = np.log(trans)

    # Feature columns (identical to 03_run_baselines.py)
    full_cols = [
        "activity_index", "n_activities_in_day", "rel_pos_in_day",
        "travel_dow", "is_weekend",
        "start_time_min", "end_time_min", "duration_min", "log_dur_min",
        "start_sin", "start_cos", "end_sin", "end_cos",
        "is_morning_peak", "is_lunch_window", "is_dinner_window", "is_school_dropoff",
        "is_first_act", "is_last_act", "is_day_start_tract", "is_day_end_tract", "is_anchor_any",
        "in_dist_m", "in_dur_min", "out_dist_m", "out_dur_min", "log_in_dist", "log_out_dist",
        "is_same_prev_tract", "is_same_next_tract",
        "is_short_stop_5", "is_short_stop_15", "is_long_stop_240",
        "today_tract_visits", "other_days_visited_tract",
        "area_km2", "built_up", "tree_cover", "grassland", "cropland", "water",
        "acs_P1_001N", "acs_H1_001N", "acs_DP03_0001E", "acs_DP03_0002E", "acs_DP03_0004E",
        "acs_DP03_0019E", "acs_DP03_0021E", "acs_DP03_0024E", "acs_DP03_0062E",
        "acs_DP04_0001E", "acs_DP04_0046E", "acs_DP04_0047E",
        "acs_pop_density", "acs_emp_ratio",
        "total_poi", "poi_density", "poi_entropy",
        "has_hospital_or_clinic", "has_school_or_k12", "has_university",
        "has_mall_or_supermarket", "n_restaurants_cafes", "n_healthcare_strict",
    ] + [c for c in df.columns if c.startswith("poi_cnt_") or c.startswith("poi_share_")]

    y_tr = tr_full["label"].values
    y_te = test100["label"].values

    results = []

    # B1 Rules
    results.append(evaluate_preds(y_te, b03.predict_rule_baseline(test100), "B1: POI+Time Rules"))

    # B2 Logit
    logit_probs = {}
    for name, cw in [("B2d: Logit (Full)", None), ("B2e: Logit (Full, balanced)", "balanced")]:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight=cw)),
        ])
        pipe.fit(tr_full[full_cols], y_tr)
        # Reorder columns to CLASSES
        col_map = [list(pipe.classes_).index(c) for c in CLASSES]
        p = pipe.predict_proba(test100[full_cols])[:, col_map]
        logit_probs[name] = p
        pred = np.array([CLASSES[i] for i in p.argmax(axis=1)])
        results.append(evaluate_preds(y_te, pred, name, probs=p))

    # B2 LightGBM
    lgb_probs = {}
    y_train_idx = train_df["label"].map(C2I).values
    y_val_idx = val_df["label"].map(C2I).values
    for name, cw in [("B2h: LightGBM (Full)", None), ("B2i: LightGBM (Full, balanced)", "balanced")]:
        clf = lgb.LGBMClassifier(
            objective="multiclass", num_class=len(CLASSES), n_estimators=500,
            learning_rate=0.05, num_leaves=31, colsample_bytree=0.8, subsample=0.8,
            min_child_samples=15, class_weight=cw, random_state=42, verbosity=-1,
        )
        clf.fit(train_df[full_cols], y_train_idx, eval_set=[(val_df[full_cols], y_val_idx)], callbacks=[lgb.early_stopping(40, verbose=False)])
        p = clf.predict_proba(test100[full_cols])
        lgb_probs[name] = p
        pred = np.array([CLASSES[i] for i in p.argmax(axis=1)])
        results.append(evaluate_preds(y_te, pred, name, probs=p))

    # Load LLM predictions
    llm_probs = {}
    for key, fname, disp in [
        ("gemini", "gemini_100chains.json", "B3a: Gemini Flash (zero-shot chain)"),
        ("codex", "codex_100chains.json", "B3b: Codex (zero-shot chain)"),
    ]:
        fpath = RES_DIR / fname
        if not fpath.exists():
            print(f"[WARN] Missing {fpath}")
            continue
        raw = json.loads(fpath.read_text(encoding="utf-8"))["predictions"]
        missing = 0
        P = np.zeros((len(test100), len(CLASSES)), dtype=float)
        preds = []
        for idx, sk in enumerate(test100["sample_key"]):
            rec = raw.get(sk)
            if rec is None:
                missing += 1
                P[idx] = np.ones(len(CLASSES)) / len(CLASSES)
                preds.append("errand_other")
            else:
                P[idx] = norm_probs(rec.get("probs", {}))
                preds.append(rec.get("predicted_label", CLASSES[P[idx].argmax()]))
        print(f"[{key}] loaded {len(test100)-missing}/{len(test100)} stops (missing={missing})")
        llm_probs[key] = P
        results.append(evaluate_preds(y_te, np.array(preds), disp, probs=P))

    # B4: Hybrid Prior Fusion (Log-linear pooling + Chain Viterbi Decoding)
    p_lgb = lgb_probs["B2h: LightGBM (Full)"]
    p_lgb_bal = lgb_probs["B2i: LightGBM (Full, balanced)"]
    p_logit = logit_probs["B2d: Logit (Full)"]

    for key, label_tag in [("gemini", "Gemini"), ("codex", "Codex")]:
        if key not in llm_probs:
            continue
        p_llm = llm_probs[key]
        # Log-linear pool with Logit (0.5 / 0.5)
        log_pool_logit = 0.5 * np.log(np.clip(p_logit, 1e-5, 1.0)) + 0.5 * np.log(np.clip(p_llm, 1e-5, 1.0))
        pp = np.exp(log_pool_logit - log_pool_logit.max(axis=1, keepdims=True))
        pp /= pp.sum(axis=1, keepdims=True)
        results.append(evaluate_preds(y_te, np.array([CLASSES[i] for i in pp.argmax(axis=1)]), f"B4: Logit + {label_tag} Prior (0.5/0.5)", probs=pp))

        # Log-linear pool with LightGBM (0.5 / 0.5)
        log_pool = 0.5 * np.log(np.clip(p_lgb, 1e-5, 1.0)) + 0.5 * np.log(np.clip(p_llm, 1e-5, 1.0))
        pp2 = np.exp(log_pool - log_pool.max(axis=1, keepdims=True))
        pp2 /= pp2.sum(axis=1, keepdims=True)
        results.append(evaluate_preds(y_te, np.array([CLASSES[i] for i in pp2.argmax(axis=1)]), f"B4: LightGBM + {label_tag} Prior (0.5/0.5)", probs=pp2))

        # Chain Viterbi Decoding on fused emission
        vit_pred = viterbi_decode(test100, log_pool, log_trans, trans_weight=0.35)
        results.append(evaluate_preds(y_te, vit_pred, f"B4+: LightGBM + {label_tag} + Chain Viterbi"))

    if "gemini" in llm_probs and "codex" in llm_probs:
        log_ens = (
            0.4 * np.log(np.clip(p_lgb, 1e-5, 1.0))
            + 0.3 * np.log(np.clip(llm_probs["gemini"], 1e-5, 1.0))
            + 0.3 * np.log(np.clip(llm_probs["codex"], 1e-5, 1.0))
        )
        pe = np.exp(log_ens - log_ens.max(axis=1, keepdims=True))
        pe /= pe.sum(axis=1, keepdims=True)
        results.append(evaluate_preds(y_te, np.array([CLASSES[i] for i in pe.argmax(axis=1)]), "B4-Ens: LGB + Gemini + Codex", probs=pe))

    (RES_DIR / "head_to_head_100chains.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== HEAD-TO-HEAD ON IDENTICAL 100 TEST CHAINS (499 stops, 281 non-home) ===")
    hdr = f"{'Model':42s} | {'Acc':5s} | {'MacF1':5s} | {'NHAcc':5s} | {'NH_F1':5s} | {'NLL':5s} | {'EscF1':5s} | {'HltR/F1':9s} | {'SchF1':5s} | {'MealF1':6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        nll = f"{r['log_loss']:5.2f}" if r["log_loss"] is not None else "  -  "
        print(
            f"{r['model']:42s} | {r['overall_acc']:5.3f} | {r['overall_macro_f1']:5.3f} | "
            f"{r['non_home_acc']:5.3f} | {r['non_home_macro_f1']:5.3f} | {nll} | "
            f"{r['escort_f1']:5.3f} | {r['healthcare_r']:.2f}/{r['healthcare_f1']:.2f} | "
            f"{r['school_f1']:5.3f} | {r['meal_f1']:6.3f}"
        )


if __name__ == "__main__":
    main()
