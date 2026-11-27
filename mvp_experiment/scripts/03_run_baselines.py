"""Step 3: Upper-bound MVP Baseline Evaluation (Fixed GT stop Tracts & time windows).

Compares:
  - Baseline 0: Majority Class
  - Baseline 1: Heuristic Rules (POI dominant category + time-of-day + duration + day anchor)
  - Baseline 2a: Multinomial Logit — Temporal + Chain Context ONLY (no spatial/POI)
  - Baseline 2b: Multinomial Logit — Temporal + Context + Existing ACS & LandCover (no OSM POI)
  - Baseline 2c: Multinomial Logit — Full (+ OSM POI features)
  - Baseline 2d: LightGBM — Existing ACS & LandCover (no OSM POI)
  - Baseline 2e: LightGBM — Full (+ OSM POI features)

Notice: Pure-auto HOV/SOV flag is intentionally EXCLUDED from main models because single AVI cameras
cannot reliably observe vehicle occupancy (HOV vs SOV); we keep strictly AVI-observable features!

Outputs:
  mvp_experiment/results/baseline_results.json
  mvp_experiment/results/baseline_summary.md
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
MVP_DATA = ROOT / "mvp_experiment" / "data"
RES_DIR = ROOT / "mvp_experiment" / "results"

CLASSES = [
    "home", "work", "school", "shop", "meal",
    "social_recreational", "escort", "healthcare", "errand_other",
]
NON_HOME_CLASSES = [c for c in CLASSES if c != "home"]


def add_cyclical_and_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_sin"] = np.sin(2 * np.pi * df["start_hour"] / 24.0)
    df["start_cos"] = np.cos(2 * np.pi * df["start_hour"] / 24.0)
    df["end_sin"] = np.sin(2 * np.pi * df["end_hour"] / 24.0)
    df["end_cos"] = np.cos(2 * np.pi * df["end_hour"] / 24.0)
    df["log_dur_min"] = np.log1p(df["duration_min"].clip(lower=0))
    df["log_in_dist"] = np.log1p(df["in_dist_m"].clip(lower=0))
    df["log_out_dist"] = np.log1p(df["out_dist_m"].clip(lower=0))
    df["is_morning_peak"] = ((df["start_hour"] >= 6.5) & (df["start_hour"] <= 9.5)).astype(int)
    df["is_lunch_window"] = ((df["start_hour"] >= 11.0) & (df["start_hour"] <= 14.0)).astype(int)
    df["is_dinner_window"] = ((df["start_hour"] >= 17.0) & (df["start_hour"] <= 20.5)).astype(int)
    df["is_school_dropoff"] = ((df["is_weekend"] == 0) & (((df["start_hour"] >= 7.0) & (df["start_hour"] <= 9.0)) | ((df["start_hour"] >= 14.5) & (df["start_hour"] <= 16.5)))).astype(int)
    df["is_anchor_any"] = ((df["is_day_start_tract"] == 1) | (df["is_day_end_tract"] == 1)).astype(int)
    return df


def predict_rule_baseline(df: pd.DataFrame) -> np.ndarray:
    """Baseline 1: POI dominant category + time-of-day + duration + day anchor heuristic rules."""
    preds = []
    for row in df.itertuples(index=False):
        # 1. Home anchor rules
        if row.is_first_act == 1 or row.is_last_act == 1:
            if row.is_day_start_tract == 1 or row.is_day_end_tract == 1:
                preds.append("home")
                continue
        if (row.is_day_start_tract == 1 and row.is_day_end_tract == 1) and row.duration_min >= 45:
            preds.append("home")
            continue

        # 2. Very short stop -> escort (pick up / drop off)
        if row.duration_min <= 10:
            preds.append("escort")
            continue

        # 3. Weekday long stay -> work or school
        if row.is_weekend == 0 and row.duration_min >= 240 and 6.0 <= row.start_hour <= 13.0:
            if row.poi_share_school > 0.25 and row.has_school_or_k12 == 1:
                preds.append("school")
            else:
                preds.append("work")
            continue

        # 4. Meal windows + dining POI
        if (11.0 <= row.start_hour <= 14.0 or 17.5 <= row.start_hour <= 20.5) and 15 <= row.duration_min <= 110 and row.poi_share_meal >= 0.20:
            preds.append("meal")
            continue

        # 5. Healthcare POI presence
        if row.dominant_poi_group == "healthcare" or (row.poi_share_healthcare >= 0.25 and 20 <= row.duration_min <= 180):
            preds.append("healthcare")
            continue

        # 6. Dominant POI mapping
        dom = row.dominant_poi_group
        if dom == "shop":
            preds.append("shop")
        elif dom == "meal":
            preds.append("meal")
        elif dom == "school":
            preds.append("school" if row.is_weekend == 0 else "social_recreational")
        elif dom in ("social_rec", "worship_civic"):
            preds.append("social_recreational")
        elif dom == "errand":
            preds.append("errand_other")
        else:
            # Fallback by duration
            if row.duration_min >= 180:
                preds.append("work" if row.is_weekend == 0 else "social_recreational")
            else:
                preds.append("shop")
    return np.array(preds)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str, multiday_mask: np.ndarray = None) -> dict:
    overall_acc = accuracy_score(y_true, y_pred)
    overall_macro_f1 = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)

    nh_mask = y_true != "home"
    nh_acc = accuracy_score(y_true[nh_mask], y_pred[nh_mask])
    nh_macro_f1 = f1_score(y_true[nh_mask], y_pred[nh_mask], labels=NON_HOME_CLASSES, average="macro", zero_division=0)

    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    per_class = {
        c: {
            "precision": round(rep[c]["precision"], 4),
            "recall": round(rep[c]["recall"], 4),
            "f1": round(rep[c]["f1-score"], 4),
            "support": int(rep[c]["support"]),
        }
        for c in CLASSES
    }

    res = {
        "model": name,
        "overall_acc": round(overall_acc, 4),
        "overall_macro_f1": round(overall_macro_f1, 4),
        "non_home_acc": round(nh_acc, 4),
        "non_home_macro_f1": round(nh_macro_f1, 4),
        "escort_recall": per_class["escort"]["recall"],
        "escort_f1": per_class["escort"]["f1"],
        "healthcare_recall": per_class["healthcare"]["recall"],
        "healthcare_f1": per_class["healthcare"]["f1"],
        "school_recall": per_class["school"]["recall"],
        "school_f1": per_class["school"]["f1"],
        "per_class": per_class,
    }
    if multiday_mask is not None:
        for tag, m in [("cold_start_1day", (~multiday_mask) & nh_mask), ("multiday_2plus", multiday_mask & nh_mask)]:
            res[f"{tag}_nh_acc"] = round(accuracy_score(y_true[m], y_pred[m]), 4)
            res[f"{tag}_nh_macro_f1"] = round(f1_score(y_true[m], y_pred[m], labels=NON_HOME_CLASSES, average="macro", zero_division=0), 4)
    return res


def main():
    acts = pd.read_csv(MVP_DATA / "mvp_auto_activities.csv", dtype={"tract_id": str})
    pois = pd.read_csv(MVP_DATA / "tract_poi_features.csv", dtype={"tract_id": str})
    df = acts.merge(pois, on="tract_id", how="left")
    df["dominant_poi_group"] = df["dominant_poi_group"].fillna("none")
    df = add_cyclical_and_derived_features(df)

    # Feature sets (strictly AVI-observable: NO HOV/SOV occupancy!)
    temporal_context_cols = [
        "activity_index", "n_activities_in_day", "rel_pos_in_day",
        "travel_dow", "is_weekend",
        "start_time_min", "end_time_min", "duration_min", "log_dur_min",
        "start_sin", "start_cos", "end_sin", "end_cos",
        "is_morning_peak", "is_lunch_window", "is_dinner_window", "is_school_dropoff",
        "is_first_act", "is_last_act", "is_day_start_tract", "is_day_end_tract", "is_anchor_any",
        "in_dist_m", "in_dur_min", "out_dist_m", "out_dur_min", "log_in_dist", "log_out_dist",
        "is_same_prev_tract", "is_same_next_tract",
        "is_short_stop_5", "is_short_stop_15", "is_long_stop_240",
        "today_tract_visits",
    ]
    history_cols = ["other_days_visited_tract"]
    existing_spatial_cols = [
        "area_km2", "built_up", "tree_cover", "grassland", "cropland", "water",
        "acs_P1_001N", "acs_H1_001N", "acs_DP03_0001E", "acs_DP03_0002E", "acs_DP03_0004E",
        "acs_DP03_0019E", "acs_DP03_0021E", "acs_DP03_0024E", "acs_DP03_0062E",
        "acs_DP04_0001E", "acs_DP04_0046E", "acs_DP04_0047E",
        "acs_pop_density", "acs_emp_ratio",
    ]
    osm_poi_cols = [
        "total_poi", "poi_density", "poi_entropy",
        "has_hospital_or_clinic", "has_school_or_k12", "has_university",
        "has_mall_or_supermarket", "n_restaurants_cafes", "n_healthcare_strict",
    ] + [c for c in df.columns if c.startswith("poi_cnt_") or c.startswith("poi_share_")]

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # Combine train+val for final test evaluation (val used only for sanity/hyperparams)
    tr_full = pd.concat([train_df, val_df], ignore_index=True)
    y_tr = tr_full["label"].values
    y_te = test_df["label"].values
    md_mask = (test_df["is_multiday_user"] == 1).values

    results = []

    # Baseline 0: Majority
    results.append(evaluate(y_te, np.full_like(y_te, "home"), "B0: Majority (home)", md_mask))

    # Baseline 1: Heuristic Rules
    y_rule = predict_rule_baseline(test_df)
    results.append(evaluate(y_te, y_rule, "B1: POI+Time Rules", md_mask))

    # Baseline 2: Multinomial Logit ablation ladder
    logit_configs = [
        ("B2a: Logit (Time+Chain only)", temporal_context_cols),
        ("B2b: Logit (+ACS & LandCover, no POI)", temporal_context_cols + existing_spatial_cols),
        ("B2c: Logit (+OSM POI, no history)", temporal_context_cols + existing_spatial_cols + osm_poi_cols),
        ("B2d: Logit (+OSM POI + History)", temporal_context_cols + existing_spatial_cols + osm_poi_cols + history_cols),
        ("B2e: Logit (Full, class_weight=balanced)", temporal_context_cols + existing_spatial_cols + osm_poi_cols + history_cols),
    ]
    for name, cols in logit_configs:
        cw = "balanced" if "balanced" in name else None
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight=cw)),
        ])
        pipe.fit(tr_full[cols], y_tr)
        pred = pipe.predict(test_df[cols])
        results.append(evaluate(y_te, pred, name, md_mask))

    # Baseline 2-GBDT: LightGBM ablation ladder
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_label = {i: c for i, c in enumerate(CLASSES)}
    y_train_idx = train_df["label"].map(label_to_idx).values
    y_val_idx = val_df["label"].map(label_to_idx).values

    lgb_configs = [
        ("B2f: LightGBM (Time+Chain only)", temporal_context_cols, None),
        ("B2g: LightGBM (+ACS & LandCover, no POI)", temporal_context_cols + existing_spatial_cols, None),
        ("B2h: LightGBM (+OSM POI + History)", temporal_context_cols + existing_spatial_cols + osm_poi_cols + history_cols, None),
        ("B2i: LightGBM (Full, balanced)", temporal_context_cols + existing_spatial_cols + osm_poi_cols + history_cols, "balanced"),
    ]
    top_lgb_fi = None
    for name, cols, cw in lgb_configs:
        clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASSES),
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            colsample_bytree=0.8,
            subsample=0.8,
            min_child_samples=15,
            class_weight=cw,
            random_state=42,
            verbosity=-1,
        )
        clf.fit(
            train_df[cols], y_train_idx,
            eval_set=[(val_df[cols], y_val_idx)],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred_idx = clf.predict(test_df[cols])
        pred = np.array([idx_to_label[int(i)] for i in pred_idx])
        results.append(evaluate(y_te, pred, name, md_mask))
        if name == "B2h: LightGBM (+OSM POI + History)":
            fi = sorted(zip(cols, clf.feature_importances_), key=lambda x: -x[1])
            top_lgb_fi = [(k, int(v)) for k, v in fi[:20]]

    RES_DIR.mkdir(parents=True, exist_ok=True)
    with open(RES_DIR / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "top_lgb_feature_importance": top_lgb_fi}, f, ensure_ascii=False, indent=2)

    # Print formatted table
    print("\n=== UPPER-BOUND MVP BASELINE RESULTS (Pure-Auto Test Set: 324 chains, 1621 acts, 929 non-home acts) ===")
    hdr = f"{'Model':42s} | {'Acc':6s} | {'MacF1':6s} | {'NH_Acc':6s} | {'NH_F1':6s} | {'EscR/F1':11s} | {'HltR/F1':11s} | {'SchR/F1':11s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['model']:42s} | {r['overall_acc']:6.3f} | {r['overall_macro_f1']:6.3f} | "
            f"{r['non_home_acc']:6.3f} | {r['non_home_macro_f1']:6.3f} | "
            f"{r['escort_recall']:.2f}/{r['escort_f1']:.2f}   | "
            f"{r['healthcare_recall']:.2f}/{r['healthcare_f1']:.2f}   | "
            f"{r['school_recall']:.2f}/{r['school_f1']:.2f}"
        )

    print("\n--- Cold-Start (1-day) vs Multi-Day (2+ days) Non-Home Macro-F1 ---")
    for r in results[1:]:
        print(
            f"{r['model']:42s} | 1-day NH_Acc={r['cold_start_1day_nh_acc']:.3f} NH_F1={r['cold_start_1day_nh_macro_f1']:.3f} "
            f"| 2+day NH_Acc={r['multiday_2plus_nh_acc']:.3f} NH_F1={r['multiday_2plus_nh_macro_f1']:.3f}"
        )

    print("\n--- Best LightGBM (B2h) Per-Class Breakdown ---")
    b2h = [r for r in results if r["model"].startswith("B2h:")][0]
    for c in CLASSES:
        pc = b2h["per_class"][c]
        print(f"  {c:20s}: P={pc['precision']:.3f}  R={pc['recall']:.3f}  F1={pc['f1']:.3f}  (n={pc['support']})")

    print("\n--- Top 15 Feature Importances (LightGBM B2h) ---")
    for k, v in top_lgb_fi[:15]:
        print(f"  {k:28s}: {v}")


if __name__ == "__main__":
    main()
