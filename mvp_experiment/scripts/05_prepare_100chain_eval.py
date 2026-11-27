"""Prepare 100 pure-auto test chains into 10 leak-free JSON batches (10 chains/batch)
for head-to-head Gemini vs Codex vs Logit vs LightGBM evaluation.

Guarantees:
  - Input batches contain ZERO ground-truth labels.
  - Every chain is presented as a full daily sequence so LLMs can use chain-level
    common-sense constraints (e.g., escort -> work -> escort, single primary work, meal timing).
  - Also records exact sample_keys so ML baselines are evaluated on the IDENTICAL 100 chains.
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MVP_DATA = ROOT / "mvp_experiment" / "data"
BATCH_DIR = MVP_DATA / "batches_100"

BATCH_SIZE = 10
N_CHAINS = 100


def main():
    prompts = [json.loads(line) for line in open(MVP_DATA / "llm_test_prompts.jsonl", encoding="utf-8")]
    chains = collections.defaultdict(list)
    for item in prompts:
        chains[item["person_day_id"]].append(item)

    # Sort activities within each chain
    for pdid in chains:
        chains[pdid].sort(key=lambda x: x["activity_index"])

    all_pdids = sorted(chains.keys())
    rng = np.random.default_rng(42)
    selected_pdids = sorted(rng.choice(all_pdids, size=min(N_CHAINS, len(all_pdids)), replace=False).tolist())

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    gt_records = []
    cls_cnt = collections.Counter()

    for b_idx in range(0, len(selected_pdids), BATCH_SIZE):
        sub_pdids = selected_pdids[b_idx : b_idx + BATCH_SIZE]
        batch_chains = []
        for pdid in sub_pdids:
            items = chains[pdid]
            c0 = items[0]["context"]
            stops = []
            for it in items:
                ctx = it["context"]
                gt_records.append({
                    "sample_key": it["sample_key"],
                    "person_day_id": pdid,
                    "activity_index": it["activity_index"],
                    "ground_truth_label": it["ground_truth_label"],
                    "is_multiday_user": it["is_multiday_user"],
                })
                cls_cnt[it["ground_truth_label"]] += 1
                # Compact, leak-free stop dict
                stops.append({
                    "sample_key": it["sample_key"],
                    "stop_idx": it["activity_index"],
                    "arrival": ctx["arrival_time"],
                    "departure": ctx["departure_time"],
                    "duration_min": ctx["duration_min"],
                    "is_first_or_last": ctx["is_first_or_last"],
                    "is_day_start_or_end_tract": ctx["is_day_start_or_end_tract"],
                    "in_km": ctx["incoming_trip_km"],
                    "out_km": ctx["outgoing_trip_km"],
                    "other_days_at_tract": ctx["other_days_visited_this_tract"],
                    "today_visits_at_tract": ctx["today_visits_to_this_tract"],
                    "tract_area_km2": ctx["tract_area_km2"],
                    "built_up": ctx["built_up_ratio"],
                    "poi_counts": ctx["poi_group_counts"],
                    "top_pois": ctx["representative_pois"][:10],
                })
            batch_chains.append({
                "person_day_id": pdid,
                "day_of_week": c0["day_of_week"],
                "is_weekend": c0["is_weekend"],
                "n_stops": len(stops),
                "stops": stops,
            })
        out_path = BATCH_DIR / f"batch_{b_idx // BATCH_SIZE:02d}.json"
        out_path.write_text(json.dumps(batch_chains, ensure_ascii=False, indent=2), encoding="utf-8")

    (BATCH_DIR / "ground_truth.json").write_text(
        json.dumps({"selected_pdids": selected_pdids, "records": gt_records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    nh_total = sum(v for k, v in cls_cnt.items() if k != "home")
    print(f"Prepared {len(selected_pdids)} chains in {len(selected_pdids) // BATCH_SIZE} batches -> {BATCH_DIR}")
    print(f"Total stops={len(gt_records)}, non-home stops={nh_total}")
    print("Class counts:", dict(cls_cnt.most_common()))


if __name__ == "__main__":
    main()
