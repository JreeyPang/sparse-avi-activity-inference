"""Step 2: Build clean pure-auto upper-bound MVP dataset from strict_chicago_all_usable_10pct.

Key design choices (strictly leak-free):
  - Pure-auto chains only (sov / hov_2 / hov_3plus) to match vehicle AVI setting.
  - Excludes non_activity_label (-1, 11, 14, 995).
  - Splits out 'healthcare' (d_purpose in {34, 106}) from 'errand' using trip_linked.csv.
  - Maps to a clean 9-class activity taxonomy:
      home, work, school, shop, meal, social_recreational, escort, healthcare, errand_other
  - Uses ONLY strict-protocol observable spatial anchors (known_day_start_tract_id / known_day_end_tract_id),
    NOT ground-truth work_tract_id / school_tract_id!
  - Preserves original household-level train/val/test split (zero household leakage).
  - Exports:
      mvp_experiment/data/mvp_auto_activities.csv
      mvp_experiment/data/llm_test_prompts.jsonl (all non-home test stops, ready for DeepSeek/OpenAI)
"""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHICAGO_DIR = ROOT / "Chicago-Trip"
STRICT_DIR = CHICAGO_DIR / "data" / "processed_fixed" / "strict_chicago_all_usable_10pct"
MVP_DATA = ROOT / "mvp_experiment" / "data"

AUTO_MODES = {"sov", "hov_2", "hov_3plus"}
HEALTHCARE_DPURPOSE = {"34", "106"}

LABEL_MAP = {
    "home": "home",
    "work": "work",
    "work_related": "work",
    "school": "school",
    "school_related": "school",
    "shop": "shop",
    "meal": "meal",
    "social_recreational": "social_recreational",
    "escort": "escort",
    "healthcare": "healthcare",
    "errand": "errand_other",
    "other": "errand_other",
    "overnight": "errand_other",
}


def load_trip_purpose_lookup():
    path = CHICAGO_DIR / "data" / "Rawdata" / "Chicago" / "MyDailyTravelData" / "trip_linked.csv"
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["linked_trip_id"]] = (row["d_purpose"], row["d_purpose_category"])
    return lookup


def load_poi_summary():
    with open(MVP_DATA / "tract_poi_summary.json", encoding="utf-8") as f:
        return json.load(f)


def fmt_min(m):
    if m is None:
        return "unknown"
    m = int(round(m))
    h, mm = divmod(m % 1440, 60)
    return f"{h:02d}:{mm:02d}"


def build():
    purpose_lookup = load_trip_purpose_lookup()
    poi_summary = load_poi_summary()

    # First pass: count usable pure-auto days per person across all splits (for cold-start vs multi-day flag)
    person_auto_days = collections.Counter()
    # Also compute leak-free historical tract visit frequency from OTHER days of the same person (strictly excluding current day!)
    person_day_records = []

    for split in ("train", "val", "test"):
        with open(STRICT_DIR / f"masked_{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gt = rec["ground_truth_chain"]
                trips = gt.get("trips", [])
                mset = {t["mode"]["label"] for t in trips}
                if not (mset and mset <= AUTO_MODES):
                    continue
                pid = gt["person_id"]
                person_auto_days[pid] += 1
                person_day_records.append((split, rec))

    # Build per-person, per-day tract sets (excluding day_start/end home anchor) for leak-free historical familiarity
    person_tract_days = collections.defaultdict(lambda: collections.defaultdict(set))
    for split, rec in person_day_records:
        gt = rec["ground_truth_chain"]
        pid = gt["person_id"]
        pdid = gt["person_day_id"]
        for a in gt["activities"]:
            tid = str(a.get("tract_id") or "")
            if tid:
                person_tract_days[pid][tid].add(pdid)

    rows = []
    llm_prompts = []
    split_counts = collections.Counter()
    class_counts = collections.defaultdict(collections.Counter)

    for split, rec in person_day_records:
        gt = rec["ground_truth_chain"]
        pdid = gt["person_day_id"]
        pid = gt["person_id"]
        hid = gt["household_id"]
        dow = int(gt.get("travel_dow") or 0)
        is_weekend = int(dow in (6, 7))
        start_tract = str(rec.get("known_day_start_tract_id") or "")
        end_tract = str(rec.get("known_day_end_tract_id") or "")
        n_obs_edges = len(rec.get("observed_edges") or [])
        n_person_days = person_auto_days[pid]

        acts = gt["activities"]
        trips = gt["trips"]
        n_acts = len(acts)
        # Map destination_activity_index -> incoming trip; origin_activity_index -> outgoing trip
        in_trip_map = {t["destination_activity_index"]: t for t in trips}
        out_trip_map = {t["origin_activity_index"]: t for t in trips}

        # Count how many times each tract is visited within today's chain
        today_tract_counts = collections.Counter(str(a.get("tract_id") or "") for a in acts)

        # Build compact chain context string for LLM
        chain_steps_desc = []
        for a in acts:
            tid = str(a.get("tract_id") or "")
            ps = poi_summary.get(tid, {})
            dom = ps.get("dominant_poi_group", "none")
            anchor_tag = " [day-start/end anchor]" if tid in (start_tract, end_tract) else ""
            chain_steps_desc.append(
                f"Stop#{a['activity_index']} {fmt_min(a.get('start_time_min'))}-{fmt_min(a.get('end_time_min'))} "
                f"({int(round(a.get('duration_min') or 0))}min, dom_POI={dom}{anchor_tag})"
            )

        for a in acts:
            flags = a.get("quality_flags") or []
            if "non_activity_label" in flags:
                continue
            aidx = a["activity_index"]
            tid = str(a.get("tract_id") or "")
            raw_lbl = a["activity_type"]["label"]
            in_t = in_trip_map.get(aidx)
            out_t = out_trip_map.get(aidx)
            dp_code = ""
            if in_t:
                dp_code = purpose_lookup.get(in_t["linked_trip_id"], ("", ""))[0]
                if dp_code in HEALTHCARE_DPURPOSE:
                    raw_lbl = "healthcare"

            label = LABEL_MAP.get(raw_lbl)
            if label is None:
                continue

            start_min = float(a.get("start_time_min") or 0.0)
            end_min = float(a.get("end_time_min") or start_min)
            dur_min = max(float(a.get("duration_min") or (end_min - start_min)), 0.0)

            in_dist_m = float(in_t.get("distance_m") or 0.0) if in_t else 0.0
            in_dur_min = float(in_t.get("duration_min") or 0.0) if in_t else 0.0
            out_dist_m = float(out_t.get("distance_m") or 0.0) if out_t else 0.0
            out_dur_min = float(out_t.get("duration_min") or 0.0) if out_t else 0.0
            in_mode = in_t["mode"]["label"] if in_t else "none"
            out_mode = out_t["mode"]["label"] if out_t else "none"

            # Leak-free historical visits (strictly on OTHER days of this person)
            other_days_at_tract = len(person_tract_days[pid][tid] - {pdid}) if tid else 0

            is_first = int(aidx == 0)
            is_last = int(aidx == n_acts - 1)
            is_anchor_start = int(bool(tid) and tid == start_tract)
            is_anchor_end = int(bool(tid) and tid == end_tract)
            is_same_prev_tract = int(bool(in_t) and str(in_t.get("origin_tract")) == tid)
            is_same_next_tract = int(bool(out_t) and str(out_t.get("destination_tract")) == tid)
            is_short_stop_15 = int(dur_min <= 15.0)
            is_short_stop_5 = int(dur_min <= 5.0)
            is_long_stop_240 = int(dur_min >= 240.0)

            row = {
                "split": split,
                "person_day_id": pdid,
                "person_id": pid,
                "household_id": hid,
                "activity_index": aidx,
                "n_activities_in_day": n_acts,
                "rel_pos_in_day": round(aidx / max(n_acts - 1, 1), 4),
                "travel_dow": dow,
                "is_weekend": is_weekend,
                "n_person_auto_days": n_person_days,
                "is_multiday_user": int(n_person_days >= 2),
                "other_days_visited_tract": other_days_at_tract,
                "today_tract_visits": today_tract_counts[tid],
                "tract_id": tid,
                "n_obs_edges_day": n_obs_edges,
                "start_time_min": round(start_min, 2),
                "end_time_min": round(end_min, 2),
                "duration_min": round(dur_min, 2),
                "start_hour": round((start_min % 1440.0) / 60.0, 3),
                "end_hour": round((end_min % 1440.0) / 60.0, 3),
                "is_first_act": is_first,
                "is_last_act": is_last,
                "is_day_start_tract": is_anchor_start,
                "is_day_end_tract": is_anchor_end,
                "in_dist_m": round(in_dist_m, 1),
                "in_dur_min": round(in_dur_min, 2),
                "out_dist_m": round(out_dist_m, 1),
                "out_dur_min": round(out_dur_min, 2),
                "is_hov_in": int(in_mode in ("hov_2", "hov_3plus")),
                "is_hov_out": int(out_mode in ("hov_2", "hov_3plus")),
                "is_same_prev_tract": is_same_prev_tract,
                "is_same_next_tract": is_same_next_tract,
                "is_short_stop_5": is_short_stop_5,
                "is_short_stop_15": is_short_stop_15,
                "is_long_stop_240": is_long_stop_240,
                "raw_label": raw_lbl,
                "d_purpose_code": dp_code,
                "label": label,
            }
            rows.append(row)
            split_counts[split] += 1
            class_counts[split][label] += 1

            # Export structured LLM prompt payload for test-set non-anchor/non-trivial stops (and all test stops)
            if split == "test":
                ps = poi_summary.get(tid, {})
                llm_prompts.append({
                    "sample_key": f"{pdid}_{aidx}",
                    "person_day_id": pdid,
                    "activity_index": aidx,
                    "ground_truth_label": label,
                    "is_multiday_user": int(n_person_days >= 2),
                    "context": {
                        "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow - 1] if 1 <= dow <= 7 else str(dow),
                        "is_weekend": bool(is_weekend),
                        "arrival_time": fmt_min(start_min),
                        "departure_time": fmt_min(end_min),
                        "duration_min": round(dur_min, 1),
                        "position_in_chain": f"{aidx + 1}/{n_acts}",
                        "is_first_or_last": bool(is_first or is_last),
                        "is_day_start_or_end_tract": bool(is_anchor_start or is_anchor_end),
                        "incoming_trip_km": round(in_dist_m / 1000.0, 2),
                        "outgoing_trip_km": round(out_dist_m / 1000.0, 2),
                        "other_days_visited_this_tract": other_days_at_tract,
                        "today_visits_to_this_tract": today_tract_counts[tid],
                        "tract_area_km2": ps.get("area_km2"),
                        "built_up_ratio": ps.get("built_up_ratio"),
                        "poi_group_counts": ps.get("poi_group_counts", {}),
                        "top_fclasses": ps.get("top_fclasses", {}),
                        "representative_pois": ps.get("representative_pois", []),
                        "daily_chain_overview": chain_steps_desc,
                    },
                })

    out_csv = MVP_DATA / "mvp_auto_activities.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_llm = MVP_DATA / "llm_test_prompts.jsonl"
    with open(out_llm, "w", encoding="utf-8") as f:
        for item in llm_prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Dataset] Saved {len(rows)} pure-auto activities -> {out_csv}")
    print(f"[Dataset] Saved {len(llm_prompts)} test LLM prompt items -> {out_llm}")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: total={split_counts[sp]:4d} | " + ", ".join(f"{k}={v}" for k, v in class_counts[sp].most_common()))


if __name__ == "__main__":
    build()
