"""Repair the processed Chicago activity-chain datasets without touching the originals.

Reads   data/processed/<name>  or  data/processed_strict/<name>
Writes  data/processed_fixed/<name>  (same file layout, vocab files copied unchanged)

Fixes (see data/processed_fixed/FIXES.md):
  1. Activity labels: code 7 = shop, 8 = meal (were swapped).
  2. Mode labels: re-labelled from the 2024-2025 CMAP HTS codebook (linked_trip_mode).
  3. Times: monotonic minute fields added next to every HH:MM:SS list
     (edge_time_min / target_edge_time_min / observed_time_min, minutes since
     travel_date 00:00, may exceed 1440); last activity no longer ends before it starts.
  4. Consecutive duplicate edges inside the same trip are collapsed (first entry kept);
     repeats at trip boundaries are left alone. Observed edges are mapped onto the
     collapsed target sequence and de-duplicated the same way.
  5. Quality flags added (labels and observations are kept, not dropped).
Token sequences (activity / mode / tract / duration tokens) are code based and unchanged.

Usage: python tools/fix_processed.py [--root .]
"""
import argparse
import collections
import json
import shutil
from pathlib import Path

ACTIVITY_LABELS = {
    "-1": "not_imputable", "1": "home", "2": "work", "3": "work_related", "4": "school",
    "5": "school_related", "6": "escort", "7": "shop", "8": "meal", "9": "social_recreational",
    "10": "errand", "11": "change_mode", "12": "overnight", "13": "other", "14": "loop",
    "995": "missing_response",
}
MODE_LABELS = {
    "-1": "not_imputable", "1": "school_bus", "2": "drive_to_ferry", "3": "drive_to_rail",
    "4": "drive_to_bus", "5": "walk_to_ferry", "6": "walk_to_rail", "7": "walk_to_bus",
    "8": "hov_3plus", "9": "hov_2", "10": "sov", "11": "bike", "12": "micromobility",
    "13": "taxi", "14": "tnc", "15": "walk", "16": "long_distance", "17": "other",
    "18": "transit", "995": "missing_response",
}
NON_ACTIVITY_CODES = {"-1", "11", "14", "995"}
OVERNIGHT_CODE = "12"
DATASETS = [
    ("processed", "chicago_all_usable"),
    ("processed", "chicago_all_usable_10pct"),
    ("processed", "chicago_all_usable_5pct"),
    ("processed_strict", "chicago_all_usable_10pct"),
]
WRAP_THRESHOLD = 720  # a drop of more than 12 h between consecutive stamps = crossed midnight


def hms_to_min(s):
    h, m, sec = s.split(":")
    return int(h) * 60 + int(m) + float(sec) / 60


def unwrap(stamps, start_hint=None):
    """HH:MM:SS strings (wrapping at 24:00) -> monotonic minutes since the travel date's 00:00."""
    out, offset, prev = [], 0.0, None
    for s in stamps:
        v = hms_to_min(s)
        if start_hint is not None and prev is None and start_hint - v > WRAP_THRESHOLD:
            offset = 1440.0 * round((start_hint - v) / 1440.0)
        if prev is not None and (v + offset) < prev - WRAP_THRESHOLD:
            offset += 1440.0
        cur = v + offset
        out.append(round(cur, 4))
        prev = cur
    return out


def collapse_runs(keys, group):
    """Indices to keep: drop i when keys[i]==keys[i-1] and group[i]==group[i-1]."""
    return [i for i in range(len(keys)) if i == 0 or not (keys[i] == keys[i - 1] and group[i] == group[i - 1])]


def fix_chain(chain, stats):
    acts = chain.get("activities", [])
    for i, a in enumerate(acts):
        t = a["activity_type"]
        new = ACTIVITY_LABELS.get(t["code"], t["label"])
        if new != t["label"]:
            stats["activity_label_fixed"] += 1
            t["label"] = new
        flags = []
        if t["code"] in NON_ACTIVITY_CODES:
            flags.append("non_activity_label")
        if t["code"] == OVERNIGHT_CODE and 0 < i < len(acts) - 1 and (a.get("duration_min") or 0) < 240:
            flags.append("overnight_short_midchain")
        if a.get("end_time_min") is not None and a.get("start_time_min") is not None and a["end_time_min"] < a["start_time_min"]:
            a["end_time_min"] = a["start_time_min"]
            a["end_time"] = a["start_time"]
            a["duration_min"] = 0.0
            flags.append("day_extends_past_midnight")
            stats["activity_end_before_start_fixed"] += 1
        if flags:
            a["quality_flags"] = flags
            for f in flags:
                stats[f"activity_flag:{f}"] += 1
    for t in chain.get("trips", []):
        m = t["mode"]
        new = MODE_LABELS.get(m["code"], m["label"])
        if new != m["label"]:
            stats["mode_label_fixed"] += 1
            m["label"] = new
        if t.get("edge_timestamps"):
            t["edge_time_min"] = unwrap(t["edge_timestamps"], start_hint=t.get("departure_time_min"))
            if t["edge_timestamps"] != sorted(t["edge_timestamps"]):
                stats["trip_edge_times_unwrapped"] += 1
        seq = t.get("edge_sequence") or []
        if seq:
            keep = collapse_runs(seq, [0] * len(seq))
            if len(keep) != len(seq):
                stats["trip_dup_edges_removed"] += len(seq) - len(keep)
                for k in ("edge_sequence", "raw_edge_sequence", "edge_timestamps", "edge_modes", "edge_time_min"):
                    if isinstance(t.get(k), list) and len(t[k]) == len(seq):
                        t[k] = [t[k][i] for i in keep]
    return chain


def chain_flags(chain):
    flags = set()
    for a in chain.get("activities", []):
        flags.update(a.get("quality_flags", []))
    return sorted(flags)


def unwrap_by_trip(stamps, trip_idx, departures):
    """Unwrap each trip's stamps anchored on that trip's continuous departure_time_min."""
    if not departures:
        return unwrap(stamps)
    out, start = [], 0
    while start < len(stamps):
        end = start
        while end < len(stamps) and trip_idx[end] == trip_idx[start]:
            end += 1
        k = trip_idx[start]
        hint = departures[k] if 0 <= k < len(departures) else None
        out.extend(unwrap(stamps[start:end], start_hint=hint))
        start = end
    return out


def fix_target(rec, stats, departures=None):
    """Collapse within-trip duplicate target edges; return old->new index map."""
    te, ti, tt = rec["target_edge_sequence"], rec["target_trip_index_sequence"], rec["target_edge_times"]
    keep = collapse_runs(te, ti)
    new_index, j = [], -1
    keep_set = set(keep)
    for i in range(len(te)):
        if i in keep_set:
            j += 1
        new_index.append(j)
    if len(keep) != len(te):
        stats["target_dup_edges_removed"] += len(te) - len(keep)
        stats["samples_with_target_dups"] += 1
    rec["target_edge_sequence"] = [te[i] for i in keep]
    rec["target_trip_index_sequence"] = [ti[i] for i in keep]
    rec["target_edge_times"] = [tt[i] for i in keep]
    rec["target_edge_time_min"] = unwrap_by_trip(rec["target_edge_times"], rec["target_trip_index_sequence"], departures)
    if tt != sorted(tt):
        stats["samples_target_times_unwrapped"] += 1
    return te, tt, new_index


def fix_observed(rec, old_te, old_tt, new_index, stats):
    oe, ot = rec["observed_edges"], rec["observed_times"]
    pos, j = [], 0
    for e, t in zip(oe, ot):
        while j < len(old_te) and not (old_te[j] == e and old_tt[j] == t):
            j += 1
        if j == len(old_te):
            stats["observed_not_matched_to_target"] += 1
            pos = None
            break
        pos.append(new_index[j])
        j += 1
    if pos is None:  # keep as is if alignment failed (reported in stats)
        rec["observed_time_min"] = unwrap(ot)
        return
    keep = [k for k in range(len(pos)) if k == 0 or pos[k] != pos[k - 1]]
    if len(keep) != len(oe):
        stats["observed_dup_edges_removed"] += len(oe) - len(keep)
        stats["samples_with_observed_dups"] += 1
    rec["observed_edges"] = [oe[k] for k in keep]
    rec["observed_times"] = [ot[k] for k in keep]
    tmin = rec["target_edge_time_min"]
    rec["observed_time_min"] = [tmin[pos[k]] for k in keep]
    if ot != sorted(ot):
        stats["samples_observed_times_unwrapped"] += 1


def edge_summary(values):
    if not values:
        return {"max": 0, "mean": 0.0, "min": 0, "total": 0}
    return {"max": max(values), "mean": sum(values) / len(values), "min": min(values), "total": sum(values)}


def process(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    stats = collections.Counter()
    handled = {"chain_dict.jsonl", "masked_train.jsonl", "masked_val.jsonl", "masked_test.jsonl",
               "truth_paths.jsonl", "invalid_chain_samples.jsonl", "data_report.json"}
    for f in src.iterdir():
        if f.is_file() and f.name not in handled:
            shutil.copy2(f, dst / f.name)

    act_counts, mode_counts = collections.Counter(), collections.Counter()
    departures = {}  # person_day_id -> continuous departure_time_min per trip index
    with open(src / "chain_dict.jsonl") as fi, open(dst / "chain_dict.jsonl", "w") as fo:
        for line in fi:
            c = fix_chain(json.loads(line), stats)
            c["quality_flags"] = chain_flags(c)
            departures[c["person_day_id"]] = [t.get("departure_time_min") for t in c["trips"]]
            for a in c["activities"]:
                act_counts[a["activity_type"]["label"]] += 1
            for t in c["trips"]:
                mode_counts[t["mode"]["label"]] += 1
            fo.write(json.dumps(c, ensure_ascii=False) + "\n")

    obs_summary, tgt_summary = {}, {}
    for split in ("train", "val", "test"):
        obs_n, tgt_n = [], []
        with open(src / f"masked_{split}.jsonl") as fi, open(dst / f"masked_{split}.jsonl", "w") as fo:
            for line in fi:
                r = json.loads(line)
                gt = r.get("ground_truth_chain")
                if isinstance(gt, dict):
                    fix_chain(gt, collections.Counter())
                    r["quality_flags"] = chain_flags(gt)
                old_te, old_tt, idx = fix_target(r, stats, departures.get(r["person_day_id"]))
                fix_observed(r, old_te, old_tt, idx, stats)
                n = len(r["observed_edges"])
                if n == 0:
                    r.setdefault("quality_flags", []).append("no_observed_edge")
                elif n == 1:
                    r.setdefault("quality_flags", []).append("single_observed_edge")
                obs_n.append(n)
                tgt_n.append(len(r["target_edge_sequence"]))
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        obs_summary[split], tgt_summary[split] = edge_summary(obs_n), edge_summary(tgt_n)

    with open(src / "truth_paths.jsonl") as fi, open(dst / "truth_paths.jsonl", "w") as fo:
        for line in fi:
            r = json.loads(line)
            fix_target(r, collections.Counter(), departures.get(r["person_day_id"]))
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(src / "invalid_chain_samples.jsonl") as fi, open(dst / "invalid_chain_samples.jsonl", "w") as fo:
        for line in fi:
            r = json.loads(line)
            if isinstance(r.get("chain"), dict):
                fix_chain(r["chain"], collections.Counter())
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = json.loads((src / "data_report.json").read_text())
    report["activity_label_counts"] = dict(sorted(act_counts.items()))
    report["mode_label_counts"] = dict(sorted(mode_counts.items()))
    report["observed_edge_count_summary"] = obs_summary
    report["target_edge_count_summary"] = tgt_summary
    report["source_dir_before_fix"] = str(src)
    report["output_dir"] = str(dst)
    report["fix_script"] = "tools/fix_processed.py"
    report["fix_stats"] = dict(sorted(stats.items()))
    (dst / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root) / "data"
    for parent, name in DATASETS:
        src = root / parent / name
        dst = root / "processed_fixed" / (name if parent == "processed" else f"strict_{name}")
        stats = process(src, dst)
        print(f"== {src} -> {dst}")
        for k, v in sorted(stats.items()):
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
