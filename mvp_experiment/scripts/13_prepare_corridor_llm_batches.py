"""Step 13: Prepare strictly leak-free corridor-level LLM prompt batches (val_60 + test_100).

Uses ONLY AVI-observable fields from `val_candidates.jsonl` / `test_candidates.jsonl`,
plus static OSM named POIs snapped to each corridor micro-zone in `network.npz` and
static `tract_poi_summary.json`. Zero ground-truth stop times, tracts, or labels are read.
"""
from __future__ import annotations

import collections
import importlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.dont_write_bytecode = True
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

network = importlib.import_module("09_build_corridor_network")

MVP = network.MVP
ROOT = MVP.parent
E2E_DIR = MVP / "data" / "e2e_reconstruction"
OUT_DIR = MVP / "data" / "corridor_llm_batches"
SUBSET_100 = MVP / "data" / "batches_100" / "ground_truth.json"
TRACT_SUMMARY = MVP / "data" / "tract_poi_summary.json"

STOP_PROB_CUTOFF = 0.20
MAX_ZONES_PER_GAP = 8
VAL_SAMPLE_DAYS = 60
BATCH_SIZE = 10

DOW_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
NOISE_FCLASSES = {
    "bench", "waste_basket", "post_box", "telephone",
    "drinking_water", "toilet", "vending_any", "bus_stop",
}


def fmt_clock(minutes: float | None) -> str:
    if minutes is None:
        return "unknown"
    m = int(round(minutes))
    day_extra = m // 1440
    h, mm = divmod(m % 1440, 60)
    return f"{h:02d}:{mm:02d}" + (f"(+{day_extra}d)" if day_extra > 0 else "")


def read_dbf_fields(path: Path):
    data = path.read_bytes()
    count, header_len, row_len = struct.unpack_from("<IHH", data, 4)
    offset = 1
    fields = {}
    for position in range(32, header_len - 1, 32):
        field = data[position:position + 32]
        name = field[:11].split(b"\0", 1)[0].decode("ascii")
        fields[name] = slice(offset, offset + field[16])
        offset += field[16]
    return data, count, header_len, row_len, fields


def build_zone_named_pois(net_data) -> dict[int, dict]:
    """Snap OSM POIs with real names to corridor micro-zones using 09's exact snapping rules."""
    settings = network.literal_settings(
        SCRIPTS_DIR / "01_fetch_chicago_poi.py",
        {"BBOX", "FCLASS_TO_GROUP", "POI_GROUPS", "NOISE_FCLASSES"},
    )
    fclass_to_group = settings["FCLASS_TO_GROUP"]
    noise_fc = set(settings.get("NOISE_FCLASSES", NOISE_FCLASSES))
    x0, y0, x1, y1 = settings["BBOX"]

    node_lonlat = net_data["node_lonlat"]
    node_zone = net_data["node_zone"]
    valid_nodes = np.flatnonzero(node_zone >= 0)
    tree = cKDTree(network.unit_sphere(node_lonlat[valid_nodes]))
    radius = 2.0 * np.sin(network.SNAP_M / (2.0 * network.EARTH_M))

    zone_names: dict[int, list[tuple[int, str]]] = collections.defaultdict(list)
    zone_groups: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    group_prio = {
        "healthcare": 0, "school": 1, "meal": 2, "shop": 3,
        "social_rec": 4, "worship_civic": 5, "errand": 6, "lodging": 7, "transport": 8, "other_poi": 9,
    }

    coords = []
    meta = []

    for layer in network.POI_LAYERS:
        dbf_path = network.POI_DIR / f"{layer}.dbf"
        shp_path = network.POI_DIR / f"{layer}.shp"
        if not shp_path.exists() or not dbf_path.exists():
            continue
        data, count, header, size, fields = read_dbf_fields(dbf_path)
        fc_slice = fields["fclass"]
        name_slice = fields.get("name")
        shapes = shp_path.read_bytes()
        position = 100
        for i in range(count):
            _, words = struct.unpack_from(">ii", shapes, position)
            position += 8
            end = position + 2 * words
            kind = struct.unpack_from("<i", shapes, position)[0]
            row = header + i * size
            if kind and data[row] != 42:
                if kind == 1:
                    lon, lat = struct.unpack_from("<2d", shapes, position + 4)
                else:
                    a, b, c, d = struct.unpack_from("<4d", shapes, position + 4)
                    lon, lat = (a + c) / 2.0, (b + d) / 2.0
                if x0 <= lon <= x1 and y0 <= lat <= y1:
                    fclass = data[row + fc_slice.start:row + fc_slice.stop].decode("utf-8", errors="ignore").strip().lower()
                    if fclass not in noise_fc:
                        group = fclass_to_group.get(fclass, "other_poi")
                        name = ""
                        if name_slice is not None:
                            name = data[row + name_slice.start:row + name_slice.stop].decode("utf-8", errors="ignore").strip()
                        coords.append((lon, lat))
                        meta.append((group, fclass, name))
            position = end

    if coords:
        arr = np.asarray(coords, dtype=np.float64)
        dist, idx = tree.query(network.unit_sphere(arr), distance_upper_bound=radius, workers=4)
        for d, i, (group, fclass, name) in zip(dist, idx, meta):
            if not np.isfinite(d) or i >= len(valid_nodes):
                continue
            z = int(node_zone[valid_nodes[i]])
            if z < 0:
                continue
            zone_groups[z][group] += 1
            if name:
                zone_names[z].append((group_prio.get(group, 9), f"{name} ({fclass})"))

    out = {}
    for z, grp in zone_groups.items():
        seen = set()
        top_names = []
        for _, label in sorted(zone_names.get(z, []), key=lambda x: (x[0], x[1])):
            if label not in seen:
                seen.add(label)
                top_names.append(label)
            if len(top_names) >= 6:
                break
        out[z] = {
            "groups": dict(grp.most_common(5)),
            "named": top_names,
        }
    return out


def format_chain_payload(chain: dict, zone_pois: dict[int, dict], tract_summary: dict) -> dict | None:
    obs = chain["observables"]
    pdid = chain["person_day_id"]
    dow = int(obs.get("travel_dow") or 0)
    start_t = str(obs.get("known_day_start_tract_id") or "")
    end_t = str(obs.get("known_day_end_tract_id") or "")
    n_hits = len(obs.get("observed_edges") or [])
    obs_times = obs.get("observed_time_min") or []

    likely_gaps = []
    skeleton = []
    for gap in chain["intervals"]:
        cands = gap.get("candidates") or []
        p_stop = float(gap.get("p_has_stop") or 0.0)
        if not cands:
            continue
        t0 = gap.get("lower_min")
        t1 = gap.get("upper_min")
        span = gap.get("span_min")
        is_boundary = bool(gap.get("start_boundary") or gap.get("end_boundary"))
        if p_stop >= STOP_PROB_CUTOFF:
            skeleton.append(
                f"Gap#{gap['index']} {fmt_clock(t0)}-{fmt_clock(t1)} "
                f"(span={round(span,1) if span is not None else 'unk'}m"
                f"{', boundary' if is_boundary else ''}, p_stop={round(p_stop,2)})"
            )
            cand_items = []
            for c in cands[:MAX_ZONES_PER_GAP]:
                zid = int(c["zone_id"])
                tid = str(c["tract_id"])
                zp = zone_pois.get(zid, {})
                ts = tract_summary.get(tid, {})
                # Keep payload compact yet rich in concrete semantic cues
                cand_items.append({
                    "zone_id": zid,
                    "tract_id": tid,
                    "is_home_anchor": bool(c.get("anchor")),
                    "detour_min": round(float(c.get("detour_min", 0.0)), 1),
                    "max_dwell_min": round(float(c.get("max_dwell_min", 0.0)), 1),
                    "micro_poi_groups": zp.get("groups", {}),
                    "micro_named_pois": zp.get("named", [])[:5],
                    "tract_dom_poi": ts.get("dominant_poi_group", "none"),
                    "tract_named_pois": (ts.get("representative_pois") or [])[:4],
                })
            likely_gaps.append({
                "gap_key": f"{pdid}_{gap['index']}",
                "gap_index": int(gap["index"]),
                "is_boundary": is_boundary,
                "window": f"{fmt_clock(t0)}-{fmt_clock(t1)}",
                "span_min": round(float(span), 1) if span is not None else None,
                "min_travel_min": round(float(gap.get("min_travel_min") or 0.0), 1),
                "p_stop_prior": round(p_stop, 3),
                "candidates": cand_items,
            })

    if not likely_gaps:
        return None

    return {
        "person_day_id": pdid,
        "day_of_week": DOW_NAMES.get(dow, str(dow)),
        "is_weekend": bool(dow in (6, 7)),
        "home_start_tract": start_t,
        "home_end_tract": end_t,
        "n_avi_hits": n_hits,
        "first_avi_time": fmt_clock(obs_times[0]) if obs_times else "none",
        "last_avi_time": fmt_clock(obs_times[-1]) if obs_times else "none",
        "daily_gap_skeleton": skeleton,
        "likely_gaps": likely_gaps,
    }


def make_output_schema() -> dict:
    probs_props = {
        c: {"type": "number"}
        for c in (
            "home", "work", "school", "shop", "meal",
            "social_recreational", "escort", "healthcare", "errand_other",
        )
    }
    probs_obj = {
        "type": "object",
        "properties": probs_props,
        "required": list(probs_props.keys()),
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "gap_predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "gap_key": {"type": "string"},
                        "brief_reasoning": {"type": "string"},
                        "stop_prob_llm": {"type": "number"},
                        "gap_activity_probs": probs_obj,
                        "top_zones": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "zone_id": {"type": "integer"},
                                    "zone_prob": {"type": "number"},
                                    "activity_probs": probs_obj,
                                },
                                "required": ["zone_id", "zone_prob", "activity_probs"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["gap_key", "brief_reasoning", "stop_prob_llm", "gap_activity_probs", "top_zones"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["gap_predictions"],
        "additionalProperties": False,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    net_data = np.load(network.CACHE, allow_pickle=False)
    tract_summary = json.loads(TRACT_SUMMARY.read_text(encoding="utf-8"))
    print("[13] Snapping named OSM POIs to corridor micro-zones...")
    zone_pois = build_zone_named_pois(net_data)
    print(f"[13] Built named POI index for {len(zone_pois):,} micro-zones.")

    # Select fixed test_100 cohort
    days100 = set(json.loads(SUBSET_100.read_text(encoding="utf-8"))["selected_pdids"])
    test_chains = [json.loads(l) for l in (E2E_DIR / "test_candidates.jsonl").open(encoding="utf-8")]
    test_100 = [c for c in test_chains if c["person_day_id"] in days100]
    assert len(test_100) == 100, f"Expected 100 test chains, got {len(test_100)}"

    # Select deterministic 60-day val subset stratified across days with likely gaps (strictly observable!)
    val_chains = [json.loads(l) for l in (E2E_DIR / "val_candidates.jsonl").open(encoding="utf-8")]
    val_with_gaps = [
        c for c in val_chains
        if any((g.get("candidates") and float(g.get("p_has_stop") or 0.0) >= STOP_PROB_CUTOFF) for g in c["intervals"])
    ]
    # Evenly stride across sorted person_day_id for deterministic household diversity
    val_with_gaps.sort(key=lambda c: c["person_day_id"])
    step = len(val_with_gaps) / VAL_SAMPLE_DAYS
    val_60 = [val_with_gaps[int(round(i * step))] for i in range(VAL_SAMPLE_DAYS)]
    # Deduplicate safety
    seen_v = set()
    val_60_clean = []
    for c in val_60:
        if c["person_day_id"] not in seen_v:
            seen_v.add(c["person_day_id"])
            val_60_clean.append(c)
    val_60 = val_60_clean

    manifest = {"stop_prob_cutoff": STOP_PROB_CUTOFF, "max_zones_per_gap": MAX_ZONES_PER_GAP, "splits": {}}
    (OUT_DIR / "output_schema.json").write_text(
        json.dumps(make_output_schema(), indent=2), encoding="utf-8"
    )

    for split_name, subset in [("val60", val_60), ("test100", test_100)]:
        payloads = []
        total_gaps = 0
        for c in subset:
            p = format_chain_payload(c, zone_pois, tract_summary)
            if p is not None:
                payloads.append(p)
                total_gaps += len(p["likely_gaps"])
        batches = []
        for bi in range(0, len(payloads), BATCH_SIZE):
            chunk = payloads[bi:bi + BATCH_SIZE]
            bname = f"{split_name}_batch_{bi // BATCH_SIZE:02d}.json"
            bpath = OUT_DIR / bname
            bpath.write_text(json.dumps({"split": split_name, "chains": chunk}, ensure_ascii=False, indent=2), encoding="utf-8")
            batches.append({"file": bname, "n_chains": len(chunk), "n_gaps": sum(len(x["likely_gaps"]) for x in chunk)})
        manifest["splits"][split_name] = {
            "selected_pdids": [c["person_day_id"] for c in subset],
            "chains_with_likely_gaps": len(payloads),
            "total_likely_gaps": total_gaps,
            "batches": batches,
        }
        print(f"[13] {split_name}: {len(subset)} chains ({len(payloads)} with likely gaps), {total_gaps} gaps -> {len(batches)} batches")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[13] Saved manifest -> {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
