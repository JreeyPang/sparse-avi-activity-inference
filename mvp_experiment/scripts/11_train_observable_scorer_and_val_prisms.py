"""Fit split-specific observable scorers and cache reduced Chicago AVI prisms."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import csv
import importlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import re
import sys
from time import perf_counter

# Set before importing libraries and forking workers.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

prism = importlib.import_module("10_build_reduced_prism_graphs")
network = prism.network
CLASSES = tuple(network.literal_settings(network.SCRIPTS / "03_run_baselines.py", {"CLASSES"})["CLASSES"])
CLASS_ID = {label: i for i, label in enumerate(CLASSES)}
DATA = network.MVP / "data"
CACHE = DATA / "e2e_reconstruction"
MODEL_FILE = CACHE / "observable_models.pkl"
ACTIVITIES = DATA / "mvp_auto_activities.csv"
TRACT_FEATURES = DATA / "tract_poi_features.csv"
CALENDAR = network.RAW / "MyDailyTravelData" / "day.csv"
SPLIT = network.STRICT / "household_split.json"
OBSERVABLES = (*prism.OBSERVABLES, "travel_dow")
EXPECTED = {"train": 1351, "val": 276, "test": 324}
MIN_DWELL = 1.5
TOP_B = 20
TRACT_CAP = 2
BOUNDARY_ACCESS = 15.0
ETA = 0.5
WORKERS = min(8, os.cpu_count() or 1)
CACHE_VERSION = 2
STOP_FOLDS = 5
SPATIAL = (
    "area_km2", "built_up", "tree_cover", "grassland", "cropland", "water",
    "acs_P1_001N", "acs_H1_001N", "acs_DP03_0001E", "acs_DP03_0002E",
    "acs_DP03_0004E", "acs_DP03_0019E", "acs_DP03_0021E", "acs_DP03_0024E",
    "acs_DP03_0062E", "acs_DP04_0001E", "acs_DP04_0046E", "acs_DP04_0047E",
    "acs_pop_density", "acs_emp_ratio",
)
ACTIVITY_FEATURES = (
    "est_start_sin", "est_start_cos", "est_end_sin", "est_end_cos",
    "is_weekend", "rel_pos_in_day", "log1p_max_dwell_min", "is_day_start_tract",
    "is_day_end_tract", "log1p_detour_min", "log1p_t_in_min", "log1p_t_out_min",
    "log1p_shortest_leg_chord_m_in", "log1p_shortest_leg_chord_m_out",
    "is_start_boundary", "is_end_boundary",
)
STOP_FEATURES = (
    "log1p_span_min", "log1p_min_travel_min", "log1p_slack_min",
    "log1p_best_feasible_dwell_min", "log1p_best_detour_min", "is_start_boundary",
    "is_end_boundary", "log1p_left_chord_m", "log1p_right_chord_m",
)
LABEL_COLUMNS = ("split", "person_day_id", "household_id", "activity_index", "tract_id",
                 "label", "start_time_min", "duration_min", "is_first_act", "is_last_act")


def candidate_file(split):
    return CACHE / f"{split}_candidates.jsonl"


def signature():
    inputs = [network.CACHE, network.COHORT, ACTIVITIES, TRACT_FEATURES, CALENDAR, SPLIT,
              Path(__file__), *(network.STRICT / f"masked_{s}.jsonl" for s in EXPECTED),
              *(network.SCRIPTS / f for f in ("09_build_corridor_network.py",
                                             "10_build_reduced_prism_graphs.py",
                                             "03_run_baselines.py"))]
    return {"version": CACHE_VERSION, "inputs": {str(p): network.file_signature(p) for p in inputs}}


def label_rows(split):
    """Read only supervised fields from one split of the canonical step-02 CSV."""
    result = defaultdict(dict)
    households = set()
    with ACTIVITIES.open() as stream:
        columns = next(csv.reader([stream.readline()]))
        if columns[0] != "split":
            raise ValueError("Expected the step 02 split in the first CSV column")
        # The step 02 CSV has one scalar-only activity per line. Filter the
        # leading split before decoding any supervised fields.
        lines = (line for line in stream if line.startswith(f"{split},"))
        for row in csv.DictReader(lines, fieldnames=columns):
            key = row["person_day_id"]
            index = int(row["activity_index"])
            if index in result[key]:
                raise ValueError(f"Duplicate activity {key}/{index}")
            result[key][index] = {k: row[k] for k in LABEL_COLUMNS}
            households.add(row["household_id"])
    return dict(result), households


PUBLIC_KEYS = frozenset(("person_day_id", *prism.OBSERVABLES))
WHITESPACE = re.compile(r"\s*")
JSON_TOKEN = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|[\[\]{}]')
VALUE_END = re.compile(r"[,}]")


def public_test_rows(path):
    """Decode only public top-level values in the mixed public/truth JSONL."""
    decode = json.JSONDecoder().raw_decode
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            position = WHITESPACE.match(line).end() + 1  # opening object
            record = {}
            while len(record) < len(PUBLIC_KEYS):
                key, position = decode(line, WHITESPACE.match(line, position).end())
                position = WHITESPACE.match(line, WHITESPACE.match(line, position).end() + 1).end()
                char = line[position]
                if key in PUBLIC_KEYS:
                    record[key], position = decode(line, position)
                elif char in "[{":
                    # Walk the untouched JSON text past the opaque subtree.
                    # String tokens can contain brackets without changing depth.
                    depth = 0
                    for token in JSON_TOKEN.finditer(line, position):
                        literal = token.group()[0]
                        depth += int(literal in "[{") - int(literal in "]}")
                        if depth == 0:
                            position = token.end()
                            break
                    else:
                        raise ValueError("Unclosed JSON value in masked test")
                elif char == '"':
                    position = JSON_TOKEN.match(line, position).end()
                else:
                    position = VALUE_END.search(line, position).start()
                position = WHITESPACE.match(line, position).end()
                if line[position] == "}":
                    break
                if line[position] != ",":
                    raise ValueError("Invalid field separator in masked test")
                position += 1
            if record.keys() != PUBLIC_KEYS:
                raise ValueError("Missing public fields in masked test")
            yield record


def read_inputs(split, ids, calendar):
    """Project only allowed covariates. IDs are keys, never model features."""
    records = []
    path = network.STRICT / f"masked_{split}.jsonl"
    public = public_test_rows(path) if split == "test" else network.json_lines(path)
    for record in public:
        person_day = record["person_day_id"]
        if person_day in ids:
            observed = {field: record[field] for field in prism.OBSERVABLES}
            observed["travel_dow"] = calendar[person_day]
            records.append((person_day, observed))
    if len(records) != EXPECTED[split] or {key for key, _ in records} != set(ids):
        raise ValueError(f"Unexpected or duplicate {split} cohort")
    return records


def read_truth(record, rows):
    """Match canonical intermediate activities using step 10's AVI tie-break."""
    positions = prism.align_observations(record)
    edge_trips = record["target_trip_index_sequence"]
    hit_trips = [edge_trips[i] for i in positions] if positions is not None else None
    truth = record["ground_truth_chain"]
    incoming = {t["destination_activity_index"]: t["trip_index"] for t in truth["trips"]}
    outgoing = {t["origin_activity_index"]: t["trip_index"] for t in truth["trips"]}
    activities = truth["activities"]
    surveyed = []
    for index, act in enumerate(activities):
        if act["activity_index"] != index:
            raise ValueError(f"Unexpected activity order for {record['person_day_id']}")
        row = rows.get(index)
        tract = str(act["tract_id"] or "")
        if row is not None and row["tract_id"] != tract:
            raise ValueError(f"Step 02 tract mismatch for {record['person_day_id']}/{index}")
        surveyed.append({
            "activity_index": index, "interval": None, "tract_id": tract,
            "activity": row["label"] if row is not None and row["label"] in CLASS_ID else None,
            "arrival_min": act["start_time_min"], "departure_min": act["end_time_min"],
            "duration_min": act["duration_min"],
        })
    # Anchor positions are based on the raw chain, before any label filtering.
    start, end = surveyed[0], surveyed[-1]
    stops = []
    for stop in surveyed[1:-1]:
        if stop["activity"] is None:
            continue
        index, arrival, departure = stop["activity_index"], stop["arrival_min"], stop["departure_min"]
        gap, match = prism.match_stop(arrival, departure, record["observed_time_min"], hit_trips,
                                    incoming.get(index), outgoing.get(index))
        stop.update(interval=gap, interval_match=match, duration_min=(
            max(0.0, departure - arrival) if arrival is not None and departure is not None else None
        ))
        stops.append(stop)
    return {
        "truth_start": start, "truth_end": end, "intermediate_stops": stops,
        "n_dropped_intermediate_activities": len(surveyed[1:-1]) - len(stops),
        "observed_positions": positions,
    }


def supervised_truth(split, rows):
    if split not in ("train", "val"):
        raise ValueError("Held-out truth belongs to stage 12 evaluate()")
    truth = {}
    for record in network.json_lines(network.STRICT / f"masked_{split}.jsonl"):
        key = record["person_day_id"]
        if key in rows:
            truth[key] = read_truth(record, rows[key])["intermediate_stops"]
    return truth


class CandidateNetwork(prism.PrismNetwork):
    @lru_cache(maxsize=32)
    def fine_window(self, x0, y0, x1, y1):
        low, high = 1000.0 * np.array([[x0, y0], [x1, y1]])
        ids = self.grid.query(low, high)
        ids = np.sort(ids[np.all((self.xy[ids] >= low) & (self.xy[ids] <= high), axis=1)])
        return ids, self.graph[ids, :][:, ids].tocsr()

    def window(self, left, right, budget):
        if budget > prism.MAX_LOCAL_BUDGET_MIN:
            return None, self.graph, left, right
        l_min, l_max = self.xy[left].min(axis=0), self.xy[left].max(axis=0)
        r_min, r_max = self.xy[right].min(axis=0), self.xy[right].max(axis=0)
        margin = budget * self.axis_speed + 1e-5
        # Sum of the two absolute axis displacements via any feasible node
        # cannot exceed the axis speed bound times the travel budget.
        low = np.minimum((l_min + r_min - margin) / 2, np.minimum(l_min, r_min))
        high = np.maximum((l_max + r_max + margin) / 2, np.maximum(l_max, r_max))
        low, high = np.floor(low / 1000).astype(int), np.ceil(high / 1000).astype(int)
        if np.prod(high - low) > 1200:
            return None, self.graph, left, right
        ids, graph = self.fine_window(*low, *high)
        return ids, graph, np.searchsorted(ids, left).astype(np.int32), np.searchsorted(ids, right).astype(np.int32)

    def distances(self, sources, sinks, budget):
        left_limit, right_limit = self.search_limits(sources, sinks, budget)
        ids, graph, left, right = self.window(sources, sinks, budget)
        forward = dijkstra(graph, indices=left, min_only=True, limit=left_limit + prism.EPS)
        shortest = float(forward[right].min())
        if shortest > budget + prism.EPS:
            return ids, forward, None, shortest
        backward = (forward if np.array_equal(left, right)
                    else dijkstra(graph, indices=right, min_only=True, limit=right_limit + prism.EPS))
        return ids, forward, backward, shortest


def exact_gap(left, right, limit):
    """A bounded Dijkstra with an additional exact Euclidean rejection."""
    if limit <= prism.MAX_LOCAL_BUDGET_MIN:
        low_l, high_l = GRAPH.xy[left].min(axis=0), GRAPH.xy[left].max(axis=0)
        low_r, high_r = GRAPH.xy[right].min(axis=0), GRAPH.xy[right].max(axis=0)
        separation = np.maximum(np.maximum(low_l - high_r, low_r - high_l), 0.0)
        if np.any(separation > (limit + prism.TIME_TOLERANCE_MIN) * GRAPH.axis_speed):
            return None, None, None, np.inf
    return GRAPH.distances(left, right, limit)


def gap_candidates(left, right, budget, anchors):
    if not len(left) or not len(right):
        return "missing_endpoint", None, []
    if not np.isin(GRAPH.component[left], GRAPH.component[right]).any():
        return "disconnected_endpoints", None, []

    # Adaptively search for the exact ranked shortlist. Outside the cutoff
    # the optimistic POI + anchor reward cannot beat the retained 20th score.
    cutoff = min(3.0, budget)
    stop_budget = budget - MIN_DWELL
    while True:
        ids, forward, backward, shortest = exact_gap(left, right, cutoff)
        if backward is not None:
            if shortest > stop_budget + prism.TIME_TOLERANCE_MIN:
                return "reachable", shortest, []
            if ids is None:
                zones, local_nodes = ALL_ZONES, GRAPH.rep
            else:
                local_zones = GRAPH.representative_zone[ids]
                local_nodes = np.flatnonzero(local_zones >= 0)
                zones = local_zones[local_nodes]
            ti, to = forward[local_nodes], backward[local_nodes]
            feasible = ti + to <= min(cutoff, stop_budget) + prism.TIME_TOLERANCE_MIN
            zones, ti, to = zones[feasible], ti[feasible], to[feasible]
            detour = np.maximum(ti + to - shortest, 0.0)
            tracts = GRAPH.zone_tract[zones]
            score = detour - GRAPH.reward[zones] - prism.ANCHOR_BONUS_MIN * anchors[tracts]
            order = np.lexsort((zones, score))
            chosen = []
            count = Counter()
            for idx in order:
                tract = int(tracts[idx])
                if count[tract] < TRACT_CAP:
                    chosen.append(idx)
                    count[tract] += 1
                    if len(chosen) == TOP_B:
                        break
            # PrismNetwork.max_reward includes the maximum POI reward and ANCHOR_BONUS_MIN.
            unseen_floor = cutoff - shortest - GRAPH.max_reward - prism.TIME_TOLERANCE_MIN
            full = cutoff >= stop_budget
            certified = len(chosen) == TOP_B and score[chosen[-1]] < unseen_floor
            if full or certified:
                # Euclidean metres are a lower bound on distance along each
                # minimum-time leg. Incoming/outgoing minutes are exact.
                points = GRAPH.xy[GRAPH.rep[zones[chosen]]]
                chords_in = np.linalg.norm(points[:, None] - GRAPH.xy[left], axis=2).min(axis=1)
                chords_out = np.linalg.norm(points[:, None] - GRAPH.xy[right], axis=2).min(axis=1)
                result = []
                for j, idx in enumerate(chosen):
                    zone, tract = int(zones[idx]), int(tracts[idx])
                    result.append({
                        "zone_id": zone, "tract_id": str(GRAPH.data["tract_ids"][tract]),
                        "node": int(GRAPH.rep[zone]), "t_in_min": float(ti[idx]),
                        "t_out_min": float(to[idx]), "detour_min": float(detour[idx]),
                        "shortest_leg_chord_m_in": float(chords_in[j]),
                        "shortest_leg_chord_m_out": float(chords_out[j]),
                        "poi_count": int(POI_TOTAL[zone]), "anchor": bool(anchors[tract]),
                        "physical_score_min": float(score[idx]),
                    })
                return "reachable", shortest, result
            needed = shortest + score[chosen[-1]] + GRAPH.max_reward + 0.05 if len(chosen) == TOP_B else 2 * cutoff
            cutoff = min(budget, max(cutoff + 0.75, needed))
        elif cutoff >= budget:
            if np.isfinite(shortest):
                return "travel_budget_infeasible", shortest, []
            return "travel_budget_infeasible", GRAPH.shortest_over_budget(left, right, budget), []
        else:
            cutoff = min(budget, max(cutoff * 2, cutoff + 1))


def extract(item):
    person_day, observed, training_gaps = item
    start, end, tokens, stamps, dow = (observed[k] for k in OBSERVABLES)
    times = np.asarray(stamps, dtype=float)
    if len(tokens) != len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError(f"Invalid AVI times for {person_day}")
    start_idx, end_idx = (GRAPH.tract_index.get(tract, -1) for tract in (start, end))
    anchors = np.zeros(GRAPH.n_tracts, dtype=bool)
    anchors[[i for i in (start_idx, end_idx) if i >= 0]] = True
    sources = [GRAPH.anchor(start_idx), *(GRAPH.observed_edge(token) for token in tokens),
               GRAPH.anchor(end_idx)]
    intervals = []
    # Calendar-day estimates for the censored boundaries. Travel to/from a
    # boundary candidate still has the fixed, independent 15-minute budget.
    day_end = max(1440.0, float(times[-1]) + BOUNDARY_ACCESS) if len(times) else 1440.0
    for index, (left, right) in enumerate(zip(sources, sources[1:])):
        # For training only, other gaps cannot contribute positive activity
        # examples. Retain their positions for the same observable day index.
        if training_gaps is not None and index not in training_gaps:
            intervals.append({"index": index, "candidates": []})
            continue
        first, last = index == 0, index == len(times)
        lower = min(0.0, float(times[0])) if first and len(times) else (0.0 if first else float(times[index - 1]))
        upper = day_end if last else float(times[index])
        span = upper - lower
        budget = min(BOUNDARY_ACCESS, span) if first or last else span
        status, shortest, candidates = gap_candidates(
            left, right, budget + MIN_DWELL if first or last else budget, anchors,
        )
        filtered = []
        for candidate in candidates:
            earliest = lower + candidate["t_in_min"]
            latest = upper - candidate["t_out_min"]
            dwell = latest - earliest
            if dwell < MIN_DWELL - prism.TIME_TOLERANCE_MIN:
                continue
            candidate.update(
                earliest_arrival_min=earliest, latest_departure_min=latest,
                max_dwell_min=max(0.0, dwell), is_day_start_tract=candidate["tract_id"] == start,
                is_day_end_tract=candidate["tract_id"] == end,
                path_length_proxy_m=candidate["shortest_leg_chord_m_in"] + candidate["shortest_leg_chord_m_out"],
            )
            filtered.append(candidate)
        intervals.append({
            "index": index, "lower_min": lower, "upper_min": upper, "span_min": span,
            "min_travel_min": shortest, "slack_min": span - shortest if shortest is not None else None,
            "start_boundary": first, "end_boundary": last, "status": status, "candidates": filtered,
        })
    return {"person_day_id": person_day, "observables": observed, "intervals": intervals}


def static_features(data, summary):
    tracts = pd.read_csv(TRACT_FEATURES, usecols=["tract_id", *SPATIAL],
                         dtype={"tract_id": str}).set_index("tract_id")
    if not tracts.index.is_unique:
        raise ValueError("Duplicate tract ID in the static features table")
    external = tracts.reindex(data["tract_ids"]).to_numpy(dtype=float)[data["zone_tract"]]
    groups = summary["micro_zones"]["poi_groups"]
    counts = data["zone_poi_counts"]
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    features = np.column_stack([shares, np.log1p(counts), external])
    match_groups = {
        "school": ("school",), "shop": ("shop",), "meal": ("meal",),
        "social_recreational": ("social_rec", "worship_civic"), "escort": ("school", "transport"),
        "healthcare": ("healthcare",), "errand_other": ("errand", "transport"),
    }
    poi_match = np.zeros((len(counts), len(CLASSES)))
    for label, names in match_groups.items():
        poi_match[:, CLASS_ID[label]] = np.log1p(counts[:, [groups.index(name) for name in names]].sum(axis=1))
    names = [*(f"zone_share_{g}" for g in groups), *(f"zone_log_count_{g}" for g in groups), *SPATIAL]
    return features, poi_match, names


def activity_row(chain, gap, candidate, spatial):
    start, end = (candidate[key] * (2 * np.pi / 1440)
                  for key in ("earliest_arrival_min", "latest_departure_min"))
    values = [
        np.sin(start), np.cos(start), np.sin(end), np.cos(end),
        chain["observables"]["travel_dow"] in (6, 7),
        gap["index"] / max(len(chain["intervals"]) - 1, 1),
        np.log1p(candidate["max_dwell_min"]), candidate["is_day_start_tract"],
        candidate["is_day_end_tract"], np.log1p(candidate["detour_min"]),
        np.log1p(candidate["t_in_min"]), np.log1p(candidate["t_out_min"]),
        np.log1p(candidate["shortest_leg_chord_m_in"]), np.log1p(candidate["shortest_leg_chord_m_out"]),
        gap["start_boundary"], gap["end_boundary"],
    ]
    return np.r_[values, spatial[candidate["zone_id"]]]


def stop_row(gap):
    candidates = gap["candidates"]
    best = candidates[0]
    return [
        np.log1p(gap["span_min"]), np.log1p(gap["min_travel_min"]), np.log1p(max(0, gap["slack_min"])),
        np.log1p(max(c["max_dwell_min"] for c in candidates)), np.log1p(best["detour_min"]),
        gap["start_boundary"], gap["end_boundary"], np.log1p(best["shortest_leg_chord_m_in"]),
        np.log1p(best["shortest_leg_chord_m_out"]),
    ]


def schedule_bands(train_labels):
    bands = {}
    for label in CLASSES:
        # Exclude day anchors whose survey-day censoring is different from a
        # reconstructed intermediate (possibly boundary) stop.
        rows = [row for day in train_labels.values() for row in day.values()
                if row["label"] == label and row["is_first_act"] == row["is_last_act"] == "0"]
        duration = np.array([float(row["duration_min"]) for row in rows])
        starts = np.array([float(row["start_time_min"]) for row in rows])
        d_min, d_low, d_high, d_max = np.maximum(
            MIN_DWELL, np.quantile(duration, (0.01, 0.2, 0.8, 0.99)),
        )
        s_low, s_high = np.quantile(starts, (0.2, 0.8))
        bands[label] = {"s_low": float(s_low), "s_high": float(s_high), "d_low": float(d_low),
                        "d_high": float(d_high), "d_min": float(d_min), "d_max": float(d_max),
                        "n_train": len(rows)}
    return bands


def train_activity(chains, truth, spatial):
    features, labels = [], []
    for chain in chains:
        for stop in truth[chain["person_day_id"]]:
            if stop["interval"] is None:
                continue
            gap = chain["intervals"][stop["interval"]]
            # Supervision selects the physically top-ranked micro-zone in
            # the true tract. No stop time, duration or truth tract goes in X.
            candidate = next((c for c in gap["candidates"] if c["tract_id"] == stop["tract_id"]), None)
            if candidate is not None:
                features.append(activity_row(chain, gap, candidate, spatial))
                labels.append(CLASS_ID[stop["activity"]])
    counts = np.bincount(labels, minlength=len(CLASSES))
    if not counts.all():
        raise ValueError("Activity training samples must cover all classes")
    model = HistGradientBoostingClassifier(
        max_iter=80, max_leaf_nodes=15, min_samples_leaf=18, l2_regularization=2.0, random_state=0,
    ).fit(features, labels)
    return model, counts / counts.sum(), counts


def fit_stop_model(x, y):
    return HistGradientBoostingClassifier(
        max_iter=90, max_leaf_nodes=12, min_samples_leaf=30, l2_regularization=2.0, random_state=0,
    ).fit(x, y)


def active_stop_gaps(chain):
    for gap in chain["intervals"]:
        gap["p_has_stop"] = 0.0  # hard physical pruning
        if gap["candidates"]:
            yield gap


def train_stop(chains, truth, household_by_day):
    """Write OOF validation scores and separately fit the test-only estimator."""
    x, y, day_index, gaps = [], [], [], []
    for i, chain in enumerate(chains):
        occupied = {t["interval"] for t in truth[chain["person_day_id"]]}
        for gap in active_stop_gaps(chain):
            x.append(stop_row(gap))
            y.append(int(gap["index"] in occupied))
            day_index.append(i)
            gaps.append(gap)
    x, y, day_index = map(np.asarray, (x, y, day_index))
    households = np.array([household_by_day[c["person_day_id"]] for c in chains])
    oof = np.full(len(y), np.nan)
    fold_by_household, folds = {}, []
    day_ids = np.arange(len(chains))
    for fold, (train, held_out) in enumerate(
            GroupKFold(n_splits=STOP_FOLDS).split(day_ids, groups=households), 1):
        train_hh, held_hh = set(households[train]), set(households[held_out])
        if train_hh & held_hh or held_hh & fold_by_household.keys():
            raise ValueError("Household assigned across validation fold boundaries")
        train_rows, held_rows = np.isin(day_index, train), np.isin(day_index, held_out)
        estimator = fit_stop_model(x[train_rows], y[train_rows])
        oof[held_rows] = estimator.predict_proba(x[held_rows])[:, 1]
        fold_by_household.update(dict.fromkeys(sorted(held_hh), fold))
        folds.append({
            "fold": fold, "fit_households": len(train_hh), "held_out_households": len(held_hh),
            "fit_days": len(train), "held_out_days": len(held_out),
            "fit_active_intervals": int(train_rows.sum()), "held_out_active_intervals": int(held_rows.sum()),
        })
    if set(fold_by_household) != set(households) or not np.isfinite(oof).all():
        raise ValueError("Incomplete out-of-fold validation scoring")
    for gap, probability in zip(gaps, oof, strict=True):
        gap["p_has_stop"] = float(probability)
    full_val = fit_stop_model(x, y)
    return full_val, len(y), {
        "splitter": "GroupKFold", "n_splits": STOP_FOLDS, "group": "household_id",
        "score_sources": {"val": "out_of_fold", "test": "full_validation_fit"},
        "fold_by_household": fold_by_household, "folds": folds,
        "full_fit_days": len(chains), "full_fit_households": len(fold_by_household),
        "full_fit_active_intervals": len(y),
    }


def score_test_stops(chains, estimator):
    gaps = [gap for chain in chains for gap in active_stop_gaps(chain)]
    for gap, probability in zip(gaps, estimator.predict_proba([stop_row(gap) for gap in gaps])[:, 1], strict=True):
        gap["p_has_stop"] = float(probability)


def add_activity_scores(chains, model, spatial, poi_match):
    rows, entries = [], []
    for chain in chains:
        for gap in chain["intervals"]:
            for candidate in gap["candidates"]:
                rows.append(activity_row(chain, gap, candidate, spatial))
                entries.append(candidate)
    probability = np.clip(model["activity"].predict_proba(rows), 1e-8, 1.0)
    debiased = np.log(probability) - ETA * np.log(model["prior"])
    for entry, raw, log_score in zip(entries, probability, debiased, strict=True):
        entry["p_obs"] = raw.tolist()
        entry["log_p_obs_debiased"] = log_score.tolist()
        entry["poi_match"] = poi_match[entry["zone_id"]].tolist()


def extract_split(pool, split, inputs, train_truth=None):
    started = perf_counter()
    chains = []
    jobs = [(key, obs, {s["interval"] for s in train_truth[key]} if train_truth is not None else None)
            for key, obs in inputs]
    for chain in pool.map(extract, jobs, chunksize=2):
        chains.append(chain)
        if len(chains) % 200 == 0 or len(chains) == len(inputs):
            print(f"[11] {split} prisms {len(chains)}/{len(inputs)} in {perf_counter() - started:.1f}s", flush=True)
    return chains


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rebuild candidate and model caches")
    args = parser.parse_args()
    started = perf_counter()
    source = signature()
    manifest = CACHE / "build_manifest.json"
    if (not args.force and manifest.exists() and MODEL_FILE.exists()
            and all(candidate_file(s).exists() for s in ("val", "test"))
            and json.loads(manifest.read_text())["signature"] == source):
        print(f"[11] Reused model and candidate caches in {perf_counter() - started:.3f}s")
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    # Invalidate the completion marker before replacing any generation member.
    manifest.unlink(missing_ok=True)
    train_labels, train_households = label_rows("train")
    val_labels, val_households = label_rows("val")
    split_map = json.loads(SPLIT.read_text())
    test = json.loads(network.COHORT.read_text())
    if test["masked_test_signature"] != network.file_signature(network.MASKED_TEST):
        raise ValueError("Step 09 cohort is stale")
    cohorts = {"train": train_labels.keys(), "val": val_labels.keys(), "test": test["person_day_ids"]}
    hh_split = split_map["household_to_split"]
    for split, households in (("train", train_households), ("val", val_households)):
        if {hh_split[h] for h in households} != {split}:
            raise ValueError(f"Household split mismatch for {split}")
    for split, ids in cohorts.items():
        if len(set(ids)) != EXPECTED[split] or not set(ids) <= set(split_map["split_to_person_day_ids"][split]):
            raise ValueError(f"Invalid {split} person-day cohort")
    days = pd.read_csv(CALENDAR, usecols=["day_id", "travel_dow"], dtype={"day_id": str})
    days = days[days["travel_dow"].isin(range(1, 8))]
    if not days["day_id"].is_unique:
        raise ValueError("Duplicate day ID in the calendar")
    calendar = dict(zip(days["day_id"], days["travel_dow"].astype(int), strict=True))
    for labels in (train_labels, val_labels):
        if any(len(rows) == 0 or key not in calendar for key, rows in labels.items()):
            raise ValueError("Calendar join failed")
    train_truth = supervised_truth("train", train_labels)
    val_truth = supervised_truth("val", val_labels)
    val_household_by_day = {}
    for key, rows in val_labels.items():
        households = {row["household_id"] for row in rows.values()}
        if len(households) != 1:
            raise ValueError(f"Inconsistent validation household for {key}")
        val_household_by_day[key] = households.pop()

    global GRAPH, ALL_ZONES, POI_TOTAL
    data, summary = network.load_cache()
    if summary["cache_version"] != network.CACHE_VERSION:
        raise ValueError("Unsupported step 09 cache")
    GRAPH = CandidateNetwork(data, summary)
    ALL_ZONES = np.arange(len(GRAPH.rep))
    POI_TOTAL = data["zone_poi_counts"].sum(axis=1)
    spatial, poi_match, spatial_names = static_features(data, summary)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("fork")) as pool:
        train = extract_split(pool, "train", read_inputs("train", cohorts["train"], calendar), train_truth)
        activity, prior, counts = train_activity(train, train_truth, spatial)
        del train
        val = extract_split(pool, "val", read_inputs("val", cohorts["val"], calendar))
        cross_fit_started = perf_counter()
        stop, stop_n, cross_fit = train_stop(val, val_truth, val_household_by_day)
        cross_fit_seconds = perf_counter() - cross_fit_started
        models = {
            "activity": activity, "stop": stop, "prior": prior.tolist(), "stop_cross_fit": cross_fit,
            "bands": schedule_bands(train_labels), "activity_class_counts": dict(zip(CLASSES, counts.tolist())),
            "activity_features": [*ACTIVITY_FEATURES, *spatial_names], "stop_features": STOP_FEATURES,
            "activity_fit_chains": len(train_labels), "stop_fit_chains": len(val_labels),
            "stop_fit_active_intervals": stop_n, "classes": CLASSES, "eta": ETA,
        }
        add_activity_scores(val, models, spatial, poi_match)
        stage = perf_counter()
        test_chains = extract_split(pool, "test", read_inputs("test", cohorts["test"], calendar))
        score_test_stops(test_chains, stop)
        add_activity_scores(test_chains, models, spatial, poi_match)
        test_seconds = perf_counter() - stage

    for split, chains in (("val", val), ("test", test_chains)):
        def write(stream):
            for chain in chains:
                stream.write(json.dumps(chain, allow_nan=False, separators=(",", ":")) + "\n")
        prism.write_atomic(candidate_file(split), write)
    # Only this tool's trusted local cache is loaded by stage 12.
    temporary = MODEL_FILE.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(models, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(MODEL_FILE)
    finally:
        temporary.unlink(missing_ok=True)
    elapsed = perf_counter() - started
    prism.write_atomic(manifest, lambda f: json.dump(
        {"signature": source, "runtime_seconds": elapsed, "test_extraction_and_scoring_seconds": test_seconds,
         "workers": WORKERS, "cohorts": EXPECTED, "stop_cross_fit": cross_fit,
         "stop_cross_fit_and_full_fit_seconds": cross_fit_seconds}, f, indent=2,
    ))
    print(f"[11] {elapsed:.3f}s total, {sum(counts)} activity examples, {stop_n} stop examples, "
          f"{STOP_FOLDS} household-grouped folds plus full-val fit in {cross_fit_seconds:.3f}s", flush=True)


if __name__ == "__main__":
    main()
