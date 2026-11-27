"""Extract observable-only Chicago time-space prisms, then audit held-out truth."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# The task authorizes only two scripts and their data/result artifacts.
sys.dont_write_bytecode = True
network = importlib.import_module("09_build_corridor_network")

CHAIN_OUTPUT = network.MVP / "data" / "reduced_prism_test_324.jsonl"
RESULTS = network.MVP / "results"
RESULT_JSON = RESULTS / "prism_validation_324chains.json"
RESULT_MD = RESULTS / "prism_validation_summary.md"
SUBSET = network.MVP / "data" / "batches_100" / "ground_truth.json"

OBSERVABLES = (
    "known_day_start_tract_id", "known_day_end_tract_id", "observed_edges", "observed_time_min",
)
CLASSES = (
    "home", "work", "school", "shop", "meal", "social_recreational", "escort", "healthcare", "errand_other",
)
TOP_B = (10, 20, 30, 50)
FULL_RANK = max(TOP_B) + 1
MISSING_RANK = FULL_RANK + 1
MIN_STOP_MIN = 5.0
BOUNDARY_BUDGET_MIN = 15.0
POI_WEIGHT_MIN = 0.75
ANCHOR_BONUS_MIN = 1.5
RUNTIME_WORKERS = 8
EPS = 1e-8
# AVI timestamps are rounded to four decimal places in the strict dataset.
TIME_TOLERANCE_MIN = 0.000051
GRID_WINDOW_M = 5000.0
MAX_LOCAL_BUDGET_MIN = 8.0
DURATION_LABELS = ("<1", "1–<5", "5–<15", "15–<60", "60+")
EMPTY_SEGMENTS = np.empty(0, dtype=np.int32)


def rate(numerator, denominator):
    return {"n": int(numerator), "denominator": int(denominator),
            "rate": float(numerator / denominator) if denominator else None}


def rows_to_indices(starts, lengths):
    offsets = np.cumsum(lengths, dtype=np.int64) - lengths
    return np.arange(lengths.sum(), dtype=np.int64) + np.repeat(starts - offsets, lengths)


def round_min(value):
    return round(float(value), 5) if np.isfinite(value) else None


def distribution(values, percentiles=(50, 90, 100)):
    names = {50: "median", 75: "p75", 90: "p90", 100: "max"}
    quantiles = np.percentile(values, percentiles).tolist() if len(values) else [None] * len(percentiles)
    return {names[p]: value for p, value in zip(percentiles, quantiles)}


class PrismNetwork:
    def __init__(self, data, summary):
        self.data = data
        self.summary = summary
        self.n_nodes = summary["nodes"]
        self.graph = csr_matrix(
            (data["graph_data"], data["graph_indices"], data["graph_indptr"]),
            shape=(self.n_nodes, self.n_nodes),
        )
        self.tract_index = {tract: index for index, tract in enumerate(data["tract_ids"])}
        self.n_tracts = len(self.tract_index)
        self.u, self.v, self.w = data["segment_u"], data["segment_v"], data["segment_min"]
        self.component = data["component"]
        self.main_component = summary["largest_component_id"]
        self.group_starts = data["segment_row_start"]
        self.unique_pair = ((self.u != self.v) &
                            np.r_[True, (self.u[1:] != self.u[:-1]) | (self.v[1:] != self.v[:-1])])
        offsets = data["vocab_segment_offsets"]
        physical = data["vocab_is_physical"]
        physical_indices = rows_to_indices(offsets[:-1][physical], np.diff(offsets)[physical])
        self.physical_segments = np.sort(data["vocab_segment_ids"][physical_indices])
        self.physical_slot = np.full(len(self.u), -1, dtype=np.int32)
        self.physical_slot[self.physical_segments] = np.arange(len(self.physical_segments))
        self.physical_bitmap_size = (len(self.physical_segments) + 7) // 8
        self.zone_tract = data["zone_tract"]
        self.rep = data["zone_representative"]
        self.representative_zone = np.full(self.n_nodes, -1, dtype=np.int32)
        self.representative_zone[self.rep] = np.arange(len(self.rep))
        self.reward = POI_WEIGHT_MIN * np.log1p(data["zone_poi_counts"].sum(axis=1))
        self.max_reward = float(self.reward.max() + ANCHOR_BONUS_MIN)
        self.xy = (data["node_lonlat"] - network.ORIGIN) * network.SCALE_M
        self.grid = network.SpatialGrid(self.xy)
        # Axis-wise upper speeds from actual graph segments make the local
        # search window exact, including for links with optimistic speeds.
        self.axis_speed = np.array([
            np.max(np.abs(self.xy[self.u, axis] - self.xy[self.v, axis]) / self.w)
            for axis in (0, 1)
        ])
        self.main_nodes = self.component == self.main_component
        self.main_segments = self.main_nodes[self.u]
        self.main_node_count = int(self.main_nodes.sum())
        self.main_edge_count = int(np.count_nonzero(self.main_segments & self.unique_pair))
        self.main_segment_count = int(self.main_segments.sum())
        self.main_zones = np.flatnonzero(self.main_nodes[self.rep])
        self.main_tracts = np.unique(self.zone_tract[self.main_zones])
        # Triangle inequality certifies the whole main component for broad
        # intervals. Only Top-50 then needs an exact, much shorter search.
        available_nodes = np.flatnonzero(self.main_nodes)
        centers = np.quantile(self.xy[available_nodes], [[0.35], [0.5], [0.65]], axis=0).reshape(-1, 2)
        landmarks = [
            available_nodes[np.argmin(np.sum((self.xy[available_nodes] - center) ** 2, axis=1))]
            for center in centers
        ]
        self.landmark_dist = np.array([
            dijkstra(self.graph, indices=landmark, min_only=True) for landmark in landmarks
        ])
        self.landmark_bound = (
            2 * np.max(self.landmark_dist[:, self.main_nodes], axis=1) + self.w[self.main_segments].max()
        )

    def anchor(self, tract):
        offsets = self.data["anchor_offsets"]
        if tract < 0:
            return np.empty(0, dtype=np.int32)
        return self.data["anchor_nodes"][offsets[tract]:offsets[tract + 1]]

    @lru_cache(maxsize=32_768)
    def observed_edge(self, token):
        """Unknown detector position permits any retained node on the hit edge."""
        offsets = self.data["vocab_segment_offsets"]
        if not 0 <= token < len(offsets) - 1:
            return np.empty(0, dtype=np.int32)
        ids = self.data["vocab_segment_ids"][offsets[token]:offsets[token + 1]]
        return np.union1d(self.u[ids], self.v[ids])

    @lru_cache(maxsize=18)
    def local_graph(self, x0, y0, x1, y1):
        low, high = GRID_WINDOW_M * np.array([[x0, y0], [x1, y1]])
        ids = self.grid.query(low, high)
        ids = np.sort(ids[np.all((self.xy[ids] >= low) & (self.xy[ids] <= high), axis=1)])
        return ids, self.graph[ids, :][:, ids].tocsr()

    def window(self, left, right, budget):
        if budget > MAX_LOCAL_BUDGET_MIN:
            return None, self.graph, left, right
        margin = budget * self.axis_speed + 1e-5
        low = np.maximum(self.xy[left].min(axis=0), self.xy[right].min(axis=0)) - margin
        high = np.minimum(self.xy[left].max(axis=0), self.xy[right].max(axis=0)) + margin
        if np.any(high < low):
            empty = np.empty(0, dtype=np.int32)
            return empty, None, empty, empty
        x0, y0 = np.floor(low / GRID_WINDOW_M).astype(int)
        x1, y1 = np.ceil(high / GRID_WINDOW_M).astype(int)
        if (x1 - x0) * (y1 - y0) > 72:
            return None, self.graph, left, right
        ids, graph = self.local_graph(x0, y0, x1, y1)

        def local(sources):
            positions = np.searchsorted(ids, sources)
            valid = positions < len(ids)
            positions = positions[valid]
            return positions[ids[positions] == sources[valid]].astype(np.int32)

        return ids, graph, local(left), local(right)

    def search_limits(self, sources, sinks, budget):
        """If d(A,B) <= h for all B, feasible d(A,x) <= (budget+h)/2."""
        if budget <= 12.0:
            return budget, budget
        da, db = self.landmark_dist[:, sources], self.landmark_dist[:, sinks]
        ab = float((da.min(axis=1) + db.max(axis=1)).min())
        ba = float((db.min(axis=1) + da.max(axis=1)).min())
        # Most long gaps are bounded by nearby observations. Tighten the
        # landmark bounds via an inexpensive local search for close seed sets.
        probe = 4.0
        ids, graph, left, right = self.window(sources, sinks, probe)
        if len(left) == len(sources) and len(right) == len(sinks):
            if ab > probe:
                forward = dijkstra(graph, indices=left, min_only=True, limit=probe + EPS)
                ab = min(ab, float(forward[right].max()))
            if ba > probe:
                backward = dijkstra(graph, indices=right, min_only=True, limit=probe + EPS)
                ba = min(ba, float(backward[left].max()))
        return min(budget, (budget + ab) / 2), min(budget, (budget + ba) / 2)

    def distances(self, sources, sinks, budget):
        left_limit, right_limit = self.search_limits(sources, sinks, budget)
        ids, graph, left, right = self.window(sources, sinks, max(left_limit, right_limit))
        if not len(left) or not len(right):
            return ids, None, None, np.inf
        forward = dijkstra(graph, indices=left, min_only=True, limit=left_limit + EPS)
        shortest = float(forward[right].min())
        if shortest > budget + EPS:
            return ids, forward, None, shortest
        # The CSR is symmetric, so routing from the right yields backward time.
        backward = (forward if np.array_equal(left, right)
                    else dijkstra(graph, indices=right, min_only=True, limit=right_limit + EPS))
        return ids, forward, backward, shortest

    def shortest_over_budget(self, left, right, budget):
        """Measure the real deficit for endpoints already known to be connected."""
        limit = max(2.0, budget + 2.0)
        while True:
            _, _, _, shortest = self.distances(left, right, limit)
            if np.isfinite(shortest):
                return shortest
            limit *= 2

    def encode_physical_segments(self, segs):
        if not len(segs):
            return EMPTY_SEGMENTS
        slots = self.physical_slot[segs]
        slots = slots[slots >= 0]
        if slots.nbytes <= self.physical_bitmap_size:
            return slots
        mask = np.zeros(len(self.physical_segments), dtype=bool)
        mask[slots] = True
        return np.packbits(mask)

    def candidate_ranks(self, ids, forward, backward, stop_budget, shortest, anchors):
        if ids is None:
            zones = np.arange(len(self.rep))
            zone_nodes = self.rep
        else:
            local_zones = self.representative_zone[ids]
            zone_nodes = np.flatnonzero(local_zones >= 0)
            zones = local_zones[zone_nodes]
        travel = forward[zone_nodes] + backward[zone_nodes]
        feasible = travel <= stop_budget + EPS
        zones, travel = zones[feasible], travel[feasible]
        score = (np.maximum(travel - shortest, 0) - self.reward[zones]
                 - ANCHOR_BONUS_MIN * anchors[self.zone_tract[zones]])
        # The last score is needed to certify the adaptive search on broad gaps.
        order = np.lexsort((zones, score))[:TOP_B[-1]]
        return zones, zones[order], score[order]

    def feasible_segments(self, feasible, ids, forward, backward, budget):
        if len(feasible) * 2 >= self.n_nodes and ids is None:
            candidates = slice(None)
            left, right = self.u, self.v
        else:
            start = self.group_starts[feasible]
            length = self.group_starts[feasible + 1] - start
            candidates = rows_to_indices(start, length)
            left, right = self.u[candidates], self.v[candidates]
            if ids is not None:
                right_local = np.searchsorted(ids, right)
                take = np.flatnonzero(right_local < len(ids))
                take = take[ids[right_local[take]] == right[take]]
                candidates = candidates[take]
                left, right = np.searchsorted(ids, left[take]), right_local[take]
        crossing = np.minimum(forward[left] + backward[right], forward[right] + backward[left])
        found = crossing + self.w[candidates] <= budget + EPS
        return np.flatnonzero(found) if isinstance(candidates, slice) else candidates[found]

    def broad_component(self, left, right, budget):
        if budget < float(self.landmark_bound.min()):
            return False
        if not (np.all(self.main_nodes[left]) and np.all(self.main_nodes[right])):
            return False
        bound = (self.landmark_bound + self.landmark_dist[:, left].min(axis=1)
                 + self.landmark_dist[:, right].min(axis=1))
        return bool(np.any(bound <= budget - MIN_STOP_MIN))

    def broad_ranks(self, left, right, budget, anchors):
        """Return the exact Top-50 of a certified full-component stop prism."""
        cutoff = min(8.0, budget)
        while True:
            ids, forward, backward, shortest = self.distances(left, right, cutoff)
            if backward is not None:
                _, top, scores = self.candidate_ranks(
                    ids, forward, backward, min(cutoff, budget - MIN_STOP_MIN), shortest, anchors,
                )
                unseen_floor = cutoff - shortest - self.max_reward
                if cutoff >= budget - MIN_STOP_MIN or (len(top) == TOP_B[-1] and scores[-1] < unseen_floor):
                    return top, shortest
                # Expand just enough to include any candidate that could beat
                # the currently ranked 50th candidate, irrespective of POIs.
                if len(top) == TOP_B[-1]:
                    cutoff = min(budget, max(cutoff + 1.0, shortest + scores[-1] + self.max_reward + 0.5))
                    continue
            cutoff = min(budget, cutoff * 2)

    def interval(self, left, right, span, anchors, node_union, segment_union, main_already_covered):
        is_interior = span is not None
        budget = span if is_interior else BOUNDARY_BUDGET_MIN
        rank = np.full(self.n_tracts, MISSING_RANK, dtype=np.uint8)
        top, full_tracts = [], np.empty(0, dtype=np.int32)
        shortest, n_nodes, n_edges, n_segments, n_zones = np.inf, 0, 0, 0, 0
        status, covers_main = "reachable", False
        segs = EMPTY_SEGMENTS

        if not (left.size and right.size):
            status = "missing_endpoint"
        elif not np.isin(self.component[left], self.component[right]).any():
            status = "disconnected_endpoints"
        elif self.broad_component(left, right, budget):
            top, shortest = self.broad_ranks(left, right, budget, anchors)
            full_tracts, n_zones = self.main_tracts, len(self.main_zones)
            n_nodes, n_edges, n_segments = self.main_node_count, self.main_edge_count, self.main_segment_count
            covers_main = True
            if not main_already_covered:
                node_union |= self.main_nodes
                segment_union |= self.main_segments
        else:
            ids, forward, backward, shortest = self.distances(left, right, budget)
            if backward is None:
                status = "travel_budget_infeasible"
                if not np.isfinite(shortest):
                    shortest = self.shortest_over_budget(left, right, budget)
            else:
                reached = np.flatnonzero(forward + backward <= budget + EPS)
                global_nodes = reached if ids is None else ids[reached]
                segs = self.feasible_segments(global_nodes, ids, forward, backward, budget)
                n_nodes, n_edges, n_segments = len(global_nodes), int(self.unique_pair[segs].sum()), len(segs)
                node_union[global_nodes] = True
                segment_union[segs] = True
                if shortest <= budget - MIN_STOP_MIN + EPS:
                    zones, top, _ = self.candidate_ranks(ids, forward, backward, budget - MIN_STOP_MIN, shortest, anchors)
                    n_zones = len(zones)
                    full_tracts = np.unique(self.zone_tract[zones])

        # Boundary clock times are unknown. Their assumed reachability budget
        # supplies candidates but cannot support temporal transit pruning.
        transit = is_interior and status == "reachable" and span - shortest < MIN_STOP_MIN - EPS
        top = np.asarray(top, dtype=np.int32)
        rank[full_tracts] = FULL_RANK
        np.minimum.at(rank, self.zone_tract[top], np.arange(1, 1 + len(top), dtype=np.uint8))
        record = {
            "span_min": round_min(span) if is_interior else None, "budget_min": round_min(budget),
            "min_travel_min": round_min(shortest),
            "slack_min": round_min(span - shortest) if is_interior and np.isfinite(shortest) else None,
            "status": status, "pure_transit": transit, "broad_component": covers_main,
            "prism_nodes": n_nodes, "prism_edges": n_edges, "prism_segment_instances": n_segments,
            "full_candidate_micro_zones": n_zones, "full_candidate_tracts": len(full_tracts),
            "top_micro_zone_ids": top.tolist(),
        }
        if status in ("missing_endpoint", "disconnected_endpoints"):
            # Preserve the available sources as observation metadata. There is
            # no two-sided prism or inferred graph edge in this fallback.
            record["fallback_seed_nodes"] = np.union1d(left, right).tolist()
        membership = None if covers_main else self.encode_physical_segments(segs)
        return record, rank, membership

    def predict(self, start_tract, end_tract, observed_edges, observed_times):
        timestamps = np.asarray(observed_times, dtype=float)
        if len(timestamps) != len(observed_edges) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError("Expected aligned, finite, nondecreasing AVI times")
        start, end = self.tract_index.get(start_tract, -1), self.tract_index.get(end_tract, -1)
        anchors = np.zeros(self.n_tracts, dtype=bool)
        anchors[[value for value in (start, end) if value >= 0]] = True
        sources = [self.anchor(start), *(self.observed_edge(edge) for edge in observed_edges), self.anchor(end)]
        node_union = np.zeros(self.n_nodes, dtype=bool)
        segment_union = np.zeros(len(self.u), dtype=bool)
        main_covered = False
        records, ranks, memberships = [], [], []
        for index, (left, right) in enumerate(zip(sources, sources[1:])):
            time_left = timestamps[index - 1] if index > 0 else None
            time_right = timestamps[index] if index < len(timestamps) else None
            span = float(time_right - time_left) if time_left is not None and time_right is not None else None
            interval, rank, membership = self.interval(left, right, span, anchors, node_union, segment_union, main_covered)
            main_covered |= interval["broad_component"]
            interval.update(index=index, start_time_min=round_min(time_left) if time_left is not None else None,
                            end_time_min=round_min(time_right) if time_right is not None else None)
            records.append(interval)
            ranks.append(rank)
            memberships.append(membership)

        # Compact union bitmap allows a separate held-out pass without keeping
        # an uncompressed graph-length mask for every chain in memory.
        offsets = self.data["vocab_segment_offsets"]
        mapped = np.flatnonzero(np.diff(offsets))
        covered_tokens = np.logical_and.reduceat(
            segment_union[self.data["vocab_segment_ids"]], offsets[mapped],
        )
        ranks = np.asarray(ranks)
        chain_rank = ranks.min(axis=0)
        top_b_chain = {}
        for b in TOP_B:
            label = str(b)
            top_b_chain[label] = int(np.count_nonzero(chain_rank <= b))
            for interval, count in zip(records, np.count_nonzero(ranks <= b, axis=1)):
                interval.setdefault("top_b_tract_counts", {})[label] = int(count)
        return {
            "intervals": records, "n_observed": len(observed_edges),
            "union_nodes": int(node_union.sum()), "union_edges": int(np.count_nonzero(segment_union & self.unique_pair)),
            "union_segment_instances": int(segment_union.sum()),
            "union_vocab_edge_tokens": int(covered_tokens.sum()),
            "union_tracts": int(np.count_nonzero(chain_rank < MISSING_RANK)),
            "union_top_b_tract_counts": top_b_chain,
        }, ranks, np.packbits(segment_union), memberships

    def token_coverage(self, targets, segment_union):
        offsets = self.data["vocab_segment_offsets"]
        indices = self.data["vocab_segment_ids"]
        coverage = {}
        for token in set(targets):
            lo, hi = offsets[token:token + 2]
            coverage[token] = bool(lo < hi and segment_union[indices[lo:hi]].all())
        return coverage

    def interval_token_coverage(self, targets, membership):
        if membership is None:
            return self.token_coverage(targets, self.main_segments)
        if not len(membership):
            return dict.fromkeys(targets, False)
        offsets = self.data["vocab_segment_offsets"]
        indices = self.data["vocab_segment_ids"]
        coverage = {}
        for token in set(targets):
            lo, hi = offsets[token:token + 2]
            slots = self.physical_slot[indices[lo:hi]]
            if membership.dtype == np.uint8:
                present = (membership[slots // 8] >> (7 - slots % 8)) & 1
            else:
                pos = np.searchsorted(membership, slots)
                present = membership[np.minimum(pos, len(membership) - 1)] == slots
            coverage[token] = bool(lo < hi and present.all())
        return coverage


def align_observations(record):
    """Find observed (token, unwrapped time) in target sequence order."""
    target = list(zip(record["target_edge_sequence"], record["target_edge_time_min"]))
    positions = []
    cursor = 0
    for token, time in zip(record["observed_edges"], record["observed_time_min"]):
        while cursor < len(target):
            edge, stamp = target[cursor]
            cursor += 1
            if edge == token and abs(stamp - time) <= TIME_TOLERANCE_MIN:
                positions.append(cursor - 1)
                break
        else:
            return None
    return positions


def match_stop(start, end, observed, observed_trips, incoming, outgoing):
    """Match the whole stop to one AVI gap, using trip order for timestamp ties."""
    if start is None or end is None or end < start:
        return None, "invalid_stop_time"
    first = bisect_left(observed, end - TIME_TOLERANCE_MIN)
    last = bisect_right(observed, start + TIME_TOLERANCE_MIN)
    if first > last:
        return None, "crosses_avi_observation"
    if first == last:
        return first, "time_contained"
    # A zero-duration stop at an observed timestamp can fit more than one
    # closed gap. Incoming observations precede it and outgoing ones follow.
    if observed_trips is not None:
        if incoming is not None:
            first = max(first, max((i + 1 for i, trip in enumerate(observed_trips) if trip <= incoming), default=0))
        if outgoing is not None:
            last = min(last, min((i for i, trip in enumerate(observed_trips) if trip >= outgoing), default=len(observed)))
    if first == last:
        return first, "trip_order_tie_break"
    return None, "ambiguous_timestamp_tie"


def load_evaluation_labels():
    settings = network.literal_settings(network.SCRIPTS / "02_build_mvp_dataset.py", {"LABEL_MAP", "HEALTHCARE_DPURPOSE"})
    with (network.RAW / "MyDailyTravelData" / "trip_linked.csv").open(encoding="utf-8") as stream:
        healthcare = {
            row["linked_trip_id"] for row in csv.DictReader(stream)
            if row["d_purpose"] in settings["HEALTHCARE_DPURPOSE"]
        }
    return settings["LABEL_MAP"], healthcare


def duration_bin(duration):
    if duration is None:
        return "unknown"
    return DURATION_LABELS[bisect_right((1, 5, 15, 60), duration + TIME_TOLERANCE_MIN)]


def evaluate(record, prediction, ranks, packed_segments, memberships, graph, labels, healthcare):
    """All truth-dependent stop matching, class lookup and edge scoring lives here."""
    observed = record["observed_time_min"]
    aligned = align_observations(record)
    trips_for_edges = record["target_trip_index_sequence"]
    observed_trips = None if aligned is None else [trips_for_edges[i] for i in aligned]
    truth = record["ground_truth_chain"]
    incoming = {t["destination_activity_index"]: t for t in truth["trips"]}
    outgoing = {t["origin_activity_index"]: t for t in truth["trips"]}
    chain_rank = ranks.min(axis=0)
    intervals = prediction["intervals"]
    non_broad = np.asarray([not interval["broad_component"] for interval in intervals])
    chain_rank_non_broad = ranks[non_broad].min(axis=0, initial=MISSING_RANK)
    stops, excluded = [], Counter()
    for activity in truth["activities"][1:-1]:
        index = activity["activity_index"]
        before, after = incoming.get(index), outgoing.get(index)
        raw_label = activity["activity_type"]["label"]
        label = "healthcare" if before and before["linked_trip_id"] in healthcare else labels.get(raw_label)
        if label is None:
            excluded[raw_label] += 1
            continue
        start, end = activity["start_time_min"], activity["end_time_min"]
        gap, match = match_stop(start, end, observed, observed_trips,
                               before["trip_index"] if before else None, after["trip_index"] if after else None)
        matched = intervals[gap] if gap is not None else None
        tract_id = activity["tract_id"]
        tract = graph.tract_index.get(tract_id, -1)
        stops.append({
            "activity_index": index, "class": label, "tract_id": tract_id,
            "start_time_min": start, "end_time_min": end,
            "duration_min": max(0.0, end - start) if start is not None and end is not None else None,
            "avi_interval": gap, "interval_match_status": match, "unknown_tract": tract < 0,
            "pruned_by_transit": bool(matched and matched["pure_transit"]),
            "matched_interval_status": matched["status"] if matched else None,
            "interval_rank": int(ranks[gap, tract]) if gap is not None and tract >= 0 else MISSING_RANK,
            "chain_rank": int(chain_rank[tract]) if tract >= 0 else MISSING_RANK,
            "chain_rank_without_broad": int(chain_rank_non_broad[tract]) if tract >= 0 else MISSING_RANK,
        })

    seq = record["target_edge_sequence"]
    physical = graph.data["vocab_is_physical"]
    physical_seq = [token for token in seq if physical[token]]
    covered = graph.token_coverage(physical_seq, np.unpackbits(packed_segments, count=len(graph.u)))
    covered_non_broad = covered
    if not non_broad.all():
        keep = np.zeros(len(graph.physical_segments), dtype=bool)
        for membership in memberships:
            if membership is None:
                continue
            if membership.dtype == np.uint8:
                keep |= np.unpackbits(membership, count=len(keep)).astype(bool)
            else:
                keep[membership] = True
        covered_non_broad = graph.interval_token_coverage(physical_seq, np.packbits(keep))

    if aligned is not None:
        for i, (lo, hi) in enumerate(zip(aligned, aligned[1:]), start=1):
            inner = [token for token in seq[lo + 1:hi] if physical[token]]
            boundary = [seq[j] for j in (lo, hi) if physical[seq[j]]]
            token_hits = graph.interval_token_coverage(inner + boundary, memberships[i])
            inner_hits = sum(token_hits[token] for token in inner)
            intervals[i]["aligned_edge_coverage"] = {
                "between_avi_observations": rate(inner_hits, len(inner)),
                "including_avi_boundary_edges": rate(inner_hits + sum(token_hits[token] for token in boundary),
                                                       len(inner) + len(boundary)),
            }

    aligned_set = None if aligned is None else set(aligned)
    misses = Counter()
    n, hits, withheld, withheld_hits = 0, 0, 0, 0
    for index, token in enumerate(seq):
        if not physical[token]:
            continue
        n += 1
        hits += covered[token]
        if not covered[token]:
            misses[token] += 1
        if aligned_set is not None and index not in aligned_set:
            withheld += 1
            withheld_hits += covered[token]
    audit = {
        "stops": stops, "excluded_non_activity_labels": dict(excluded),
        "observations_aligned_to_target": aligned is not None,
        "physical_edges": rate(hits, n), "withheld_physical_edges": rate(withheld_hits, withheld),
        "physical_edges_without_broad": rate(sum(covered_non_broad[token] for token in physical_seq), n),
        "missed_physical_token_occurrences": dict(sorted(misses.items())),
    }
    return audit


def summarize(chains, graph):
    stops = [s for chain in chains for s in chain["evaluation"]["stops"]]
    intervals = [i for chain in chains for i in chain["intervals"]]

    def recall(group):
        thresholds = list(zip(map(str, TOP_B), TOP_B)) + [("full_prism", FULL_RANK)]
        return {
            frame: {label: rate(sum(stop[field] <= threshold for stop in group), len(group))
                    for label, threshold in thresholds}
            for frame, field in (("interval_matched", "interval_rank"), ("chain_union", "chain_rank"))
        }

    def false_negatives(group):
        return rate(sum(stop["pruned_by_transit"] for stop in group), len(group))

    def aggregate_edges(field):
        ratios = [chain["evaluation"][field] for chain in chains]
        return rate(sum(x["n"] for x in ratios), sum(x["denominator"] for x in ratios))

    interior = [i for i in intervals if i["span_min"] is not None]
    active = [i for i in intervals if i["full_candidate_micro_zones"]]
    local_active = [i for i in active if not i["broad_component"]]
    infeasible = [i for i in interior if i["status"] == "travel_budget_infeasible"]
    infeasible_by_span = {
        "zero_span": [i for i in infeasible if i["span_min"] == 0],
        "positive_span": [i for i in infeasible if i["span_min"] > 0],
    }
    infeasible_metrics = {
        "fraction_interior_intervals": rate(len(infeasible), len(interior)),
        "boundary_intervals": sum(i["span_min"] is None and i["status"] == "travel_budget_infeasible" for i in intervals),
        "stops_in_infeasible_intervals": rate(
            sum(s["matched_interval_status"] == "travel_budget_infeasible" for s in stops), len(stops)),
    }
    for label, group in {"all_infeasible": infeasible, **infeasible_by_span}.items():
        infeasible_metrics[label] = {
            "count": len(group),
            "span_min": distribution([i["span_min"] for i in group]),
            "shortest_minus_span_min": distribution([-i["slack_min"] for i in group]),
        }

    interval_coverage = {}
    for label, group in (("overall", interior),
                         ("excluding_broad_component", [i for i in interior if not i["broad_component"]])):
        measured = [i["aligned_edge_coverage"] for i in group if "aligned_edge_coverage" in i]
        interval_coverage[label] = {"interior_intervals": len(group), "aligned_intervals": len(measured)}
        for scope in ("between_avi_observations", "including_avi_boundary_edges"):
            ratios = [i[scope] for i in measured if i[scope]["denominator"]]
            interval_coverage[label][scope] = {
                "physical_edge_occurrences": rate(sum(r["n"] for r in ratios), sum(r["denominator"] for r in ratios)),
                "intervals_with_physical_edges": len(ratios),
                "mean_interval_recall": float(np.mean([r["rate"] for r in ratios])) if ratios else None,
            }

    top_b_tracts = {}
    for b in map(str, TOP_B):
        top_b_tracts[b] = {
            "per_active_interval": float(np.mean([i["top_b_tract_counts"][b] for i in active])) if active else None,
            "per_chain_union": float(np.mean([c["union_top_b_tract_counts"][b] for c in chains])),
        }

    fields = ("union_nodes", "union_edges", "union_segment_instances", "union_vocab_edge_tokens", "union_tracts")
    means = {field: float(np.mean([chain[field] for chain in chains])) for field in fields}
    full_graph = graph.summary
    class_groups = {label: [s for s in stops if s["class"] == label] for label in CLASSES}
    duration_groups = {label: [] for label in (*DURATION_LABELS, "unknown")}
    for stop in stops:
        duration_groups[duration_bin(stop["duration_min"])].append(stop)
    n_transit = sum(i["pure_transit"] for i in intervals)
    excluded = Counter()
    for chain in chains:
        excluded.update(chain["evaluation"]["excluded_non_activity_labels"])
    numbers = [c["prediction_runtime_sec"] for c in chains]
    return {
        "n_chains": len(chains), "n_stops": len(stops),
        "n_non_home_stops": sum(s["class"] != "home" for s in stops),
        "recall": {"overall": recall(stops),
                   "non_home": recall([s for s in stops if s["class"] != "home"]),
                   "per_class": {label: recall(group) for label, group in class_groups.items()}},
        "stop_interval_matching": {
            "matched": rate(sum(s["avi_interval"] is not None for s in stops), len(stops)),
            "status_counts": dict(Counter(s["interval_match_status"] for s in stops)),
            "unknown_tract_stops": sum(s["unknown_tract"] for s in stops),
            "excluded_non_activity_labels": dict(excluded),
        },
        "transit_pruning": {
            "fraction_all_intervals": rate(n_transit, len(intervals)),
            "fraction_interior_intervals": rate(n_transit, len(interior)),
            "false_negative_rate": false_negatives(stops),
            "false_negative_rate_among_matched_stops": false_negatives([s for s in stops if s["avi_interval"] is not None]),
            "by_class": {label: false_negatives(group) for label, group in class_groups.items()},
            "by_duration_min": {label: false_negatives(group) for label, group in duration_groups.items()},
        },
        "interval_status_counts": dict(Counter(i["status"] for i in intervals)),
        "infeasible_interval_audit": infeasible_metrics,
        "per_interval_edge_coverage": interval_coverage,
        "top_b_mean_tracts": top_b_tracts,
        "broad_component_effect": {
            "active_intervals_broad": rate(len(active) - len(local_active), len(active)),
            "active_intervals_localized": rate(len(local_active), len(active)),
            "chains_with_broad_interval": rate(
                sum(any(i["broad_component"] for i in c["intervals"]) for c in chains), len(chains)),
            "broad_interval_span_min": distribution([i["span_min"] for i in interior if i["broad_component"]]),
            "full_prism_chain_recall_without_broad": rate(sum(s["chain_rank_without_broad"] <= FULL_RANK for s in stops), len(stops)),
            "chain_union_physical_edge_coverage_without_broad": aggregate_edges("physical_edges_without_broad"),
        },
        "edge_coverage": {
            "physical_target_edge_occurrences": aggregate_edges("physical_edges"),
            "withheld_physical_target_edge_occurrences": aggregate_edges("withheld_physical_edges"),
            "chains_with_observation_target_alignment": rate(
                sum(c["evaluation"]["observations_aligned_to_target"] for c in chains), len(chains)),
        },
        "compression": {
            "baseline": {"graph_nodes": full_graph["nodes"], "csr_graph_edges": full_graph["csr_undirected_edges"],
                         "edge_vocab": full_graph["edge_vocab_size"], "tracts": graph.n_tracts,
                         "micro_zones": len(graph.rep)},
            "mean_union": means,
            "mean_fraction_retained_nodes": means["union_nodes"] / full_graph["nodes"],
            "mean_fraction_retained_csr_edges": means["union_edges"] / full_graph["csr_undirected_edges"],
            "mean_fraction_retained_vocab_edges": means["union_vocab_edge_tokens"] / full_graph["edge_vocab_size"],
            "mean_fraction_retained_tracts": means["union_tracts"] / graph.n_tracts,
            "mean_reduction_factor_nodes": full_graph["nodes"] / means["union_nodes"],
            "mean_reduction_factor_csr_edges": full_graph["csr_undirected_edges"] / means["union_edges"],
            "mean_reduction_factor_vocab_edges": full_graph["edge_vocab_size"] / means["union_vocab_edge_tokens"],
            "mean_reduction_factor_tracts": graph.n_tracts / means["union_tracts"],
            "mean_interval_nodes": float(np.mean([i["prism_nodes"] for i in intervals])),
            "mean_interval_csr_edges": float(np.mean([i["prism_edges"] for i in intervals])),
            "mean_interval_full_micro_zones": float(np.mean([i["full_candidate_micro_zones"] for i in intervals])),
            "mean_active_interval_full_micro_zones": float(np.mean([i["full_candidate_micro_zones"] for i in active])) if active else None,
            "localized_active_interval_sizes": {
                field: distribution([i[field] for i in local_active], (50, 75, 90))
                for field in ("full_candidate_micro_zones", "full_candidate_tracts", "prism_nodes", "prism_edges")
            },
            "mean_interval_top_b_micro_zones": {
                str(b): float(np.mean([min(b, len(i["top_micro_zone_ids"])) for i in intervals])) for b in TOP_B},
        },
        "per_chain_runtime_sec": {
            "workers": RUNTIME_WORKERS,
            "median": float(np.median(numbers)), "p90": float(np.percentile(numbers, 90)), "max": float(max(numbers)),
        },
    }


def percentage(metric):
    return "—" if metric["rate"] is None else f"{metric['rate']:.2%} ({metric['n']:,}/{metric['denominator']:,})"


def pretty_recall(recall, frame):
    return [percentage(recall[frame][label]) for label in [*map(str, TOP_B), "full_prism"]]


def pretty_distribution(metrics, digits):
    return " / ".join(f"{value:,.{digits}f}" if value is not None else "—" for value in metrics.values())


def markdown(results):
    config = results["parameters"]
    zones = results["network"]["micro_zones"]
    lines = [
        "# Chicago time-space prism validation", "",
        f"The corridor cache contains {results['network']['nodes']:,} graph nodes, "
        f"{results['network']['csr_undirected_edges']:,} undirected CSR edges and "
        f"{zones['micro_zones']:,} micro-zones across {zones['tracts_with_micro_zones']:,} road-served tracts "
        f"of {zones['tracts']:,}. The {zones['poi_assigned_to_micro_zone']:,} snapped and assigned POIs "
        f"average {zones['poi_per_micro_zone']:.2f} per zone and {zones['poi_per_tract']:.2f} per tract.",
        f"The bounding box yielded {zones['poi_in_bbox']:,} POIs, of which {zones['poi_within_snap_radius']:,} "
        f"snapped within {zones['poi_snap_radius_m']:g} m and {zones['poi_assigned_to_micro_zone']:,} landed in "
        "a tract micro-zone. Step 01 assigned 126,854 by joining the POI coordinates directly to tract polygons.",
        "",
        f"The last full corridor cache build took {results['network']['build_timing_seconds']['total_s']:.3f} s "
        "through the bulk array save.",
        "",
        f"Interior intervals use unwrapped AVI time and require {config['min_stop_min']:g} minutes of dwell. "
        f"Start, end and zero-observation intervals use an assumed {config['boundary_budget_min']:g}-minute "
        "reachability window because day-anchor times are unobserved. Boundaries are ineligible for transit pruning.",
        "",
        "The bidirectional travel-time graph uses optimistic class speeds, retained shared OSM vertices and virtual "
        "tract access links. A detector hit exposes the retained nodes on its edge as possible positions. "
        "Physical edge coverage requires every retained segment of a physical token to be feasible in the tested "
        "prism or union. Interval coverage uses the aligned target path for each interior gap. The primary measure "
        "uses the withheld physical edge occurrences strictly between the AVI hits. The endpoint-inclusive measure "
        "also tests the complete AVI boundary edges. Virtual links are excluded throughout.",
        "",
        "Top-B cuts are applied to each interval's ranked micro-zones before converting them to parent tracts. "
        "Interval-matched recall requires the whole stop to fit in one AVI gap, with target trip order resolving "
        "rounded timestamp ties. An unmatched stop remains in the denominator as an interval miss. "
        "Chain-union recall tests the tract anywhere in the chain, so it can count a hit even if the correct "
        "interval was pruned. The first and last activities are excluded from the intermediate-stop audit.",
        "",
    ]
    for key, name in (("all_324", "324 pure-auto chains"), ("mvp_100", "100-chain MVP subset")):
        result = results[key]
        matching = result["stop_interval_matching"]
        excluded = matching["excluded_non_activity_labels"]
        lines += [f"## {name}", "", f"{result['n_stops']:,} intermediate stops, "
                  f"{result['n_non_home_stops']:,} non-home. AVI interval matched "
                  f"{percentage(matching['matched'])}.",
                  f"Excluded non-activity labels {sum(excluded.values()):,} "
                  f"({', '.join(f'{k} {v:,}' for k, v in sorted(excluded.items())) or 'none'}). "
                  f"Stop-matching statuses were {', '.join(f'{k} {v:,}' for k, v in sorted(matching['status_counts'].items()))}.",
                  "",
                  "| Stop group | Matching | Top-10 | Top-20 | Top-30 | Top-50 | Full prism |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        recall = result["recall"]
        for label, group in [("Overall", recall["overall"]), ("Non-home", recall["non_home"]),
                             *recall["per_class"].items()]:
            for frame, title in (("interval_matched", "Interval"), ("chain_union", "Chain")):
                lines.append("| " + " | ".join([label, title, *pretty_recall(group, frame)]) + " |")
        pruning = result["transit_pruning"]
        counts = result["interval_status_counts"]
        edge = result["edge_coverage"]
        comp = result["compression"]
        times = result["per_chain_runtime_sec"]
        infeasible = result["infeasible_interval_audit"]
        broad = result["broad_component_effect"]
        lines += [
            "",
            f"- Transit-only intervals among all gaps {percentage(pruning['fraction_all_intervals'])}. "
            f"Among interior gaps {percentage(pruning['fraction_interior_intervals'])}.",
            f"- Transit-pruning false negatives {percentage(pruning['false_negative_rate'])} of all true "
            f"intermediate stops and {percentage(pruning['false_negative_rate_among_matched_stops'])} among matched stops.",
            f"- Interval statuses were {', '.join(f'{k} {v:,}' for k, v in sorted(counts.items()))}.",
            f"- Chain-union physical target-edge coverage {percentage(edge['physical_target_edge_occurrences'])}. "
            f"Withheld target-edge coverage {percentage(edge['withheld_physical_target_edge_occurrences'])}.",
            f"- Mean chain union retained {comp['mean_union']['union_nodes']:,.0f} nodes "
            f"({comp['mean_fraction_retained_nodes']:.2%} of the graph), "
            f"{comp['mean_union']['union_edges']:,.0f} CSR edges "
            f"({comp['mean_fraction_retained_csr_edges']:.2%}) and "
            f"{comp['mean_union']['union_tracts']:,.1f} candidate tracts "
            f"({comp['mean_fraction_retained_tracts']:.2%}).",
            f"- Mean chain union retained {comp['mean_union']['union_vocab_edge_tokens']:,.0f} of the "
            f"{comp['baseline']['edge_vocab']:,} vocabulary edge tokens "
            f"({comp['mean_fraction_retained_vocab_edges']:.2%}).",
            f"- Per-chain prediction runtime in seconds was {times['median']:.3f} median, "
            f"{times['p90']:.3f} p90 and {times['max']:.3f} maximum with {times['workers']} workers.",
            "",
            "### Infeasible interval audit",
            "",
            f"Travel-budget infeasible gaps account for {percentage(infeasible['fraction_interior_intervals'])} "
            f"of interior intervals and contain {percentage(infeasible['stops_in_infeasible_intervals'])} of intermediate "
            f"stops. Boundary infeasibilities total {infeasible['boundary_intervals']:,}. "
            "These gaps are tracked separately from reachable, transit-only gaps.",
            "",
            "| Infeasible interior gaps | Count | Span min, median / p90 / max | Shortest minus span min, median / p90 / max |",
            "|---|---:|---:|---:|",
        ]
        for label in ("all_infeasible", "zero_span", "positive_span"):
            group = infeasible[label]
            lines.append(f"| {label.replace('_', ' ')} | {group['count']:,} | {pretty_distribution(group['span_min'], 4)} "
                         f"| {pretty_distribution(group['shortest_minus_span_min'], 4)} |")
        lines += [
            "",
            "### Per-interval edge coverage",
            "",
            "| Interior gaps | Aligned | Withheld edge coverage | Mean per-gap withheld coverage | With AVI boundary edges |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, metrics in result["per_interval_edge_coverage"].items():
            withheld = metrics["between_avi_observations"]
            mean_gap = withheld["mean_interval_recall"]
            mean_text = f"{mean_gap:.2%}" if mean_gap is not None else "—"
            inclusive = metrics["including_avi_boundary_edges"]["physical_edge_occurrences"]
            lines.append(f"| {label.replace('_', ' ')} | {metrics['aligned_intervals']:,}/{metrics['interior_intervals']:,} "
                         f"| {percentage(withheld['physical_edge_occurrences'])} | {mean_text} "
                         f"({withheld['intervals_with_physical_edges']:,} gaps) | {percentage(inclusive)} |")
        lines += [
            "",
            "### Per-interval compression",
            "",
            f"Of intervals with at least one stop candidate, {percentage(broad['active_intervals_broad'])} "
            f"cover the entire main component and {percentage(broad['active_intervals_localized'])} use localized prisms. "
            f"Broad gaps occur in {percentage(broad['chains_with_broad_interval'])} of chains. "
            f"Their span in minutes is {pretty_distribution(broad['broad_interval_span_min'], 1)} at median / p90 / max.",
            "",
            "Broad multi-hour gaps contribute to the high chain-union full-prism recall and edge coverage by making "
            "the entire main component available to those chains. Removing the flagged broad intervals gives "
            f"{percentage(broad['full_prism_chain_recall_without_broad'])} chain-union full-prism stop recall and "
            f"{percentage(broad['chain_union_physical_edge_coverage_without_broad'])} chain-union physical edge coverage. "
            "Unflagged prisms can also span large areas.",
            "",
            "| Localized active interval | Median / p75 / p90 |",
            "|---|---:|",
        ]
        lines.extend(f"| {field.replace('_', ' ')} | {pretty_distribution(quantiles, 1)} |"
                     for field, quantiles in comp["localized_active_interval_sizes"].items())
        lines += [
            "",
            f"| Cutoff | Mean distinct tracts per active gap | Mean distinct tracts per chain | Share of {comp['baseline']['tracts']:,} tracts, per gap / per chain |",
            "|---|---:|---:|---:|",
        ]
        for b, means in result["top_b_mean_tracts"].items():
            interval_mean, chain_mean = means["per_active_interval"], means["per_chain_union"]
            total_tracts = comp["baseline"]["tracts"]
            lines.append(f"| Top-{b} | {interval_mean:.2f} | {chain_mean:.2f} "
                         f"| {interval_mean / total_tracts:.2%} / {chain_mean / total_tracts:.2%} |")
        lines += [
            "",
            "| Pruned stop audit | All true stops in group |",
            "|---|---:|",
        ]
        lines.extend(f"| {label} | {percentage(metric)} |" for label, metric in pruning["by_class"].items())
        lines.extend(f"| {label} min | {percentage(metric)} |" for label, metric in pruning["by_duration_min"].items())
        lines.append("")
    lines.extend([
        f"Total script time before writing the summary {results['runtime_seconds']['total_before_summary']:.3f} s.",
        "",
    ])
    return "\n".join(lines)


def write_atomic(path, write):
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def initialize_worker(graph):
    global worker_graph
    worker_graph = graph


def predict_chain(item):
    person_day_id, observed = item
    started = perf_counter()
    metadata, ranks, packed, memberships = worker_graph.predict(*observed)
    metadata.update(person_day_id=person_day_id, prediction_runtime_sec=perf_counter() - started,
                    observables=dict(zip(OBSERVABLES, observed)))
    return metadata, ranks, packed, memberships


def main():
    clock = perf_counter()
    cohort = json.loads(network.COHORT.read_bytes())
    if cohort["masked_test_signature"] != network.file_signature(network.MASKED_TEST):
        raise ValueError("Run step 09 again to refresh the pure-auto test cohort")
    ids = set(cohort["person_day_ids"])
    data, cache_summary = network.load_cache()
    if (cache_summary["cache_version"] != network.CACHE_VERSION
            or cache_summary["sources"] != network.source_signature()):
        raise ValueError("Run step 09 again to refresh the corridor network cache")
    graph = PrismNetwork(data, cache_summary)
    load_seconds = perf_counter() - clock

    # Only the ID (for joining) and the four protocol observables reach the
    # prediction stage. Pure-auto eligibility comes from the separate 09 audit.
    inputs = [
        (row["person_day_id"], tuple(row[key] for key in OBSERVABLES))
        for row in network.json_lines(network.MASKED_TEST) if row["person_day_id"] in ids
    ]
    if len(inputs) != 324 or len(ids) != 324:
        raise ValueError(f"Expected 324 pure-auto test person-days, got {len(inputs)} rows / {len(ids)} IDs")

    predicted, predictions_started = {}, perf_counter()
    # Workers share the immutable graph and landmarks via copy-on-write.
    with ProcessPoolExecutor(max_workers=RUNTIME_WORKERS, mp_context=mp.get_context("fork"),
                             initializer=initialize_worker, initargs=(graph,)) as pool:
        for idx, prediction in enumerate(pool.map(predict_chain, inputs), 1):
            predicted[prediction[0]["person_day_id"]] = prediction
            if idx % 50 == 0 or idx == len(inputs):
                print(f"[Prisms] {idx}/{len(inputs)} in {perf_counter() - predictions_started:.1f} s", flush=True)
    prediction_seconds = perf_counter() - predictions_started

    labels, healthcare = load_evaluation_labels()
    subset = set(json.loads(SUBSET.read_bytes())["selected_pdids"])
    if len(subset) != 100 or not subset <= ids:
        raise ValueError("MVP subset does not match the 100 preselected pure-auto person-days")
    rows = []
    evaluation_started = perf_counter()
    for rec in network.json_lines(network.MASKED_TEST):
        person_day_id = rec["person_day_id"]
        if person_day_id in predicted:
            metadata, ranks, packed, memberships = predicted.pop(person_day_id)
            metadata["evaluation"] = evaluate(rec, metadata, ranks, packed, memberships, graph, labels, healthcare)
            rows.append(metadata)
    summary_324 = summarize(rows, graph)
    summary_100 = summarize([row for row in rows if row["person_day_id"] in subset], graph)
    results = {
        "n_chains_324": len(rows),
        "n_chains_100": summary_100["n_chains"],
        "parameters": {
            "observables": list(OBSERVABLES), "top_b": [*TOP_B, "full_prism"],
            "min_stop_min": MIN_STOP_MIN, "boundary_budget_min": BOUNDARY_BUDGET_MIN,
            "poi_weight_min": POI_WEIGHT_MIN, "anchor_bonus_min": ANCHOR_BONUS_MIN,
            "timestamp_tolerance_min": TIME_TOLERANCE_MIN,
            "prediction_workers": RUNTIME_WORKERS,
            "boundary_times": "unobserved, with an assumed reachability budget",
            "full_prism_rank_marker": FULL_RANK, "miss_rank_marker": MISSING_RANK,
        },
        "network": {k: v for k, v in cache_summary.items() if k != "sources"},
        "observed_edge_coverage": cohort["observation_coverage"],
        "all_324": summary_324,
        "mvp_100": summary_100,
        "runtime_seconds": {"load_and_setup": load_seconds, "prediction_wall_time": prediction_seconds,
                            "evaluation_and_aggregation": perf_counter() - evaluation_started},
    }

    def write_rows(stream):
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")

    write_atomic(CHAIN_OUTPUT, write_rows)
    results["runtime_seconds"]["total_before_summary"] = perf_counter() - clock
    write_atomic(RESULT_JSON, lambda stream: json.dump(results, stream, allow_nan=False, indent=2))
    write_atomic(RESULT_MD, lambda stream: stream.write(markdown(results)))

    print(f"{'Cohort':<16} {'Stops':>6} {'Top-50 interval':>17} {'Top-50 union':>15} "
          f"{'Transit FN':>12} {'Own-gap edges':>16}")
    for key in ("all_324", "mvp_100"):
        r = results[key]
        print(f"{key:<16} {r['n_stops']:>6} "
              f"{r['recall']['overall']['interval_matched']['50']['rate']:>17.2%} "
              f"{r['recall']['overall']['chain_union']['50']['rate']:>15.2%} "
              f"{r['transit_pruning']['false_negative_rate']['rate']:>12.2%} "
              f"{r['per_interval_edge_coverage']['overall']['between_avi_observations']['physical_edge_occurrences']['rate']:>16.2%}")
    print(f"[10] {perf_counter() - clock:.3f} s")


if __name__ == "__main__":
    main()
