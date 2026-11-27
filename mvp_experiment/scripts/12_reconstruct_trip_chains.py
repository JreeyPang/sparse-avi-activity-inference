"""Reconstruct, route and evaluate 324 held-out Chicago sparse-AVI days."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import importlib
import json
import math
import multiprocessing as mp
import pickle
import sys
from time import perf_counter
import warnings

sys.dont_write_bytecode = True
features = importlib.import_module("11_train_observable_scorer_and_val_prisms")
prism, network = features.prism, features.network

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

CLASSES, CLASS_ID = features.CLASSES, features.CLASS_ID
N_CLASSES = len(CLASSES)
HOME, WORK, SCHOOL, MEAL = (CLASS_ID[a] for a in ("home", "work", "school", "meal"))
MODEL_NAMES = (
    "M1_Decoupled_Physical", "M2_Joint_HAPP_Uncalibrated", "M3_Joint_HAPP_InvCalibrated",
)
WEIGHT_NAMES = (
    "detour_per_minute", "stop_log_odds", "log_poi_count", "day_anchor",
    "log_activity_potential", "activity_poi_match",
    "early_start_per_30min", "late_start_per_30min", "short_dwell_per_30min",
    "long_dwell_per_30min",
)
THETA0 = np.array([1., 4., 0.75, 1.5, 3., 0.75, 0.9, 0.9, 3., 1.5])
SCHEDULE_SCALE = 30.0
REGULARIZER = 0.12
MAX_ITERS = 14
PATIENCE = 5
FALSE_STOP_MARGIN = 8.0
MISSED_STOP_MARGIN = 12.0
COUNT_CAPS = {WORK: 2, SCHOOL: 2, MEAL: 3}
CONSISTENT = (WORK, SCHOOL)
SOLVER_OPTIONS = {"threads": 1, "mip_rel_gap": 0.0}
RESULTS = network.MVP / "results"


@dataclass(frozen=True)
class Choice:
    gap: int
    candidate: int
    activity: int
    tract: str


def psi_sem(chain, gap, candidate):
    """Higher is better, in log units. Any observable-only provider can replace this callable."""
    return np.asarray(candidate["log_p_obs_debiased"]) + (1 - features.ETA) * math.log(N_CLASSES)


class Day:
    """Feasible stop assignments and distance/schedule data, derived only from the prism."""
    def __init__(self, chain, model, semantic_potential=psi_sem):
        self.chain = chain
        self.gaps = [gap for gap in chain["intervals"] if gap["candidates"]]
        bands = np.array([[model["bands"][a][field] for field in
                           ("s_low", "s_high", "d_low", "d_high", "d_min", "d_max")]
                          for a in CLASSES])
        self.choices, self.groups, earliest, latest, row_bands, static = [], [], [], [], [], []
        for gap in self.gaps:
            self.groups.append([])
            prob = np.clip(gap["p_has_stop"], 1e-8, 1 - 1e-8)
            stop_cost = float(np.log1p(-prob) - np.log(prob))
            for j, candidate in enumerate(gap["candidates"]):
                potential = semantic_potential(chain, gap, candidate)
                if np.shape(potential) != (N_CLASSES,) or not np.isfinite(potential).all():
                    raise ValueError("Semantic potential must contain nine finite scores")
                for activity in range(N_CLASSES):
                    if activity == HOME and not candidate["anchor"]:
                        continue
                    if candidate["max_dwell_min"] < bands[activity, 4]:
                        continue
                    self.groups[-1].append(len(self.choices))
                    self.choices.append(Choice(gap["index"], j, activity, candidate["tract_id"]))
                    earliest.append(candidate["earliest_arrival_min"])
                    latest.append(candidate["latest_departure_min"])
                    row_bands.append(bands[activity])
                    static.append([candidate["detour_min"], stop_cost, -math.log1p(candidate["poi_count"]),
                                   -int(candidate["anchor"]), -potential[activity],
                                   -candidate["poi_match"][activity]])
        self.earliest, self.latest = np.asarray(earliest), np.asarray(latest)
        self.bands = np.asarray(row_bands).reshape(-1, 6)
        self.static = np.asarray(static).reshape(-1, 6)
        self.vertices = self.schedule_vertices()

    def schedule_vertices(self):
        """Exact partial minimization of the four positive schedule penalties.

        At fixed dwell, clip the preferred arrival to the feasible interval.
        The resulting piecewise-linear dwell cost can change slope only at
        either duration band boundary or when latest arrival crosses s_low.
        Longer dwell and a clipped mid-band arrival break flat optima.
        """
        s_lo, s_hi, d_lo, d_hi, d_min, d_max = self.bands.T
        earliest, latest = self.earliest, self.latest
        d_max = np.minimum(d_max, latest - earliest)
        dwell = np.clip(np.column_stack([
            d_lo, d_hi, latest - s_lo, d_min, d_max,
        ]), d_min[:, None], d_max[:, None])
        arrival = np.clip(((s_lo + s_hi) / 2)[:, None], earliest[:, None], latest[:, None] - dwell)
        deviations = np.stack([
            np.maximum(s_lo[:, None] - arrival, 0.), np.maximum(arrival - s_hi[:, None], 0.),
            np.maximum(d_lo[:, None] - dwell, 0.), np.maximum(dwell - d_hi[:, None], 0.),
        ], axis=-1)
        return np.dstack([arrival, dwell, deviations / SCHEDULE_SCALE])

    def schedule(self, theta):
        cost = self.vertices[:, :, 2:] @ theta[6:]
        # Lexicographic secondary choice within each stop option. Keep the
        # exact primary schedule minimum used in the MILP and its oracle.
        minimum = cost.min(axis=1)
        tied = cost == minimum[:, None]
        best = np.argmax(np.where(tied, self.vertices[:, :, 1], -np.inf), axis=1)
        return self.vertices[np.arange(len(best)), best], minimum

    def stops(self, selected, schedule):
        results = []
        for i in selected:
            choice = self.choices[i]
            candidate = self.chain["intervals"][choice.gap]["candidates"][choice.candidate]
            arrival, duration = schedule[i, :2]
            results.append(stop_record(choice.gap, candidate, choice.activity, arrival, duration))
        return results


class Rows:
    """Assemble SciPy HiGHS CSC constraints without dense intermediate rows."""
    def __init__(self):
        self.row, self.col, self.values, self.upper, self.lower = [], [], [], [], []

    def add(self, terms, upper, lower=-np.inf):
        for col, coeff in terms:
            if coeff:
                self.row.append(len(self.upper))
                self.col.append(col)
                self.values.append(coeff)
        self.upper.append(upper)
        self.lower.append(lower)

    def build(self, columns):
        matrix = csc_matrix((self.values, (self.row, self.col)), shape=(len(self.upper), columns))
        return LinearConstraint(matrix, self.lower, self.upper)


def objective_bound(groups, choices, costs):
    """Remove assignments whose optimistic objective loses to a feasible day."""
    options = [sorted(group, key=costs.__getitem__) for group in groups]
    floors = [min(0., costs[group[0]]) for group in options]
    # Greedy selections supply an upper bound only. The optimistic lower
    # bound relaxes daily constraints, so this pruning preserves the optimum.
    chosen_tracts, counts, upper = {}, Counter(), 0.
    for j in sorted(range(len(options)), key=floors.__getitem__):
        for index in options[j]:
            choice = choices[index]
            a = choice.activity
            if a in COUNT_CAPS and counts[a] >= COUNT_CAPS[a]:
                continue
            if a in CONSISTENT and chosen_tracts.get(a, choice.tract) != choice.tract:
                continue
            upper += costs[index]
            counts[a] += 1
            if a in CONSISTENT:
                chosen_tracts[a] = choice.tract
            break
    room = upper - sum(floors) + 1e-7
    return [[i for i in group if costs[i] <= floor + room]
            for floor, group in zip(floors, options, strict=True)]


def solve(day, theta, adjustment=None):
    """Exact reduced HAPP with continuous schedules, gated penalties and daily coupling."""
    schedule, schedule_cost = day.schedule(theta)
    static_cost = day.static @ theta[:6]
    if adjustment is not None:
        static_cost = static_cost + adjustment
    total_cost = static_cost + schedule_cost

    # Each group has one y, or skip. For equal contributions to daily
    # constraints, all but the cheapest choice are provably dominated.
    # Reapply for every loss-augmented objective as dominance can change.
    keep = []
    for group in day.groups:
        best = {}
        for i in group:
            choice = day.choices[i]
            if total_cost[i] > 0.0:  # replacing by skip can only relax daily caps
                continue
            key = (choice.activity, choice.tract) if choice.activity in CONSISTENT else (
                choice.activity if choice.activity == MEAL else None)
            if key not in best or total_cost[i] < total_cost[best[key]]:
                best[key] = i
        if best:
            keep.append(list(best.values()))
    keep = [group for group in objective_bound(keep, day.choices, total_cost) if group]
    if not keep:
        return [], schedule, 0.0

    objective, integrality, upper_bounds = [], [], []
    rows = Rows()

    def variable(cost=0., upper=np.inf, binary=False):
        col = len(objective)
        objective.append(cost)
        upper_bounds.append(upper)
        integrality.append(int(binary))
        return col

    indicator = {}
    tract_terms = {}
    counts = {a: [] for a in COUNT_CAPS}
    selection_columns = {}
    for group in keep:
        gap = day.chain["intervals"][day.choices[group[0]].gap]
        lower, span = gap["lower_min"], gap["span_min"]
        z_skip = variable(upper=1., binary=True)
        y = {i: variable(static_cost[i], upper=1., binary=True) for i in group}
        selection_columns.update(y)
        rows.add([(z_skip, 1.), *((col, 1.) for col in y.values())], 1., 1.)
        arrival = variable(upper=span)
        active_classes = sorted({day.choices[i].activity for i in group})
        dwell = {a: variable(upper=span) for a in active_classes}
        early, late, short, long = [variable(weight) for weight in theta[6:]]
        rows.add([(arrival, -1.), *((y[i], day.earliest[i] - lower) for i in group)], 0.)
        rows.add([(arrival, 1.), *((col, 1.) for col in dwell.values()),
                  *((y[i], lower - day.latest[i]) for i in group)], 0.)
        # Tight perspective Big-M gates set arrival, dwell and the penalty
        # lower bounds to zero on skip. Bands are selected by the class
        # marginal sum(y[m,u,a]); they incur no phantom cost on other classes.
        rows.add([(arrival, -1. / SCHEDULE_SCALE), (early, -1.),
                  *((y[i], (day.bands[i, 0] - lower) / SCHEDULE_SCALE) for i in group)], 0.)
        rows.add([(arrival, 1. / SCHEDULE_SCALE), (late, -1.),
                  *((y[i], (lower - day.bands[i, 1]) / SCHEDULE_SCALE) for i in group)], 0.)
        rows.add([(short, -1.), *((col, -1. / SCHEDULE_SCALE) for col in dwell.values()),
                  *((y[i], day.bands[i, 2] / SCHEDULE_SCALE) for i in group)], 0.)
        rows.add([(long, -1.), *((col, 1. / SCHEDULE_SCALE) for col in dwell.values()),
                  *((y[i], -day.bands[i, 3] / SCHEDULE_SCALE) for i in group)], 0.)
        for activity in active_classes:
            opts = [i for i in group if day.choices[i].activity == activity]
            rows.add([(dwell[activity], -1.), *((y[i], day.bands[i, 4]) for i in opts)], 0.)
            rows.add([(dwell[activity], 1.), *((y[i], -min(span, day.bands[i, 5])) for i in opts)], 0.)
        # Exact convex-hull objective bound after per-choice schedule
        # minimization. This strengthens the root LP without dropping
        # continuous variables or changing the integer optimum.
        deviations = (early, late, short, long)
        rows.add([*((j, -weight) for j, weight in zip(deviations, theta[6:])),
                  *((y[i], schedule_cost[i]) for i in group)], 0.)
        for i in group:
            choice = day.choices[i]
            if choice.activity in CONSISTENT:
                key = choice.activity, choice.tract
                if key not in indicator:
                    indicator[key] = variable(upper=1., binary=True)
                    tract_terms[key] = []
                tract_terms[key].append((y[i], 1.))
                rows.add([(y[i], 1.), (indicator[key], -1.)], 0.)
            if choice.activity in COUNT_CAPS:
                counts[choice.activity].append((y[i], 1.))

    for key, terms in tract_terms.items():
        cap = COUNT_CAPS[key[0]]
        if len(terms) > cap:
            rows.add([*terms, (indicator[key], -cap)], 0.)
    for activity in CONSISTENT:
        cols = [(col, 1.) for (a, _), col in indicator.items() if a == activity]
        if len(cols) > 1:
            rows.add(cols, 1.)
    for activity, terms in counts.items():
        if len(terms) > COUNT_CAPS[activity]:
            rows.add(terms, COUNT_CAPS[activity])
    constraints = rows.build(len(objective))
    bounds = Bounds(np.zeros(len(objective)), upper_bounds)
    started = perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected", category=RuntimeWarning)
        result = milp(objective, integrality=integrality, bounds=bounds,
                      constraints=constraints, options=SOLVER_OPTIONS)
    ms = 1000 * (perf_counter() - started)
    if not result.success:
        raise RuntimeError(f"HiGHS failed for {day.chain['person_day_id']}: {result.message}")
    selected = sorted(i for i, col in selection_columns.items() if result.x[col] > 0.5)
    if not math.isclose(result.fun, total_cost[selected].sum(), rel_tol=1e-6, abs_tol=1e-5):
        raise RuntimeError("HiGHS objective disagrees with exact schedule minimization")
    return selected, schedule, ms


def stop_record(gap, c, activity, arrival, duration):
    return {
        "interval": gap, "zone_id": c["zone_id"], "node": c["node"], "tract_id": c["tract_id"],
        "activity": CLASSES[activity], "arrival_min": float(arrival),
        "departure_min": float(arrival + duration), "duration_min": float(duration),
    }


def decoupled(day, model, threshold):
    predictions = []
    for gap in day.gaps:
        if gap["p_has_stop"] < threshold:
            continue
        candidate = gap["candidates"][0]
        activity = int(np.argmax(candidate["p_obs"]))
        band = model["bands"][CLASSES[activity]]
        duration = min(candidate["max_dwell_min"], band["d_high"])
        arrive = np.clip((band["s_low"] + band["s_high"]) / 2, candidate["earliest_arrival_min"],
                         candidate["latest_departure_min"] - duration)
        predictions.append(stop_record(gap["index"], candidate, activity, arrive, duration))
    return predictions


def supervision(day, truth, alpha):
    """Graded weighted end-to-end task loss, relative to skipping each gap."""
    groups = {}
    for stop in truth:
        groups.setdefault(stop["interval"], []).append(stop)
    values = []
    for option in day.choices:
        target = groups.get(option.gap)
        if not target:
            values.append(FALSE_STOP_MARGIN)
            continue
        skip = MISSED_STOP_MARGIN * max(alpha[t["activity"]] for t in target)
        # With multiple surveyed stops in one gap, reward the best recovered
        # one. Penalties for additional missed stops are assignment constants.
        credit = max(
            MISSED_STOP_MARGIN * alpha[t["activity"]] * (
                0.15 + 0.35 * (option.tract == t["tract_id"])
                + 0.30 * (CLASSES[option.activity] == t["activity"])
                + 0.20 * (option.tract == t["tract_id"] and CLASSES[option.activity] == t["activity"])
            )
            for t in target
        )
        values.append(-min(credit, skip))
    return np.asarray(values)


def feature_sum(day, selection, schedule):
    return np.r_[day.static[selection].sum(axis=0), schedule[selection, 2:].sum(axis=0)]


def prf(tp, predicted, actual):
    return {"tp": int(tp), "predicted": int(predicted), "actual": int(actual),
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / actual if actual else 0.0,
            "f1": 2 * tp / (predicted + actual) if predicted + actual else 0.0}


def select_stop_threshold(chains, truth):
    """Select M1's threshold using the cached OOF p_has_stop on all validation days."""
    probabilities, targets = [], []
    n_actual = 0
    for chain in chains:
        occupied = {s["interval"] for s in truth[chain["person_day_id"]] if s["interval"] is not None}
        n_actual += len(occupied)
        for gap in chain["intervals"]:
            if gap["candidates"]:
                probabilities.append(gap["p_has_stop"])
                targets.append(gap["index"] in occupied)
    p, y = np.asarray(probabilities), np.asarray(targets, dtype=bool)
    grid = np.arange(15, 76) / 100
    selected = p[None, :] >= grid[:, None]
    predicted = selected.sum(axis=1)
    hits = (selected & y).sum(axis=1)
    best = int(np.argmax(2 * hits / (predicted + n_actual)))
    return float(grid[best]), prf(hits[best], predicted[best], n_actual)


def calibration_metrics(predictions, truth):
    n_stop = n_pred_gaps = n_true_gaps = n_triple = n_pred = n_true = 0
    actual, predicted, hits = Counter(), Counter(), Counter()
    for pred, real in zip(predictions, truth, strict=True):
        pg, rg = ({x["interval"] for x in group if x["interval"] is not None} for group in (pred, real))
        n_stop += len(pg & rg)
        n_pred_gaps += len(pg)
        n_true_gaps += len(rg)
        n_pred += len(pred)
        n_true += len(real)
        actual.update(t["activity"] for t in real)
        predicted.update(t["activity"] for t in pred)
        got = {(t["interval"], t["activity"]) for t in real}
        hits.update(t["activity"] for t in pred if (t["interval"], t["activity"]) in got)
        expected = {(t["interval"], t["tract_id"], t["activity"]) for t in real}
        n_triple += sum((p["interval"], p["tract_id"], p["activity"]) in expected for p in pred)
    macro = float(np.mean([prf(hits[a], predicted[a], actual[a])["f1"] for a in CLASSES[1:]]))
    sf1 = prf(n_stop, n_pred_gaps, n_true_gaps)["f1"]
    triple = prf(n_triple, n_pred, n_true)["f1"]
    return macro, sf1, triple


def calibrate(days, truth):
    """Projected proximal subgradient on the 1-slack mean structured hinge.

    For feasible fixed y_ref, the oracle minimizes C_theta(y) - L(y).
    The hinge is C_theta(y_ref) - C_theta(y*) + L(y*) - L(y_ref).
    Its subgradient is Phi(y_ref) - Phi(y*). A proximal step on the
    ridge penalty centered at theta0 and box projection fix theta_tt=1.
    """
    counts = Counter(t["activity"] for stops in truth for t in stops)
    n = sum(counts.values())
    alpha = {a: math.sqrt(n / (N_CLASSES * counts[a])) for a in CLASSES}
    losses = [supervision(day, real, alpha) for day, real in zip(days, truth, strict=True)]
    references = []
    for day, loss in zip(days, losses, strict=True):
        # Fixed feasible reference surrogate: minimize C_theta0 + 1000 L
        # under the same day constraints. The reference stays fixed while
        # scoring its features on the original cost scale.
        selection, sched, _ = solve(day, THETA0, adjustment=1000 * loss)
        references.append((feature_sum(day, selection, sched), loss[selection].sum()))
    theta = THETA0.copy()
    best_score, best_theta, best_step = (-np.inf, -np.inf), theta.copy(), 0
    trace = []
    for step in range(MAX_ITERS):
        gradient = np.zeros_like(theta)
        total_hinge = 0.
        predictions = []
        oracle_times = []
        for day, loss, (reference, ref_loss) in zip(days, losses, references, strict=True):
            adversary, sched, ms = solve(day, theta, adjustment=-loss)
            delta = reference - feature_sum(day, adversary, sched)
            hinge = float(theta @ delta + loss[adversary].sum() - ref_loss)
            if hinge < -1e-5:
                raise RuntimeError("Loss-augmented oracle failed to dominate its feasible reference")
            total_hinge += max(0., hinge)
            if hinge > 1e-9:
                gradient += delta
            if ms:
                oracle_times.append(ms)
            prediction, sched, _ = solve(day, theta)
            predictions.append(day.stops(prediction, sched))
        macro, stop_f1, triple = calibration_metrics(predictions, truth)
        mean_hinge = total_hinge / len(days)
        objective = mean_hinge + 0.5 * REGULARIZER * np.sum((theta - THETA0) ** 2)
        score = ((macro + stop_f1 + triple) / 3, -objective)
        if score > best_score:
            best_score, best_theta, best_step = score, theta.copy(), step
        trace.append({
            "iteration": step, "mean_suboptimality_loss": mean_hinge, "regularized_objective": float(objective),
            "val_non_home_interval_activity_macro_f1": macro, "val_stop_f1": stop_f1,
            "val_triple_f1": triple, "checkpoint_score": score[0],
            "weights": weight_dict(theta), "oracle_milp_ms": {
                **prism.distribution(oracle_times), "n_calls": len(oracle_times),
                "n_pruned_without_solve": len(days) - len(oracle_times),
            },
        })
        print(f"[12] val {step:02d} hinge {mean_hinge:.3f}, stop {stop_f1:.3f}, "
              f"non-home macro {macro:.3f}, triple {triple:.3f}", flush=True)
        if step >= 6 and step - best_step >= PATIENCE:
            break
        step_size = 0.9 / math.sqrt(1 + step)
        updated = (theta - step_size * gradient / len(days) + step_size * REGULARIZER * THETA0) / (
            1 + step_size * REGULARIZER)
        theta = np.clip(updated, [1, 0, 0, 0, 0, 0, .02, .02, .02, .02],
                        [1, 40, 20, 20, 40, 20, 25, 25, 25, 25])
    return best_theta, {"best_iteration": best_step, "alpha": alpha, "val_counts": dict(counts),
                        "ridge": REGULARIZER, "trace": trace, "theta0": weight_dict(THETA0),
                        "theta_star": weight_dict(best_theta)}


def weight_dict(theta):
    return dict(zip(WEIGHT_NAMES, map(float, theta), strict=True))


class Router:
    """Recover continuous graph walks through sensor tokens and chosen zone nodes."""
    def __init__(self, data, summary):
        self.data = data
        self.n = summary["nodes"]
        self.graph = csr_matrix((data["graph_data"], data["graph_indices"], data["graph_indptr"]),
                                shape=(self.n, self.n))
        self.xy = (data["node_lonlat"] - network.ORIGIN) * network.SCALE_M
        self.grid = network.SpatialGrid(self.xy)
        u, v, w = (data[key] for key in ("segment_u", "segment_v", "segment_min"))
        self.axis_speed = np.array([np.max(np.abs(self.xy[u, j] - self.xy[v, j]) / w) for j in (0, 1)])
        tokens = np.full(len(u), -1, dtype=np.int32)
        vocab_lengths = np.diff(data["vocab_segment_offsets"])
        tokens[data["vocab_segment_ids"]] = np.repeat(np.arange(len(vocab_lengths)), vocab_lengths)
        physical = data["vocab_is_physical"]
        self.tokens = tokens
        self.physical = physical
        self.segment_keys = u.astype(np.int64) * self.n + v
        self.tract_index = {tract: i for i, tract in enumerate(data["tract_ids"])}

    def anchor(self, tract):
        idx = self.tract_index.get(tract)
        if idx is None:
            return ()
        offsets = self.data["anchor_offsets"]
        return tuple(map(int, self.data["anchor_nodes"][offsets[idx]:offsets[idx + 1]]))

    def directions(self, token):
        offsets = self.data["vocab_pair_offsets"]
        pairs = self.data["vocab_node_pairs"][offsets[token]:offsets[token + 1]]
        if not len(pairs):
            raise ValueError(f"AVI token {token} has no real network endpoints")
        return sorted({(int(a), int(b)) for u, v in pairs for a, b in ((u, v), (v, u))})

    @lru_cache(maxsize=32)
    def window(self, x0, y0, x1, y1):
        low, high = 1000. * np.array([[x0, y0], [x1, y1]])
        ids = self.grid.query(low, high)
        ids = np.sort(ids[np.all((self.xy[ids] >= low) & (self.xy[ids] <= high), axis=1)])
        return ids, self.graph[ids, :][:, ids].tocsr()

    def path_tokens(self, nodes):
        if len(nodes) < 2:
            return [], 0
        left, right = nodes[:-1], nodes[1:]
        keys = np.minimum(left, right).astype(np.int64) * self.n + np.maximum(left, right)
        pos = np.searchsorted(self.segment_keys, keys)
        if not np.array_equal(self.segment_keys[pos], keys):
            raise RuntimeError("Dijkstra returned an unknown segment")
        tokens = self.tokens[pos]
        keep = (tokens >= 0) & self.physical[tokens]
        # Keep virtual/unmapped segment counts to expose projected gaps.
        return collapse(tokens[keep].tolist()), int(np.count_nonzero(~keep))

    def routes(self, sources, targets, hint=0.):
        """Minimum time from any source to each target, with path predecessors."""
        sources, targets = np.asarray(sources), np.asarray(targets)
        if not len(sources) or not len(targets):
            return {}
        components = self.data["component"]
        reachable = np.isin(components[targets], components[sources])
        targets = targets[reachable]
        if not len(targets):
            return {}
        if len(targets) == 1 and int(targets[0]) in sources:
            return {int(targets[0]): (0., [], 0)}
        l_min, l_max = self.xy[sources].min(axis=0), self.xy[sources].max(axis=0)
        r_min, r_max = self.xy[targets].min(axis=0), self.xy[targets].max(axis=0)
        separation = np.maximum(np.maximum(l_min - r_max, r_min - l_max), 0.)
        cutoff = max(.25, float((separation / self.axis_speed).max()) * 2 + .10, hint)
        while True:
            local = cutoff <= 9
            if local:
                margin = self.axis_speed * cutoff + 1e-5
                low = np.floor((l_min + r_min - margin) / 2000.).astype(int)
                high = np.ceil((l_max + r_max + margin) / 2000.).astype(int)
                # Include even disconnected/far source endpoints so local IDs
                # stay defined; only feasible paths can influence the result.
                low = np.minimum(low, np.floor(np.minimum(l_min, r_min) / 1000.).astype(int))
                high = np.maximum(high, np.ceil(np.maximum(l_max, r_max) / 1000.).astype(int))
                ids, graph = self.window(*low, *high)
                start, end = np.searchsorted(ids, sources), np.searchsorted(ids, targets)
            else:
                graph, start, end = self.graph, sources, targets
            # Cache geometry, rather than storing dense predecessor trees.
            distance, predecessor, roots = dijkstra(
                graph, indices=start, min_only=True, return_predecessors=True,
                limit=cutoff + prism.EPS if local else np.inf,
            )
            if np.isfinite(distance[end]).all():
                break
            if not local:
                raise RuntimeError("Network component disagrees with unbounded Dijkstra")
            cutoff *= 2
        result = {}
        for target, terminal in zip(targets, end, strict=True):
            current = terminal
            root = roots[current]
            nodes = [current]
            while current != root:
                current = predecessor[current]
                nodes.append(current)
            nodes = np.asarray(nodes[::-1], dtype=np.int32)
            tokens, connectors = self.path_tokens(ids[nodes] if local else nodes)
            result[int(target)] = (float(distance[terminal]), tokens, connectors)
        return result

    def recover(self, chain, models):
        observed = chain["observables"]
        edges = observed["observed_edges"]
        hits = [self.directions(edge) for edge in edges]
        left_anchor = self.anchor(observed["known_day_start_tract_id"])
        right_anchor = self.anchor(observed["known_day_end_tract_id"])
        # Leave anchoring censored within its tract. Every retained AVI state
        # chooses an actual traversal direction of that observed road edge.
        missing = (np.inf, [], 0)
        route_cache = {}

        def routes(left, right, hint):
            key = left, right
            if key not in route_cache:
                route_cache[key] = self.routes(left, right, hint)
            return route_cache[key]

        for model in models.values():
            stops = {stop["interval"]: stop for stop in model["stops"]}
            history = []
            # Lexicographic objective: avoid breaks, favor AVI/schedule
            # consistency, then minimize total travel among the orientations.
            previous = [(0, 0, 0., 0.)]
            for i, gap in enumerate(chain["intervals"]):
                incoming = [left_anchor] if i == 0 else [(out,) for _, out in hits[i - 1]]
                outgoing = [right_anchor] if i == len(hits) else [(entry,) for entry, _ in hits[i]]
                goal_nodes = tuple(sorted({node for group in outgoing for node in group}))
                stop = stops.get(i)
                node = stop["node"] if stop else None
                hint = min(8., (gap["min_travel_min"] or 0.) + .15)
                if stop:
                    after = routes((node,), goal_nodes, hint / 2)
                choices = []
                for left in incoming:
                    if stop:
                        before = routes(left, (node,), hint / 2).get(node, missing)
                    else:
                        paths = routes(left, goal_nodes, hint)
                    options = []
                    for group in outgoing:
                        if stop:
                            tail = min((after.get(z, missing) for z in group), default=missing)
                            cost = before[0] + tail[0]
                            options.append((cost, collapse(before[1] + tail[1]), before[2] + tail[2],
                                            before[0], tail[0]))
                        else:
                            cost, tokens, connectors = min((paths.get(z, missing) for z in group), default=missing)
                            options.append((cost, tokens, connectors, None, None))
                    choices.append(options)
                new, back = [], []
                for j in range(len(outgoing)):
                    def cost(k):
                        travel, _, _, tin, tout = choices[k][j]
                        breaks, violations, excess, minutes = previous[k]
                        if not np.isfinite(travel):
                            return breaks + 1, violations, excess, minutes
                        if stop:
                            over = max(0., gap["lower_min"] + tin - stop["arrival_min"]) + max(
                                0., stop["departure_min"] + tout - gap["upper_min"])
                        else:
                            over = max(0., travel - gap["span_min"])
                        violation = over > prism.TIME_TOLERANCE_MIN
                        return breaks, violations + int(violation), excess + (over if violation else 0.), minutes + travel
                    k = min(range(len(incoming)), key=cost)
                    new.append(cost(k))
                    back.append((k, choices[k][j]))
                previous = new
                history.append(back)
            used, directions, state = [], [], 0
            for i in range(len(history) - 1, -1, -1):
                if i < len(hits):
                    directions.append(hits[i][state])
                state, path = history[i][state]
                used.append(path)
            used.reverse()
            model["avi_direction_nodes"] = directions[::-1]
            all_edges, gap_paths, breaks, connectors, late, schedule_conflicts = [], [], [], 0, [], []
            for i, (travel, tokens, n_connectors, tin, tout) in enumerate(used):
                gap = chain["intervals"][i]
                stop = stops.get(i)
                if not np.isfinite(travel):
                    breaks.append(i)
                elif travel + (stop["duration_min"] if stop else 0.) > gap["span_min"] + prism.TIME_TOLERANCE_MIN:
                    late.append(i)
                if stop and np.isfinite(travel) and (
                    gap["lower_min"] + tin > stop["arrival_min"] + prism.TIME_TOLERANCE_MIN
                    or stop["departure_min"] + tout > gap["upper_min"] + prism.TIME_TOLERANCE_MIN
                ):
                    schedule_conflicts.append(i)
                connectors += n_connectors
                gap_paths.append(tokens)
                all_edges.extend(tokens)
                if i < len(edges) and self.physical[edges[i]]:
                    all_edges.append(edges[i])  # traversed as the oriented AVI arc above
            model["physical_edge_sequence"] = collapse(all_edges)
            model["interval_physical_edge_sequences"] = gap_paths
            model["interval_travel_min"] = [path[0] if np.isfinite(path[0]) else None for path in used]
            model["route_break_intervals"] = breaks
            model["nonphysical_or_unmapped_segments"] = connectors
            model["route_exceeds_gap_intervals"] = late
            model["route_schedule_conflict_intervals"] = schedule_conflicts
        return models


def collapse(tokens):
    return [token for i, token in enumerate(tokens) if i == 0 or tokens[i - 1] != token]


def recover_chain(item):
    chain, models = item
    return ROUTER.recover(chain, models)


def count_matches(pred, real, fields):
    def keys(stops):
        return Counter(tuple(s[k] for k in fields) for s in stops if s["interval"] is not None)
    return sum((keys(pred) & keys(real)).values())


def edit_stats(left, right):
    """Levenshtein distance and LCS length for short daily stop sequences."""
    edits = list(range(len(right) + 1))
    matches = [0] * (len(right) + 1)
    for i, item in enumerate(left, 1):
        next_edit, next_match = [i], [0]
        for j, target in enumerate(right, 1):
            next_edit.append(min(edits[j] + 1, next_edit[-1] + 1, edits[j - 1] + (item != target)))
            next_match.append(matches[j - 1] + 1 if item == target else max(matches[j], next_match[-1]))
        edits, matches = next_edit, next_match
    return edits[-1], matches[-1]


def distance_km(left, right, centroids):
    if left not in centroids or right not in centroids:
        return None
    a, b = np.radians([centroids[left], centroids[right]])
    delta = (b - a) / 2
    term = np.sin(delta[1]) ** 2 + np.cos(a[1]) * np.cos(b[1]) * np.sin(delta[0]) ** 2
    return float(2 * network.EARTH_M / 1000 * np.arcsin(np.sqrt(np.clip(term, 0, 1))))


def route_overlap(pred, actual):
    return len(pred & actual), len(pred), len(actual)


def sum_prf(rows):
    return prf(*map(sum, zip(*rows))) if rows else prf(0, 0, 0)


def mean(values):
    return float(np.mean(values)) if values else None


def pairs_for_schedule(pred, real):
    by_gap = {}
    for act in real:
        if act["interval"] is not None:
            by_gap.setdefault(act["interval"], []).append(act)
    matches = []
    for stop in pred:
        target = by_gap.get(stop["interval"])
        if target:
            # Same-tract match when available, then the first chronological
            # stop in that interval. Predicted time and purpose are never used.
            matches.append((stop, min(target, key=lambda x: (x["tract_id"] != stop["tract_id"],
                                                           x["activity_index"]))))
    return matches


def activity_metrics(pairs):
    counts = Counter((r["activity"], p["activity"]) for p, r in pairs)
    actual = Counter(r["activity"] for _, r in pairs)
    predicted = Counter(p["activity"] for p, _ in pairs)
    per_class = {a: prf(counts[a, a], predicted[a], actual[a]) for a in CLASSES}
    non_home = [(p, r) for p, r in pairs if r["activity"] != "home"]
    predicted_nh = Counter(p["activity"] for p, _ in non_home)
    nh = {a: prf(counts[a, a], predicted_nh[a], actual[a]) for a in CLASSES[1:]}
    return {
        "matched_count": len(pairs), "non_home_matched_count": len(non_home),
        "accuracy": mean([p["activity"] == r["activity"] for p, r in pairs]),
        "non_home_accuracy": mean([p["activity"] == r["activity"] for p, r in non_home]),
        "non_home_macro_f1": mean([value["f1"] for value in nh.values()]),
        "per_class": per_class,
    }


def coupling(rows):
    changed, ambiguous, ambiguous_stop_disagreement = [], [], []
    added = removed = active = 0
    for row in rows:
        first, _, third = [row["predictions"][name] for name in MODEL_NAMES]
        m1, m3 = [{s["interval"]: s for s in m["stops"]} for m in (first, third)]
        truth = {}
        for stop in row["intermediate_truth"]:
            truth.setdefault(stop["interval"], set()).add(stop["tract_id"])
        observed = set(row["observables"]["observed_edges"])
        for index, n_tracts in enumerate(row["n_candidate_tracts"]):
            if not n_tracts:
                continue
            active += 1
            a, b = m1.get(index), m3.get(index)
            added += int(a is None and b is not None)
            removed += int(a is not None and b is None)
            x, y = a["tract_id"] if a else None, b["tract_id"] if b else None
            different = a is not None and b is not None and x != y
            if not (different or n_tracts >= 2):
                continue
            targets = truth.get(index, ())
            actual_gap = row["withheld_truth_by_gap"][index]
            overlap = None if actual_gap is None else [
                route_overlap(set(model["interval_physical_edge_sequences"][index]) - observed, actual_gap)
                for model in (first, third)
            ]
            detail = (row["person_day_id"], x in targets, y in targets, actual_gap, overlap, (a is not None, b is not None))
            if different:
                changed.append(detail)
            if n_tracts >= 2:
                ambiguous.append(detail)
                if (a is None) != (b is None):
                    ambiguous_stop_disagreement.append(detail)

    def describe(items):
        corrections = sum(not left and right for _, left, right, *_ in items)
        harms = sum(left and not right for _, left, right, *_ in items)
        comparable = [item for item in items if item[3] is not None]
        m1, m3 = [sum_prf([item[4][i] for item in comparable]) for i in (0, 1)]
        ids = {item[0] for item in items}
        chain = [sum_prf([row["path_metrics"][name]["withheld"] for row in rows
                          if row["person_day_id"] in ids]) for name in (MODEL_NAMES[0], MODEL_NAMES[2])]
        return {
            "intervals": len(items), "chains": len(ids),
            "intervals_where_both_predict_stop": sum(all(flags) for *_, flags in items),
            "m3_added_stops": sum(flags == (False, True) for *_, flags in items),
            "m3_removed_stops": sum(flags == (True, False) for *_, flags in items),
            "m3_corrects_m1_wrong_tract": corrections, "m3_hurts_m1_correct_tract": harms,
            "correction_rate": corrections / len(items) if items else None,
            "harm_rate": harms / len(items) if items else None,
            "net_correct_tracts": corrections - harms,
            "withheld_interval_path_f1_m1": m1["f1"], "withheld_interval_path_f1_m3": m3["f1"],
            "withheld_interval_path_f1_delta": m3["f1"] - m1["f1"],
            "withheld_interval_path_aligned_count": len(comparable),
            "withheld_chain_path_f1_delta": chain[1]["f1"] - chain[0]["f1"],
        }

    return {"active_intervals": active, "m3_added_stops": added, "m3_removed_stops": removed,
            "different_tract_when_both_stop": describe(changed), "ambiguous_2plus_tracts": describe(ambiguous),
            "ambiguous_stop_existence_disagreement": describe(ambiguous_stop_disagreement)}


def summarize(rows, centroids):
    all_counts, found = Counter(), Counter()
    for row in rows:
        for stop in row["intermediate_truth"]:
            all_counts[stop["activity"]] += 1
            gap = stop["interval"]
            found[stop["activity"]] += bool(gap is not None and
                                             stop["tract_id"] in row["candidate_tracts"][gap])
    active_tract_counts = [n for row in rows for n in row["n_candidate_tracts"] if n]
    missing_start = [row["truth_start"]["activity"] is None for row in rows]
    missing_end = [row["truth_end"]["activity"] is None for row in rows]
    missing_either = sum(a or b for a, b in zip(missing_start, missing_end, strict=True))
    result = {
        "n_chains": len(rows), "n_true_intermediate_stops": sum(all_counts.values()),
        "truth_label_audit": {
            "chains_missing_start_label": sum(missing_start),
            "chains_missing_end_label": sum(missing_end),
            "chains_missing_either_anchor_label": missing_either,
            "chains_missing_both_anchor_labels": sum(a and b for a, b in zip(missing_start, missing_end, strict=True)),
            "chains_with_both_anchor_labels": len(rows) - missing_either,
            "dropped_intermediate_activities": sum(row["n_dropped_intermediate_activities"] for row in rows),
        },
        "n_true_unassigned_stops": sum(t["interval"] is None for row in rows for t in row["intermediate_truth"]),
        "candidate_recall": {
            "overall": prism.rate(sum(found.values()), sum(all_counts.values())),
            "by_class": {a: prism.rate(found[a], all_counts[a]) for a in CLASSES},
            "mean_distinct_tracts_per_active_interval": mean(active_tract_counts),
            "active_intervals": len(active_tract_counts),
            "interval_status_counts": dict(Counter(s for row in rows for s in row["interval_statuses"])),
        },
        "models": {}, "coupling_audit": coupling(rows),
    }
    for name in MODEL_NAMES:
        stop_counts, exact, triple, order, all_path, withheld_path = [], [], [], [], [], []
        stop_differences, pairs, similarity, similarity_valid_anchors, runtimes = [], [], [], [], []
        breaks = connectors = late = conflict = n_pruned = 0
        for row in rows:
            model = row["predictions"][name]
            pred, real = model["stops"], row["intermediate_truth"]
            pg = {p["interval"] for p in pred}
            rg = {t["interval"] for t in real if t["interval"] is not None}
            stop_counts.append((len(pg & rg), len(pg), len(rg)))
            stop_differences.append(abs(len(pred) - len(real)))
            exact.append((count_matches(pred, real, ("interval", "tract_id")), len(pred), len(real)))
            triple.append((count_matches(pred, real, ("interval", "tract_id", "activity")), len(pred), len(real)))
            _, lcs = edit_stats([p["tract_id"] for p in pred], [t["tract_id"] for t in real])
            order.append((lcs, len(pred), len(real)))
            pairs.extend(pairs_for_schedule(pred, real))
            sequence = model["activity_sequence"]
            truth = [r["activity"] if r["activity"] is not None else "<missing>" for r in row["truth_chain"]]
            edits, _ = edit_stats(sequence, truth)
            daily_similarity = 1 - edits / max(len(sequence), len(truth))
            similarity.append(daily_similarity)
            if row["truth_start"]["activity"] is not None and row["truth_end"]["activity"] is not None:
                similarity_valid_anchors.append(daily_similarity)
            p = row["path_metrics"][name]
            all_path.append(p["physical"])
            withheld_path.append(p["withheld"])
            ms = model["milp_solve_ms"]
            if ms == 0:
                n_pruned += 1
            elif ms is not None:
                runtimes.append(ms)
            breaks += len(model["route_break_intervals"])
            late += len(model["route_exceeds_gap_intervals"])
            conflict += len(model["route_schedule_conflict_intervals"])
            connectors += model["nonphysical_or_unmapped_segments"]
        distances, wrong_tract_distances = [], []
        for pred_stop, true_stop in pairs:
            pred_tract, true_tract = pred_stop["tract_id"], true_stop["tract_id"]
            distance = distance_km(pred_tract, true_tract, centroids)
            if distance is not None:
                distances.append(distance)
                if pred_tract != true_tract:
                    wrong_tract_distances.append(distance)
        result["models"][name] = {
            "stop_detection": sum_prf(stop_counts), "daily_stop_count_mae": mean(stop_differences),
            "exact_interval_tract": sum_prf(exact), "chain_order_tract": sum_prf(order),
            "median_tract_centroid_distance_error_km": float(np.median(distances)) if distances else None,
            "mean_tract_centroid_distance_error_km": mean(distances),
            "p75_tract_centroid_distance_error_km": float(np.percentile(distances, 75)) if distances else None,
            "median_wrong_tract_centroid_distance_error_km": (
                float(np.median(wrong_tract_distances)) if wrong_tract_distances else None
            ),
            "centroid_distance_n": len(distances), "wrong_tract_centroid_distance_n": len(wrong_tract_distances),
            "matched_arrival_mae_min": mean([abs(p["arrival_min"] - r["arrival_min"]) for p, r in pairs
                                            if r["arrival_min"] is not None]),
            "matched_duration_mae_min": mean([abs(p["duration_min"] - r["duration_min"]) for p, r in pairs
                                             if r["duration_min"] is not None]),
            "triple_match": sum_prf(triple), "matched_stop_activity": activity_metrics(pairs),
            "daily_activity_sequence_similarity": mean(similarity),
            "daily_activity_sequence_similarity_valid_anchors": mean(similarity_valid_anchors),
            "physical_edge_set": sum_prf(all_path), "withheld_physical_edge_set": sum_prf(withheld_path),
            "route_audit": {"broken_intervals": breaks, "nonphysical_or_unmapped_segments": connectors,
                            "travel_plus_dwell_exceeds_interval": late, "recovered_route_schedule_conflicts": conflict},
            "milp_solve_ms": {**prism.distribution(runtimes), "n_calls": len(runtimes),
                              "n_pruned_without_solve": n_pruned,
                              "n_over_50ms": sum(ms > 50 for ms in runtimes)},
        }
    return result


def evaluate(test, predictions, physical):
    """The sole test truth read. Invoked only after all models and paths are final."""
    if len(test) != 324 or len(predictions) != 324:
        raise ValueError("Refusing to inspect held-out truth before all predictions exist")
    test_chains = {chain["person_day_id"]: chain for chain in test}
    if set(predictions) != set(test_chains) or any(set(p) != set(MODEL_NAMES) for p in predictions.values()):
        raise ValueError("Missing finalized day/model predictions")
    labels = features.label_rows("test")[0]
    subset = set(json.loads(prism.SUBSET.read_text())["selected_pdids"])
    if len(subset) != 100 or not subset <= set(predictions):
        raise ValueError("Fixed MVP selection must contain 100 held-out IDs")
    tract_features = json.loads(network.TRACTS.read_text())["features"]
    centroids = {r["properties"]["GEOID"]: (float(r["properties"]["Lon_Centroid"]),
                                          float(r["properties"]["Lat_Centroid"])) for r in tract_features}
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
        path_metrics = {}
        for name, model in pred.items():
            reconstructed = set(model["physical_edge_sequence"])
            if not all(physical[e] for e in reconstructed):
                raise ValueError(f"Nonphysical token in model output for {key}")
            path_metrics[name] = {"physical": route_overlap(reconstructed, actual),
                                  "withheld": route_overlap(reconstructed - observed, actual - observed)}
        if alignment is not None:
            boundaries = [-1, *alignment, len(targets)]
            gap_paths = [{t for t in targets[a + 1:b] if physical[t] and t not in observed}
                         for a, b in zip(boundaries, boundaries[1:])]
        else:
            gap_paths = [None for _ in chain["intervals"]]

        truth_start, truth_end = surveyed["truth_start"], surveyed["truth_end"]
        candidate_tracts = [{c["tract_id"] for c in gap["candidates"]} for gap in chain["intervals"]]
        rows.append({
            "person_day_id": key, "mvp_100": key in subset, "observables": chain["observables"],
            "truth_start": truth_start, "truth_end": truth_end,
            "truth_chain": [truth_start, *truth, truth_end], "intermediate_truth": truth,
            "n_dropped_intermediate_activities": surveyed["n_dropped_intermediate_activities"],
            "predictions": pred,
            "path_metrics": path_metrics, "candidate_tracts": candidate_tracts,
            "n_candidate_tracts": [len(c) for c in candidate_tracts],
            "interval_statuses": [gap["status"] for gap in chain["intervals"]],
            "withheld_truth_by_gap": gap_paths, "alignment_ok": alignment is not None,
        })
    if len(rows) != len(test_chains):
        raise ValueError("Held-out truth join is incomplete")
    summaries = {"all_324": summarize(rows, centroids),
                 "mvp_100": summarize([r for r in rows if r["mvp_100"]], centroids)}
    return rows, summaries


def percent(value):
    return "—" if value is None else f"{value:.2%}"


def number(value):
    return "—" if value is None else f"{value:.3f}"


def report(data):
    protocol = data["protocol"]
    stop_fit = protocol["stop_cross_fit"]
    lines = [
        "# End-to-end sparse AVI reconstruction",
        "",
        "The household-disjoint split has 1,351 training days, 276 validation days, "
        "and 324 held-out days. The 100-day cohort uses the fixed step 05 selection. "
        "Predictions use AVI tokens and unwrapped timestamps, start/end tracts, and weekday/weekend "
        "from the survey calendar join on day ID. Training labels and schedule bands are from "
        "the canonical step 02 activities.",
        "",
        f"The validation stop probabilities come from {stop_fit['n_splits']}-fold GroupKFold grouped by "
        f"household_id across {stop_fit['full_fit_households']} households. Each day is scored by a "
        "stop classifier trained on the other four folds. The M1 threshold, M3 separation oracle and "
        "checkpoint selection all use those out-of-fold probabilities. A separate stop classifier fit "
        "on all 276 validation days scores the test candidates. Threshold and checkpoint metrics summarize "
        "development-set selection using OOF inputs.",
        "",
        f"M1 selected a threshold of **{protocol['stop_probability_threshold_m1']:.2f}** on the "
        f"OOF validation probabilities (stop F1 **{percent(protocol['m1_oof_validation_stop_detection']['f1'])}**).",
        "",
        "| Stop fold | Fit households | Held-out households | Fit days | Held-out days | Fit active gaps | Held-out active gaps |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *(f"| {f['fold']} | {f['fit_households']} | {f['held_out_households']} | {f['fit_days']} | "
          f"{f['held_out_days']} | {f['fit_active_intervals']} | {f['held_out_active_intervals']} |"
          for f in stop_fit["folds"]),
        "",
        "Step 11 fits activity probabilities on observable candidate features in the labeled training "
        "tract. It learns 20th–80th percentile preferred time bands and 1st–99th percentile "
        "hard dwell bounds. Interior candidate dwell is at least 1.5 minutes. Boundaries use a "
        "15-minute road-access budget and assumed day bounds. The top 20 candidates include at most two zones "
        "per tract. Shortest-path travel time is in minutes. Euclidean chord distance in metres is an "
        "additional straight-line path-length proxy.",
        "",
        "M1 applies the calibrated physical gap threshold, picks the top physical zone and assigns the raw "
        "activity argmax. M2 jointly selects skip or one stop per AVI interval, with continuous arrival and "
        "class-gated dwell, train-derived soft schedule penalties, consistent work/school tracts and daily caps. "
        "This includes intervals over 120 minutes. M3 uses the same model with validation-calibrated weights. "
        "The joint models prefer longer dwell among equal-cost schedule vertices, preserving the primary MILP "
        "objective. M1 uses the feasible dwell up to the upper preferred duration band.",
        "",
        "Routing chooses a continuous sequence of orientations along the observed road tokens and uses "
        "exact shortest graph legs through each selected stop. Among orientations it first minimizes "
        "disconnections and schedule conflicts, then travel time. The graph permits virtual road-access links. "
        "Only physical tokens enter link scores. The route audit counts virtual/unmapped segments, "
        "disconnections and timing conflicts between specific oriented road endpoints and the optimistic prism.",
        "",
        "Stop detection counts occupied AVI gaps. Spatial and triple scores use one-to-one exact keys "
        "with all intermediate stops in the denominator. Chain-order tract F1 uses the longest common "
        "subsequence. Schedule, centroid and purpose metrics pair stops in the same interval, choosing "
        "the exact tract when available and the first surveyed stop when a tie remains. "
        "Non-home classification metrics are conditioned on non-home truth. Sequence similarity uses the original "
        "start and end slots, with an explicit `<missing>` token for each unavailable anchor in its all-day metric. "
        "The valid-anchor-only metric uses days with both anchor labels available. Unlabeled intermediate activities "
        "are excluded from stop and sequence metrics and counted below. Path PRF pools within-day link-set counts. "
        "Withheld scoring subtracts "
        "all given tokens from both predicted and true sets.",
    ]
    display = [
        ("Stop detection", "stop_detection"), ("Exact interval + tract", "exact_interval_tract"),
        ("Chain-order tract", "chain_order_tract"), ("Interval + tract + activity", "triple_match"),
        ("Physical link set", "physical_edge_set"), ("Withheld physical link set", "withheld_physical_edge_set"),
    ]
    for key, title in (("all_324", "All 324 days"), ("mvp_100", "Fixed 100-day MVP subset")):
        group = data[key]
        cand = group["candidate_recall"]
        missing = group["truth_label_audit"]
        lines.extend([
            "", f"## {title}", "",
            f"{group['n_true_intermediate_stops']} scored intermediate stops, "
            f"{group['n_true_unassigned_stops']} without an assignable AVI interval. "
            f"The {cand['active_intervals']:,} active intervals average "
            f"{cand['mean_distinct_tracts_per_active_interval']:.2f} candidate tracts.",
            "",
            f"Missing survey labels affected {missing['chains_missing_start_label']} start anchors and "
            f"{missing['chains_missing_end_label']} end anchors. {missing['chains_missing_either_anchor_label']} days "
            f"had at least one missing anchor and {missing['chains_missing_both_anchor_labels']} had both missing. "
            f"The valid-anchor-only sequence denominator is {missing['chains_with_both_anchor_labels']} days. "
            f"Excluded intermediate activities with missing or non-activity labels numbered "
            f"{missing['dropped_intermediate_activities']}.",
            "",
            "| Candidate tract recall in true interval | Hits / true stops | Recall |",
            "|---|---:|---:|",
        ])
        for label, item in [("Overall", cand["overall"]), *cand["by_class"].items()]:
            lines.append(f"| {label} | {item['n']} / {item['denominator']} | {percent(item['rate'])} |")
        if key == "all_324":
            escort = cand["by_class"]["escort"]
            lines.extend([
                "",
                f"The prior step 10 Top-20 audit recovered 90/135 escort tracts (66.67%) in the "
                f"true gap and averaged 4.82 tracts per active gap. Here the corresponding "
                f"escort change is {(escort['rate'] - 90 / 135) * 100:+.2f} percentage points.",
            ])
        lines += ["", "| Metric | Model | Precision | Recall | F1 | TP / predicted / true |",
                  "|---|---|---:|---:|---:|---:|"]
        for metric, field in display:
            for name, model in group["models"].items():
                r = model[field]
                lines.append(f"| {metric} | {name[:2]} | {percent(r['precision'])} | {percent(r['recall'])} | "
                             f"{percent(r['f1'])} | {r['tp']:,} / {r['predicted']:,} / {r['actual']:,} |")
        m1, m2, m3 = (group["models"][name] for name in MODEL_NAMES)

        def delta(before, after, metric):
            change = (after[metric]["f1"] - before[metric]["f1"]) * 100
            return f"{change:+.2f} pp"

        lines.extend([
            "",
            f"M2 changes stop F1 by {delta(m1, m2, 'stop_detection')}, strict triple F1 by "
            f"{delta(m1, m2, 'triple_match')}, and withheld physical F1 by "
            f"{delta(m1, m2, 'withheld_physical_edge_set')} relative to M1. "
            f"Validation calibration changes those three scores by "
            f"{delta(m2, m3, 'stop_detection')}, {delta(m2, m3, 'triple_match')}, and "
            f"{delta(m2, m3, 'withheld_physical_edge_set')} relative to M2.",
        ])
        lines += [
            "", "| Matched-stop and day metrics | M1 | M2 | M3 |",
            "|---|---:|---:|---:|",
        ]
        for label, field, formatter in [
            ("Daily stop count MAE", "daily_stop_count_mae", number),
            ("Median tract centroid error (km)", "median_tract_centroid_distance_error_km", number),
            ("Mean tract centroid error (km)", "mean_tract_centroid_distance_error_km", number),
            ("P75 tract centroid error (km)", "p75_tract_centroid_distance_error_km", number),
            ("Median tract centroid error, wrong tract only (km)", "median_wrong_tract_centroid_distance_error_km", number),
            ("Arrival MAE (min)", "matched_arrival_mae_min", number),
            ("Duration MAE (min)", "matched_duration_mae_min", number),
            ("Daily activity sequence similarity, all days", "daily_activity_sequence_similarity", percent),
            ("Daily activity sequence similarity, both anchor labels valid",
             "daily_activity_sequence_similarity_valid_anchors", percent),
        ]:
            lines.append(f"| {label} | " + " | ".join(formatter(m[field]) for m in group["models"].values()) + " |")
        for label, field in [
            ("Matched activity accuracy", "accuracy"), ("Matched non-home activity accuracy", "non_home_accuracy"),
            ("Matched non-home macro F1", "non_home_macro_f1"),
        ]:
            lines.append(f"| {label} | " + " | ".join(percent(m["matched_stop_activity"][field])
                                                     for m in group["models"].values()) + " |")
        lines += ["", "| Matched activity class F1 (true support) | M1 | M2 | M3 |", "|---|---:|---:|---:|"]
        for label in CLASSES:
            values = [m["matched_stop_activity"]["per_class"][label] for m in group["models"].values()]
            lines.append(f"| {label} | " + " | ".join(f"{percent(v['f1'])} ({v['actual']})" for v in values) + " |")
        tail_changes = []
        for label in ("school", "healthcare"):
            base_f1, fitted_f1 = (m["matched_stop_activity"]["per_class"][label]["f1"] for m in (m2, m3))
            direction = "rose" if fitted_f1 > base_f1 else "fell" if fitted_f1 < base_f1 else "was unchanged"
            tail_changes.append(f"{label} {direction} ({percent(base_f1)} → {percent(fitted_f1)})")
        lines.extend([
            "",
            f"From M2 to M3, matched F1 for {tail_changes[0]} and for {tail_changes[1]}. "
            "Tail weighting changed the validation loss used to fit shared model weights. "
            "Class-specific evidence came from observable activity scores and POI features, without LLM priors. "
            "Matched class F1 also depends on which stops each model recovers.",
        ])
        audit = group["coupling_audit"]
        disagreement = audit["ambiguous_stop_existence_disagreement"]
        lines += [
            "", "### Bidirectional coupling", "",
            f"M3 added {audit['m3_added_stops']} and removed {audit['m3_removed_stops']} stops relative to M1 "
            f"across {audit['active_intervals']:,} active gaps. The two-stop comparison isolates tract changes. "
            f"Within the ambiguous gaps, M3 added {disagreement['m3_added_stops']} stops and "
            f"removed {disagreement['m3_removed_stops']}.",
            "", "| Gaps | Count | Corrected | Harmed | Net | Withheld gap F1 M1 → M3 | Δ gap F1 | Δ whole-day F1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for label, key in (
            ("Both stop, changed tract", "different_tract_when_both_stop"),
            ("At least two candidate tracts", "ambiguous_2plus_tracts"),
            ("Ambiguous and disagree on stop existence", "ambiguous_stop_existence_disagreement"),
        ):
            a = audit[key]
            lines.append(
                f"| {label} | {a['intervals']:,} | {a['m3_corrects_m1_wrong_tract']} ({percent(a['correction_rate'])}) | "
                f"{a['m3_hurts_m1_correct_tract']} ({percent(a['harm_rate'])}) | {a['net_correct_tracts']:+} | "
                f"{percent(a['withheld_interval_path_f1_m1'])} → {percent(a['withheld_interval_path_f1_m3'])} | "
                f"{a['withheld_interval_path_f1_delta'] * 100:+.2f} pp | "
                f"{a['withheld_chain_path_f1_delta'] * 100:+.2f} pp |"
            )
        lines += [
            "", "HiGHS timing percentiles use solved calls only. Days with trivially pruned MILPs are counted separately.",
            "", "| Model | HiGHS median / p90 / max (ms) | Solved | Pruned | Calls >50 ms | Broken gaps | "
            "Virtual/unmapped segments | Travel + dwell over gap | Route–schedule conflicts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, model in group["models"].items():
            times, route = model["milp_solve_ms"], model["route_audit"]
            distribution = " / ".join(number(times[k]) for k in ("median", "p90", "max"))
            lines.append(f"| {name[:2]} | {distribution} | {times['n_calls']} | {times['n_pruned_without_solve']} | "
                         f"{times['n_over_50ms']} | {route['broken_intervals']} | "
                         f"{route['nonphysical_or_unmapped_segments']} | {route['travel_plus_dwell_exceeds_interval']} | "
                         f"{route['recovered_route_schedule_conflicts']} |")

    cal = data["calibration"]
    lines += [
        "", "## Validation-only inverse optimization", "",
        "The margin-rescaled oracle minimizes reconstruction cost minus weighted task loss in HiGHS. "
        "Precomputed per-choice schedule vertices give the exact minimum schedule cost for each weight vector. "
        "The fixed feasible reference surrogate solves the same day constraints with the objective "
        "C(θ₀) + 1,000 × task loss. Each projected proximal subgradient update targets the mean structured hinge "
        "with a ridge penalty centered on θ₀. "
        "The OOF-input validation checkpoint score averages occupied-gap F1, strict triple F1 and end-to-end non-home "
        "interval/activity macro F1, with regularized loss breaking ties. "
        f"Selected iteration **{cal['best_iteration']}**. Detour is fixed at one equivalent minute per minute.",
        "", "| Cost term | M2 | M3 |", "|---|---:|---:|",
    ]
    for key, value in cal["theta_star"].items():
        lines.append(f"| {key} | {cal['theta0'][key]:.4f} | {value:.4f} |")
    lines += ["", "| Validation class | Count | Tail weight |", "|---|---:|---:|"]
    for a in CLASSES:
        lines.append(f"| {a} | {cal['val_counts'][a]} | {cal['alpha'][a]:.3f} |")
    lines += [
        "", "| Iteration | Mean hinge | Ridge objective | Val macro F1 | Val stop F1 | Val triple F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in cal["trace"]:
        lines.append(
            f"| {r['iteration']} | {r['mean_suboptimality_loss']:.4f} | {r['regularized_objective']:.4f} | "
            f"{percent(r['val_non_home_interval_activity_macro_f1'])} | {percent(r['val_stop_f1'])} | "
            f"{percent(r['val_triple_f1'])} |"
        )
    lines += ["", "## Runtime", "", "| Stage | Seconds |", "|---|---:|"]
    for stage, value in data["runtime_seconds"].items():
        lines.append(f"| {stage.replace('_', ' ')} | {value:.3f} |")
    return "\n".join(lines) + "\n"


def clock_time(value):
    if value is None:
        return "unknown"
    day, minute = divmod(int(round(value)), 1440)
    hour, minute = divmod(minute, 60)
    return f"{hour:02d}:{minute:02d}" + (f" (+{day}d)" if day else "")


def case_report(rows):
    first, _, third = MODEL_NAMES

    def score(row, name):
        pred = row["predictions"][name]["stops"]
        real = row["intermediate_truth"]
        return count_matches(pred, real, ("interval", "tract_id", "activity"))

    def path_f1(row, name):
        return prf(*row["path_metrics"][name]["withheld"])["f1"]

    def contains(row, label):
        return any(t["activity"] == label for t in row["intermediate_truth"])

    def repaired(row):
        m1 = {t["interval"]: t for t in row["predictions"][first]["stops"]}
        m3 = {t["interval"]: t for t in row["predictions"][third]["stops"]}
        return sum(t["interval"] in m1 and t["interval"] in m3
                   and m1[t["interval"]]["tract_id"] != t["tract_id"]
                   and m3[t["interval"]]["tract_id"] == t["tract_id"]
                   and m3[t["interval"]]["activity"] == t["activity"] for t in row["intermediate_truth"])

    picks, used = [], set()

    def pick(title, eligible, rank, explanation):
        options = [r for r in rows if r["person_day_id"] not in used and eligible(r)]
        if options:
            row = max(options, key=rank)
            used.add(row["person_day_id"])
            picks.append((title, explanation, row))

    pick("Home–work–home commuter", lambda r: [t["activity"] for t in r["truth_chain"]] == ["home", "work", "home"],
         lambda r: (score(r, third), path_f1(r, third)),
         "A single surveyed work stop between the two day anchors.")
    pick("Multi-stop shopping and meal day", lambda r: contains(r, "shop") and contains(r, "meal")
         and len(r["intermediate_truth"]) >= 3, lambda r: (score(r, third), path_f1(r, third)),
         "Several surveyed stops test whether the joint model can recover the day order.")
    # Keep the reviewed healthcare day in the report as an anchor regression case.
    for label in ("healthcare", "escort"):
        pick(f"{label.capitalize()} day", lambda r: contains(r, label),
             lambda r: (label == "healthcare" and r["person_day_id"] == "240014960101",
                        sum(t["activity"] == label and any(
                 p["activity"] == label and p["interval"] == t["interval"] and p["tract_id"] == t["tract_id"]
                 for p in r["predictions"][third]["stops"]) for t in r["intermediate_truth"]),
                        score(r, third), path_f1(r, third)),
             f"The surveyed chain contains at least one {label} stop.")
    pick("Joint model corrects a location and identifies its activity", lambda r: repaired(r) > 0,
         lambda r: (repaired(r), score(r, third) - score(r, first), path_f1(r, third) - path_f1(r, first)),
         "At least one shared predicted stop moves from an incorrect tract to the surveyed tract with the correct purpose.")
    pick("Failure where M1 has more exact stops", lambda r: score(r, first) > score(r, third),
         lambda r: (score(r, first) - score(r, third), path_f1(r, first) - path_f1(r, third)),
         "M3 loses at least one exact interval, tract and purpose match relative to M1.")
    pick("Failure on a sparse multi-stop day", lambda r: len(r["intermediate_truth"]) >= 3 and score(r, third) == 0,
         lambda r: (-len(r["observables"]["observed_edges"]), -path_f1(r, third)),
         "M3 recovers no exact triples on this surveyed multi-stop day.")

    lines = [
        "# Held-out sparse AVI case studies", "",
        "Examples were chosen after scoring and include the previously reviewed healthcare day. "
        "The cases include corrections and failures. "
        "The home anchors below show the known tracts. Their clocks are censored in the reconstruction. "
        "Each AVI pair shows physical edge vocabulary ID at the given timestamp. Gap 0 precedes the first "
        "hit and gap k follows hit k. Clock times are rounded to one minute for display.",
    ]
    for title, description, row in picks:
        inputs = row["observables"]
        hits = [f"`{edge}@{clock_time(time)}`" for edge, time in zip(inputs["observed_edges"], inputs["observed_time_min"])]
        pred = row["predictions"]
        lines += [
            "", f"## {title}", "",
            f"Day **{row['person_day_id']}**, {'weekend' if inputs['travel_dow'] in (6, 7) else 'weekday'}. {description}",
            "",
            f"AVI inputs · start tract `{inputs['known_day_start_tract_id']}`, "
            f"end tract `{inputs['known_day_end_tract_id']}`, {len(hits)} hits.",
            "",
            " ".join(hits) if hits else "No AVI hits.",
            "", "| AVI gap | Surveyed stops | M1 physical baseline | M3 calibrated joint model |",
            "|---|---|---|---|",
        ]

        def display(stops):
            if not stops:
                return "—"
            return "<br>".join(f"{s['activity'] or 'label unavailable'} · `{s['tract_id'] or 'unknown'}` · "
                               f"{clock_time(s['arrival_min'])}–{clock_time(s['departure_min'])}"
                               + (f" · {s['duration_min']:.1f} min" if s["duration_min"] is not None else "")
                               for s in stops)

        lines.append(f"| Day start | {display([row['truth_start']])} | "
                     f"{display([pred[first]['day_anchors'][0]])} | {display([pred[third]['day_anchors'][0]])} |")
        all_stops = [row["intermediate_truth"], pred[first]["stops"], pred[third]["stops"]]
        gaps = sorted({t["interval"] for group in all_stops for t in group}, key=lambda n: -1 if n is None else n)
        for gap in gaps:
            lines.append(f"| {gap if gap is not None else 'Unassigned'} | "
                         + " | ".join(display([t for t in group if t["interval"] == gap]) for group in all_stops) + " |")
        lines.append(f"| Day end | {display([row['truth_end']])} | "
                     f"{display([pred[first]['day_anchors'][1]])} | {display([pred[third]['day_anchors'][1]])} |")
        lines.extend([
            "", f"Exact triples · M1 {score(row, first)}, M3 {score(row, third)}, "
            f"surveyed intermediate stops {len(row['intermediate_truth'])}. "
            f"Withheld physical link F1 · M1 {percent(path_f1(row, first))}, M3 {percent(path_f1(row, third))}.",
        ])
    return "\n".join(lines) + "\n"


def main(semantic_potential=psi_sem):
    started = perf_counter()
    build = json.loads((features.CACHE / "build_manifest.json").read_text())
    if build["signature"] != features.signature():
        raise ValueError("Stage 11 caches are stale. Run stage 11 again.")
    with features.MODEL_FILE.open("rb") as stream:
        model = pickle.load(stream)
    val = list(network.json_lines(features.candidate_file("val")))
    if len(val) != 276:
        raise ValueError("Expected 276 validation prism records")
    val_labels, val_households = features.label_rows("val")
    val_truth = features.supervised_truth("val", val_labels)
    cross_fit = build["stop_cross_fit"]
    if (cross_fit != model["stop_cross_fit"]
            or cross_fit["splitter"] != "GroupKFold" or cross_fit["group"] != "household_id"
            or cross_fit["n_splits"] != features.STOP_FOLDS
            or cross_fit["score_sources"] != {"val": "out_of_fold", "test": "full_validation_fit"}
            or set(cross_fit["fold_by_household"]) != val_households
            or {c["person_day_id"] for c in val} != set(val_truth)):
        raise ValueError("Expected household-grouped OOF stop scores for validation")
    threshold, threshold_metrics = select_stop_threshold(val, val_truth)
    print(f"[12] M1 OOF validation threshold {threshold:.2f}, stop F1 {threshold_metrics['f1']:.4f}", flush=True)
    before_cal = perf_counter()
    theta, calibration = calibrate([Day(c, model, semantic_potential) for c in val],
                                   [val_truth[c["person_day_id"]] for c in val])
    calibration["validation_stop_score_source"] = cross_fit["score_sources"]["val"]
    after_cal = perf_counter()
    print(f"[12] calibration {after_cal - before_cal:.2f}s, selected iteration "
          f"{calibration['best_iteration']} with weights {weight_dict(theta)}", flush=True)
    test = list(network.json_lines(features.candidate_file("test")))
    if len(test) != 324 or len({c["person_day_id"] for c in test}) != 324:
        raise ValueError("Expected 324 held-out prism records")
    if any(set(c["observables"]) != set(features.OBSERVABLES) for c in [*val, *test]):
        raise ValueError("Unexpected input fields in candidate cache")
    decisions = []
    for chain in test:
        day = Day(chain, model, semantic_potential)
        models = {MODEL_NAMES[0]: {"stops": decoupled(day, model, threshold), "milp_solve_ms": None}}
        for name, weights in zip(MODEL_NAMES[1:], (THETA0, theta), strict=True):
            selection, schedule, ms = solve(day, weights)
            models[name] = {"stops": day.stops(selection, schedule), "milp_solve_ms": ms}
        for prediction in models.values():
            prediction["activity_sequence"] = ["home", *(s["activity"] for s in prediction["stops"]), "home"]
            prediction["day_anchors"] = [
                {"activity": "home", "tract_id": chain["observables"][key],
                 "arrival_min": None, "departure_min": None, "duration_min": None}
                for key in ("known_day_start_tract_id", "known_day_end_tract_id")
            ]
        decisions.append((chain, models))
    after_decision = perf_counter()
    print(f"[12] {len(decisions)} days of stop and schedule decisions in {after_decision - after_cal:.2f}s", flush=True)

    global ROUTER
    data, net_summary = network.load_cache()
    ROUTER = Router(data, net_summary)
    predictions = {}
    with ProcessPoolExecutor(max_workers=features.WORKERS, mp_context=mp.get_context("fork")) as pool:
        for (chain, _), models in zip(decisions, pool.map(recover_chain, decisions, chunksize=2), strict=True):
            predictions[chain["person_day_id"]] = models
            if len(predictions) % 100 == 0 or len(predictions) == len(test):
                print(f"[12] recovered paths {len(predictions)}/{len(test)} in "
                      f"{perf_counter() - after_decision:.1f}s", flush=True)
    after_route = perf_counter()
    rows, summaries = evaluate(test, predictions, data["vocab_is_physical"])
    output = {
        "n_chains_324": len(rows), "n_chains_100": summaries["mvp_100"]["n_chains"],
        "protocol": {
            "prediction_observables": features.OBSERVABLES, "split_days": features.EXPECTED,
            "stop_probability_threshold_m1": threshold, "m1_oof_validation_stop_detection": threshold_metrics,
            "stop_cross_fit": cross_fit, "debiased_activity_eta": model["eta"],
            "activity_sequence_similarity": {
                "all_days_missing_anchor_token": "<missing>",
                "also_report_only_days_with_both_anchor_labels": True,
                "unlabeled_intermediate_activities": "excluded and counted",
            },
            "short_stop_min": features.MIN_DWELL, "max_candidates_per_tract": features.TRACT_CAP,
            "max_candidates_per_interval": features.TOP_B, "boundary_road_access_budget_min": features.BOUNDARY_ACCESS,
            "max_stops_per_interval": 1, "activity_classes": CLASSES,
            "activity_features": model["activity_features"], "stop_features": model["stop_features"],
            "activity_fit_class_counts": model["activity_class_counts"], "train_schedule_bands": model["bands"],
            "path_workers": features.WORKERS, "unaligned_avi_chains": sum(not r["alignment_ok"] for r in rows),
        },
        **summaries, "calibration": calibration,
        "runtime_seconds": {"stage_11_build": build["runtime_seconds"],
                            "stop_cross_fit_and_full_fit": build["stop_cross_fit_and_full_fit_seconds"],
                            "stage_12_initial_load": before_cal - started,
                            "validation_calibration": after_cal - before_cal,
                            "test_stop_schedule_decisions": after_decision - after_cal,
                            "network_load_and_path_recovery": after_route - after_decision},
        "chains": [
            {k: r[k] for k in ("person_day_id", "mvp_100", "observables", "truth_start", "truth_end",
                                "truth_chain", "n_dropped_intermediate_activities", "predictions", "path_metrics")}
            for r in rows
        ],
    }
    prism.write_atomic(RESULTS / "e2e_case_studies.md", lambda f: f.write(case_report(rows)))
    output["runtime_seconds"]["evaluation_and_case_studies"] = perf_counter() - after_route
    output["runtime_seconds"]["stage_12_before_report_write"] = perf_counter() - started
    prism.write_atomic(RESULTS / "e2e_reconstruction_324chains.json",
                       lambda f: json.dump(output, f, allow_nan=False, separators=(",", ":")))
    output["runtime_seconds"]["stage_12_through_json_write"] = perf_counter() - started
    prism.write_atomic(RESULTS / "e2e_reconstruction_summary.md", lambda f: f.write(report(output)))
    for name, r in summaries["all_324"]["models"].items():
        print(f"[12] {name} stop F1 {r['stop_detection']['f1']:.4f}, triple F1 {r['triple_match']['f1']:.4f}, "
              f"withheld path F1 {r['withheld_physical_edge_set']['f1']:.4f}")
    print(f"[12] {perf_counter() - started:.3f}s total", flush=True)


if __name__ == "__main__":
    main()
