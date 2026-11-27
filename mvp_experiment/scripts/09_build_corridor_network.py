"""Build a bidirectional, lower-bound Chicago routing graph and 350 m micro-zones.

Distances follow the road polylines. Speeds in km/h are optimistic envelopes
for reachability and are not estimates of observed speed or legal access.
"""
from __future__ import annotations

import argparse
import ast
from array import array
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import struct
import tempfile
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

SCRIPTS = Path(__file__).resolve().parent
MVP = SCRIPTS.parent
CHICAGO = MVP.parent / "Chicago-Trip"
RAW = CHICAGO / "data" / "Rawdata" / "Chicago"
STRICT = CHICAGO / "data" / "processed_fixed" / "strict_chicago_all_usable_10pct"
MASKED_TEST = STRICT / "masked_test.jsonl"
EDGE_VOCAB = STRICT / "edge_vocab.json"
ROADS = RAW / "Transportation Network" / "road_tract_network.geojsonl"
TRACTS = CHICAGO / "Chicago" / "Tracts" / "TractZonalHistogram.geojson"
TRACT_LINKS = TRACTS.with_name("tract_virtual_links.geojsonl")
POI_DIR = MVP / "data" / "osm_il"
POI_LAYERS = tuple(
    f"gis_osm_{family}{suffix}_free_1"
    for family in ("pois", "pofw", "transport")
    for suffix in ("", "_a")
)
CACHE = MVP / "data" / "corridor_network" / "network.npz"
COHORT = CACHE.with_name("pure_auto_test_cohort.json")
CACHE_VERSION = 2

# Optimistic passenger-car envelopes in km/h: freeway 145 (~90 mph),
# arterial 85–120, and local/parallel corridors about 70–80 (~45–50 mph).
# Map-matched footways, cycleways and paths can proxy parallel roads in auto
# data, so their weights use the local-road envelope, independent of test
# paths, observed times and posted pedestrian/bicycle speeds.
# A posted motor speed can raise the envelope; virtual connectors have their
# own access weights.
CLASS_KMH = {
    "motorway": 145, "motorway_link": 105, "trunk": 120, "trunk_link": 95,
    "primary": 105, "primary_link": 85, "secondary": 95, "secondary_link": 75,
    "tertiary": 85, "tertiary_link": 70, "unclassified": 85, "residential": 70,
    "living_street": 80, "service": 80, "busway": 85,
    "cycleway": 80, "footway": 80, "path": 80, "pedestrian": 80, "steps": 80,
    "track": 80, "track_grade1": 80, "track_grade2": 80, "track_grade3": 80,
    "track_grade4": 80, "track_grade5": 80, "bridleway": 80,
    "tract_driving": 70, "tract_walk&cycle": 45,
}
OTHER_KMH = 80
EXCLUDED = {"transit_walk", "tract_walk"}
VIRTUAL_FCLASSES = {"transit_walk", "tract_walk", "tract_driving", "tract_walk&cycle"}
MAXSPEED_FACTOR = 1.25
ZONE_M = 350.0
SNAP_M = 300.0
# Local equirectangular coordinates at 42 N. A 300 m snap uses a unit-sphere
# chord distance instead, so longitude scale varies with latitude correctly.
ORIGIN = np.array([-89.0, 41.0])
SCALE_M = np.array([111195.08 * np.cos(np.deg2rad(42.0)), 111195.08])
COORD_PRECISION = 10_000_000
COORD_STRIDE = 100_000_000
EARTH_M = 6_371_008.8


def literal_settings(path, names):
    """Reuse the upstream literal tables without importing GIS or ML code."""
    module = ast.parse(path.read_text(encoding="utf-8"))
    return {
        stmt.targets[0].id: ast.literal_eval(stmt.value)
        for stmt in module.body
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id in names
    }


def json_lines(path):
    with path.open("rb") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def pure_auto(record, auto_modes):
    trips = record["ground_truth_chain"]["trips"]
    return bool(trips) and {trip["mode"]["label"] for trip in trips} <= auto_modes


def source_signature():
    paths = [ROADS, TRACTS, TRACT_LINKS, EDGE_VOCAB, SCRIPTS / "01_fetch_chicago_poi.py"]
    paths.extend(POI_DIR / (name + ext) for name in POI_LAYERS for ext in (".shp", ".dbf"))
    signature = {}
    for path in paths:
        stat = path.stat()
        signature[str(path.relative_to(MVP.parent))] = [stat.st_size, stat.st_mtime_ns]
    model = {
        "class_kmh": CLASS_KMH, "other_kmh": OTHER_KMH, "maxspeed_factor": MAXSPEED_FACTOR,
        "zone_m": ZONE_M, "snap_m": SNAP_M, "excluded_fclasses": sorted(EXCLUDED),
        "virtual_fclasses": sorted(VIRTUAL_FCLASSES), "origin": ORIGIN.tolist(),
        "scale_m": SCALE_M.tolist(), "coordinate_precision": COORD_PRECISION,
        "coordinate_stride": COORD_STRIDE, "earth_m": EARTH_M,
    }
    encoded = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    return {"files": signature, "model_constants_sha256": sha256(encoded).hexdigest()}


def coordinate_keys(lonlat):
    grid = np.rint((lonlat - ORIGIN) * COORD_PRECISION).astype(np.int64)
    return grid[:, 0] * COORD_STRIDE + grid[:, 1]


def read_road_parts(vocab):
    coordinates, sizes = array("d"), array("i")
    speeds, tokens = array("f"), array("i")
    physical_parts = bytearray()
    feature_counts = Counter()
    physical_tokens = np.fromiter(
        (" & " not in key for key in sorted(vocab, key=vocab.get)), dtype=bool,
    )
    for feature in json_lines(ROADS):
        props, geometry = feature["properties"], feature["geometry"]
        fclass = props["fclass"]
        physical = fclass not in VIRTUAL_FCLASSES
        raw_id = str(props["osm_id"])
        token = vocab.get(raw_id, -1)
        if token >= 0 and physical != physical_tokens[token]:
            raise ValueError(f"Road class {fclass} disagrees with vocab physical flag for {raw_id}")
        feature_counts[fclass] += 1
        if fclass in EXCLUDED:
            continue
        parts = geometry["coordinates"]
        if geometry["type"] == "LineString":
            parts = [parts]
        elif geometry["type"] != "MultiLineString":
            raise ValueError(f"Unexpected road geometry {geometry['type']}")
        speed = max(CLASS_KMH.get(fclass, OTHER_KMH),
                    MAXSPEED_FACTOR * (props.get("maxspeed") or 0))
        for part in parts:
            if len(part) < 2:
                continue
            coordinates.extend(value for vertex in part for value in vertex[:2])
            sizes.append(len(part))
            speeds.append(speed)
            tokens.append(token)
            physical_parts.append(physical)
    return (np.frombuffer(coordinates, dtype=np.float64).reshape(-1, 2),
            np.frombuffer(sizes, dtype=np.int32), np.frombuffer(speeds, dtype=np.float32),
            np.frombuffer(tokens, dtype=np.int32), np.frombuffer(physical_parts, dtype=bool),
            physical_tokens, feature_counts)


def group_offsets(groups, universe_size):
    return np.r_[0, np.bincount(groups, minlength=universe_size).cumsum()].astype(np.int32)


def make_graph(lonlat, sizes, speeds, tokens, physical_parts, vocab_size):
    starts = np.r_[0, sizes.cumsum()[:-1]]
    ends = starts + sizes - 1
    keys, inverse, counts = np.unique(coordinate_keys(lonlat), return_inverse=True, return_counts=True)

    # Retain part endpoints, shared vertices and virtual attachment points.
    # Compress shape-only runs into one segment with the sum of polyline lengths.
    retained = counts > 1
    retained[inverse[starts]] = True
    retained[inverse[ends]] = True
    node_id = np.cumsum(retained, dtype=np.int32) - 1
    taken = np.flatnonzero(retained[inverse])
    part = np.searchsorted(ends, taken, side="left")
    adjacent = np.flatnonzero(part[1:] == part[:-1])
    part_of_segment = part[adjacent]
    endpoint_ids = node_id[inverse[taken]]
    u, v = endpoint_ids[adjacent], endpoint_ids[adjacent + 1]

    # Equirectangular segment lengths use the midpoint latitude. Omit the
    # line-to-line jumps before accumulation so only within-part lengths count.
    delta = np.diff(lonlat, axis=0)
    delta[:, 0] *= np.cos(np.deg2rad((lonlat[:-1, 1] + lonlat[1:, 1]) / 2))
    length = np.hypot(delta[:, 0], delta[:, 1]) * (EARTH_M * np.pi / 180.0)
    length[ends[:-1]] = 0.0
    prefix = np.r_[0.0, length.cumsum()]
    weight = (prefix[taken[adjacent + 1]] - prefix[taken[adjacent]])
    weight = np.maximum(weight * 0.06 / speeds[part_of_segment], 1e-8)

    retained_keys = keys[retained]
    nodes = np.column_stack(np.divmod(retained_keys, COORD_STRIDE)) / COORD_PRECISION + ORIGIN
    physical_segments = physical_parts[part_of_segment]
    real_nodes = np.zeros(len(nodes), dtype=bool)
    real_nodes[u[physical_segments]] = True
    real_nodes[v[physical_segments]] = True
    raw_u, raw_v = np.minimum(u, v), np.maximum(u, v)
    # Sort physical segment instances by endpoint pair. CSR uses the minimum
    # across parallel instances, while the coverage audit retains each weight.
    order = np.lexsort((weight, raw_v, raw_u))
    u, v, weight = raw_u[order], raw_v[order], weight[order]
    first_pair = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
    put = first_pair & (u != v)
    a, b, w = u[put], v[put], weight[put]
    graph = csr_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])),
                       shape=(len(nodes), len(nodes)))

    seg_token = tokens[part_of_segment[order]]
    sidx = np.flatnonzero(seg_token >= 0)
    sidx = sidx[np.argsort(seg_token[sidx], kind="stable")]

    mapped_parts = np.flatnonzero(tokens >= 0)
    mapped_parts = mapped_parts[np.argsort(tokens[mapped_parts], kind="stable")]
    pairs = np.column_stack((node_id[inverse[starts[mapped_parts]]],
                             node_id[inverse[ends[mapped_parts]]]))
    return graph, real_nodes, {
        "node_lonlat": nodes,
        "segment_u": u, "segment_v": v, "segment_min": weight,
        "segment_row_start": group_offsets(u, len(nodes)),
        "vocab_segment_ids": sidx.astype(np.int32),
        "vocab_segment_offsets": group_offsets(seg_token[sidx], vocab_size),
        "vocab_node_pairs": pairs, "vocab_pair_offsets": group_offsets(tokens[mapped_parts], vocab_size),
    }


class SpatialGrid:
    """Sorted grid bins used to retrieve polygon bounding-box candidates."""

    def __init__(self, points):
        self.points = points
        self.origin = points.min(axis=0)
        self.step = 1000.0
        self.cells = np.floor((points - self.origin) / self.step).astype(np.int32)
        self.stride = int(self.cells[:, 1].max()) + 1
        keys = self.cells[:, 0] * self.stride + self.cells[:, 1]
        self.order = np.argsort(keys, kind="stable")
        self.keys = keys[self.order]
        self.max_cell = self.cells.max(axis=0)

    def query(self, low, high):
        start = np.maximum(np.floor((low - self.origin) / self.step).astype(int), 0)
        end = np.minimum(np.floor((high - self.origin) / self.step).astype(int), self.max_cell)
        if np.any(start > end):
            return np.empty(0, dtype=np.int32)
        bases = np.arange(start[0], end[0] + 1) * self.stride
        starts = np.searchsorted(self.keys, bases + start[1], side="left")
        ends = np.searchsorted(self.keys, bases + end[1], side="right")
        return np.concatenate([self.order[a:b] for a, b in zip(starts, ends)])


def in_ring(points, ring):
    """Ray casting in vectorized blocks, including points on ring boundaries."""
    lo, hi = ring.min(axis=0), ring.max(axis=0)
    eligible = np.flatnonzero(np.all((points >= lo) & (points <= hi), axis=1))
    result = np.zeros(len(points), dtype=bool)
    if not eligible.size:
        return result
    start, end = ring[:-1], ring[1:]
    dx, dy = (end - start).T
    usable = (dx != 0) | (dy != 0)
    x1, y1 = start[usable].T
    x2, y2 = end[usable].T
    dx, dy = dx[usable], dy[usable]
    above = dy > 0
    batch_size = max(1, 160_000 // max(1, len(dx)))
    for begin in range(0, len(eligible), batch_size):
        idx = eligible[begin:begin + batch_size]
        x, y = points[idx].T[:, :, None]
        cross = dx * (y - y1) - dy * (x - x1)
        ray = ((y1 > y) != (y2 > y)) & ((cross > 0) == above)
        on_side = ((np.abs(cross) <= 1e-7) &
                   (x >= np.minimum(x1, x2)) & (x <= np.maximum(x1, x2)) &
                   (y >= np.minimum(y1, y2)) & (y <= np.maximum(y1, y2)))
        result[idx] = (np.count_nonzero(ray, axis=1) % 2 != 0) | on_side.any(axis=1)
    return result


def assign_tracts(nodes):
    xy = (nodes - ORIGIN) * SCALE_M
    grid = SpatialGrid(xy)
    tracts = json.loads(TRACTS.read_bytes())["features"]
    tracts.sort(key=lambda feature: feature["properties"]["GEOID"])
    ids = np.array([feature["properties"]["GEOID"] for feature in tracts])
    assignment = np.full(len(nodes), -1, dtype=np.int32)
    for index, feature in enumerate(tracts):
        shape = feature["geometry"]
        polygons = shape["coordinates"]
        if shape["type"] == "Polygon":
            polygons = [polygons]
        elif shape["type"] != "MultiPolygon":
            raise ValueError(f"Unexpected tract geometry {shape['type']}")
        for polygon in polygons:
            if not polygon or not polygon[0]:
                continue
            exterior, *holes = polygon
            ring = (np.asarray(exterior) - ORIGIN) * SCALE_M
            idx = grid.query(ring.min(axis=0), ring.max(axis=0))
            idx = idx[assignment[idx] < 0]
            mask = in_ring(xy[idx], ring)
            for hole in holes:
                mask &= ~in_ring(xy[idx], (np.asarray(hole) - ORIGIN) * SCALE_M)
            assignment[idx[mask]] = index
    return ids, assignment, xy


def tract_attachments(nodes, tract_ids, node_tract, arrays, vocab):
    """Use matched driving-link road ends as tract anchor entrances."""
    tract_index = {key: idx for idx, key in enumerate(tract_ids)}
    matches, stats = [], Counter()
    keys = coordinate_keys(nodes)
    for link in json_lines(TRACT_LINKS):
        properties = link["properties"]
        if properties["virtual_link_fclass"] != "driving":
            continue
        stats["driving_links"] += 1
        tract = tract_index.get(properties["tract_id"])
        if tract is None:
            stats["unknown_tract"] += 1
            continue
        # Link geometry is ordered tract centroid -> vertex of properties.road_id.
        key = coordinate_keys(np.asarray(link["geometry"]["coordinates"][-1:]))[0]
        node = int(np.searchsorted(keys, key))
        if node == len(keys) or keys[node] != key:
            stats["missing_link_node"] += 1
            continue
        raw_id = properties["road_id"]
        token = vocab.get(raw_id)
        if token is not None:
            lo, hi = arrays["vocab_segment_offsets"][token:token + 2]
            segments = arrays["vocab_segment_ids"][lo:hi]
            if (not np.any(arrays["segment_u"][segments] == node)
                    and not np.any(arrays["segment_v"][segments] == node)):
                stats["road_id_attachment_mismatches"] += 1
                continue
        if node_tract[node] < 0:
            node_tract[node] = tract
            stats["tract_assignment_fallback_nodes"] += 1
        matches.append((tract, node))
    pairs = np.unique(np.asarray(matches, dtype=np.int32).reshape(-1, 2), axis=0)
    stats["tracts_with_driving_attachments"] = len(np.unique(pairs[:, 0]))
    return pairs, dict(stats)


def read_dbf(path):
    """Return record storage and the fclass slice, honoring deleted rows."""
    data = path.read_bytes()
    count, header_len, row_len = struct.unpack_from("<IHH", data, 4)
    offset = 1
    fields = {}
    for position in range(32, header_len - 1, 32):
        field = data[position:position + 32]
        name = field[:11].split(b"\0", 1)[0].decode("ascii")
        fields[name] = slice(offset, offset + field[16])
        offset += field[16]
    if data[header_len - 1] != 13 or offset != row_len:
        raise ValueError(f"Malformed DBF header: {path}")
    if len(data) < header_len + count * row_len:
        raise ValueError(f"Truncated DBF rows: {path}")
    return data, count, header_len, row_len, fields["fclass"]


def read_pois(settings):
    groups = settings["POI_GROUPS"]
    group_index = {label: i for i, label in enumerate(groups)}
    fclass_groups = {name.encode(): group_index[group] for name, group in settings["FCLASS_TO_GROUP"].items()}
    other = group_index["other_poi"]
    x0, y0, x1, y1 = settings["BBOX"]
    points, poi_groups, layer_stats = [], [], {}
    for layer in POI_LAYERS:
        data, count, header, size, field = read_dbf(POI_DIR / (layer + ".dbf"))
        shapes = (POI_DIR / (layer + ".shp")).read_bytes()
        version, shape_type = struct.unpack_from("<ii", shapes, 28)
        if struct.unpack_from(">i", shapes)[0] != 9994 or version != 1000 or shape_type not in (1, 5):
            raise ValueError(f"Unexpected shapefile header: {layer}")
        start_count, position = len(points), 100
        for i in range(count):
            number, words = struct.unpack_from(">ii", shapes, position)
            position += 8
            end = position + 2 * words
            kind = struct.unpack_from("<i", shapes, position)[0]
            row = header + i * size
            if number != i + 1 or end > len(shapes) or kind not in (0, shape_type):
                raise ValueError(f"SHP/DBF record alignment failure: {layer}, row {i}")
            if kind and data[row] != 42:
                if kind == 1:
                    lon, lat = struct.unpack_from("<2d", shapes, position + 4)
                else:
                    a, b, c, d = struct.unpack_from("<4d", shapes, position + 4)
                    lon, lat = (a + c) / 2, (b + d) / 2
                if x0 <= lon <= x1 and y0 <= lat <= y1:
                    fclass = data[row + field.start:row + field.stop].strip().lower()
                    points.append((lon, lat))
                    poi_groups.append(fclass_groups.get(fclass, other))
            position = end
        if position != len(shapes):
            raise ValueError(f"SHP/DBF record count mismatch: {layer}")
        layer_stats[layer] = {"shape_type": shape_type, "records": count, "in_bbox": len(points) - start_count}
    return np.asarray(points).reshape(-1, 2), np.asarray(poi_groups, dtype=np.int32), layer_stats


def unit_sphere(points):
    lon, lat = np.deg2rad(points).T
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def snap_pois(nodes, road_ids, poi_settings):
    pois, groups, layers = read_pois(poi_settings)
    radius = 2 * np.sin(SNAP_M / (2 * EARTH_M))
    tree = cKDTree(unit_sphere(nodes[road_ids]))
    distance, nearest = tree.query(unit_sphere(pois), distance_upper_bound=radius, workers=4)
    snapped = np.isfinite(distance)
    return road_ids[nearest[snapped]], groups[snapped], layers, len(pois)


def micro_zones(xy, node_tract, road_ids, tract_ids, attachment_pairs, snapped_pois, poi_groups):
    poi_nodes, poi_group_ids, layers, n_pois = snapped_pois
    available = road_ids[node_tract[road_ids] >= 0]
    cells = np.floor(xy[available] / ZONE_M).astype(np.int32)
    zone_keys, zone_id, sizes = np.unique(
        np.column_stack((node_tract[available], cells)), axis=0, return_inverse=True,
        return_counts=True,
    )
    center_x = np.bincount(zone_id, weights=xy[available, 0]) / sizes
    center_y = np.bincount(zone_id, weights=xy[available, 1]) / sizes
    center_dist = np.hypot(xy[available, 0] - center_x[zone_id], xy[available, 1] - center_y[zone_id])
    ranked = np.lexsort((available, center_dist, zone_id))
    first = np.r_[0, sizes.cumsum()[:-1]]
    representatives = available[ranked[first]]

    node_zone = np.full(len(xy), -1, dtype=np.int32)
    node_zone[available] = zone_id
    poi_zones = node_zone[poi_nodes]
    valid = poi_zones >= 0
    n_groups = len(poi_groups)
    poi_counts = np.bincount(
        poi_zones[valid] * n_groups + poi_group_ids[valid], minlength=len(zone_keys) * n_groups,
    ).reshape(-1, n_groups).astype(np.int32)

    # Use observed tract entrances; a tract without an explicit driving link
    # exposes its in-tract zone representatives as boundary sources.
    anchor_pairs = list(map(tuple, attachment_pairs))
    missing = np.ones(len(tract_ids), dtype=bool)
    missing[attachment_pairs[:, 0]] = False
    use = missing[zone_keys[:, 0]]
    anchor_pairs.extend(zip(zone_keys[use, 0], representatives[use]))
    pairs = np.unique(np.asarray(anchor_pairs, dtype=np.int32).reshape(-1, 2), axis=0)
    placed = int(valid.sum())
    summary = {
        "tracts": len(tract_ids), "tracts_with_micro_zones": int(len(np.unique(zone_keys[:, 0]))),
        "micro_zones": len(zone_keys), "grid_cell_m": ZONE_M, "micro_zones_per_tract": len(zone_keys) / len(tract_ids),
        "poi_groups": poi_groups, "poi_layers": layers, "poi_snap_radius_m": SNAP_M,
        "poi_in_bbox": n_pois, "poi_within_snap_radius": len(poi_nodes),
        "poi_assigned_to_micro_zone": placed,
        "poi_per_micro_zone": placed / len(zone_keys), "poi_per_tract": placed / len(tract_ids),
        "tracts_with_anchor_nodes": int(len(np.unique(pairs[:, 0]))),
        "physical_nodes_outside_study_tracts": int(len(road_ids) - len(available)),
    }
    return {
        "zone_tract": zone_keys[:, 0], "zone_representative": representatives.astype(np.int32),
        "zone_poi_counts": poi_counts, "zone_cells": zone_keys[:, 1:], "zone_node_count": sizes,
        "node_zone": node_zone, "anchor_nodes": pairs[:, 1],
        "anchor_offsets": group_offsets(pairs[:, 0], len(tract_ids)),
    }, summary


def measured_coverage(covered, total):
    return {"covered": int(covered), "total": int(total), "fraction": float(covered / total) if total else None}


def test_cohort():
    auto_modes = literal_settings(SCRIPTS / "02_build_mvp_dataset.py", {"AUTO_MODES"})["AUTO_MODES"]
    person_day_ids, observed = [], []
    for record in json_lines(MASKED_TEST):
        if pure_auto(record, auto_modes):
            person_day_ids.append(record["person_day_id"])
            observed.extend(record["observed_edges"])
    return person_day_ids, observed


def file_signature(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def observation_coverage(arrays, main_component, observed):
    offsets, pairs = arrays["vocab_pair_offsets"], arrays["vocab_node_pairs"]
    present = np.diff(offsets) > 0
    on_main = np.zeros(len(present), dtype=bool)
    present_ids = np.flatnonzero(present)
    main_part = np.all(arrays["component"][pairs] == main_component, axis=1)
    # Multiple parts of a token are covered if at least one part is in the
    # largest component. Presence and connectivity are reported independently.
    on_main[present_ids] = np.logical_or.reduceat(main_part, offsets[present_ids])
    unique = np.unique(observed)
    return {
        "occurrences_present": measured_coverage(present[observed].sum(), len(observed)),
        "unique_present": measured_coverage(present[unique].sum(), len(unique)),
        "occurrences_in_largest_component": measured_coverage(on_main[observed].sum(), len(observed)),
        "unique_in_largest_component": measured_coverage(on_main[unique].sum(), len(unique)),
    }


def load_cache():
    # Discard the archive handle after loading to keep replacement safe.
    with np.load(CACHE, allow_pickle=False) as cache:
        summary = json.loads(cache["metadata"].item())
        data = {key: cache[key] for key in cache.files if key != "metadata"}
    return data, summary


def rebuild(signature):
    started = stage = perf_counter()
    timings = {}

    def finish_stage(name):
        nonlocal stage
        now = perf_counter()
        timings[name] = now - stage
        stage = now

    vocab = json.loads(EDGE_VOCAB.read_bytes())
    if not np.array_equal(np.sort(list(vocab.values())), np.arange(len(vocab))):
        raise ValueError("Expected dense edge_vocab IDs")
    lonlat, sizes, speeds, tokens, physical_parts, physical_vocab, counts = read_road_parts(vocab)
    finish_stage("roads")
    graph, real_nodes, data = make_graph(lonlat, sizes, speeds, tokens, physical_parts, len(vocab))
    del lonlat, sizes, speeds, tokens, physical_parts
    nodes = data["node_lonlat"]
    print(f"[Roads] {sum(counts.values()):,} features, {len(nodes):,} nodes, {graph.nnz // 2:,} CSR edges", flush=True)
    n_components, component = connected_components(graph, directed=False)
    component_sizes = np.bincount(component)
    main = int(component_sizes.argmax())
    data.update(component=component, vocab_is_physical=physical_vocab,
                graph_data=graph.data, graph_indices=graph.indices, graph_indptr=graph.indptr)
    finish_stage("graph")
    tract_ids, node_tract, xy = assign_tracts(nodes)
    attachment_pairs, attachment_summary = tract_attachments(nodes, tract_ids, node_tract, data, vocab)
    finish_stage("tracts")
    # Virtual-link centroids serve routing. Only actual road nodes seed POI zones.
    poi_settings = literal_settings(SCRIPTS / "01_fetch_chicago_poi.py", {"FCLASS_TO_GROUP", "POI_GROUPS", "BBOX"})
    road_ids = np.flatnonzero(real_nodes)
    snapped_pois = snap_pois(nodes, road_ids, poi_settings)
    finish_stage("pois")
    zones, zone_summary = micro_zones(xy, node_tract, road_ids, tract_ids, attachment_pairs,
                                      snapped_pois, poi_settings["POI_GROUPS"])
    data.update(zones, tract_ids=tract_ids, node_tract=node_tract)
    finish_stage("zones")
    summary = {
        "cache_version": CACHE_VERSION, "sources": signature,
        "feature_classes": dict(counts), "network_features": sum(counts.values()),
        "speed_floor_kmh": CLASS_KMH, "other_speed_floor_kmh": OTHER_KMH,
        "maxspeed_multiplier": MAXSPEED_FACTOR, "excluded_fclasses": sorted(EXCLUDED),
        "virtual_fclasses": sorted(VIRTUAL_FCLASSES),
        "nodes": len(nodes), "csr_undirected_edges": graph.nnz // 2,
        "segment_instances": len(data["segment_u"]), "edge_vocab_size": len(vocab),
        "component_count": n_components, "largest_component_id": main,
        "largest_component_nodes": int(component_sizes[main]),
        "tract_attachment": attachment_summary, "micro_zones": zone_summary,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=CACHE.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, **data)
            stream.close()
            finish_stage("save")
            timings["total_s"] = stage - started
            summary["build_timing_seconds"] = timings
            # Record the bulk save time inside the archive it describes.
            # Only this small metadata entry and the atomic rename follow.
            with ZipFile(temporary, "a") as archive, archive.open("metadata.npy", "w") as metadata:
                np.lib.format.write_array(metadata, np.asarray(json.dumps(summary)), allow_pickle=False)
            temporary.replace(CACHE)
        finally:
            temporary.unlink(missing_ok=True)
    return data, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild even when the cache matches current inputs.")
    args = parser.parse_args()
    clock = perf_counter()
    signature = source_signature()
    data, summary = None, None
    if not args.force and CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as cache:
            existing = json.loads(cache["metadata"].item())
        if existing["cache_version"] == CACHE_VERSION and existing["sources"] == signature:
            data, summary = load_cache()
            print(f"[Cache] Loaded {CACHE.relative_to(MVP.parent)}")
    rebuilt = data is None
    if rebuilt:
        data, summary = rebuild(signature)
    timings = " ".join(f"{key}={value:.3f}" for key, value in summary["build_timing_seconds"].items())
    print(f"[Build timing, {'current' if rebuilt else 'last full build'}, seconds] {timings}")
    cohort_ids, observed = test_cohort()
    coverage = observation_coverage(data, summary["largest_component_id"], observed)
    cohort = {"masked_test_signature": file_signature(MASKED_TEST), "person_day_ids": cohort_ids,
              "observation_coverage": coverage}
    with tempfile.NamedTemporaryFile(dir=CACHE.parent, mode="w", encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(cohort, stream, indent=2)
            stream.close()
            temporary.replace(COHORT)
        finally:
            temporary.unlink(missing_ok=True)
    for label, result in coverage.items():
        print(f"[AVI] {label}: {result['covered']:,}/{result['total']:,} ({result['fraction']:.3%})")
    zones = summary["micro_zones"]
    print(f"[Zones] {zones['micro_zones']:,} micro-zones / {zones['tracts']:,} tracts "
          f"({zones['micro_zones_per_tract']:.2f} per tract)")
    print(f"[POIs] {zones['poi_assigned_to_micro_zone']:,}/{zones['poi_in_bbox']:,} assigned "
          f"({zones['poi_per_micro_zone']:.2f} per zone, {zones['poi_per_tract']:.2f} per tract)")
    print(f"[09] {perf_counter() - clock:.3f} s")
    if coverage["occurrences_in_largest_component"]["fraction"] < 0.995:
        raise SystemExit("AVI connectivity to the largest component is below 99.5%.")


if __name__ == "__main__":
    main()
