"""Step 1: Fetch Illinois OSM POI layers from Geofabrik via HTTP Range requests
and spatially join them to Chicago Census Tracts (TractZonalHistogram.geojson).

Outputs:
  mvp_experiment/data/osm_il/            - Extracted OSM POI shapefiles (~19 MB)
  mvp_experiment/data/tract_poi_summary.json - Per-tract POI counts, shares, entropy, top named POIs (for LLM)
  mvp_experiment/data/tract_poi_features.csv - Per-tract tabular POI + land-cover + ACS features (for ML)
"""
import collections
import csv
import io
import json
import math
import zipfile
from pathlib import Path

import requests
import shapefile  # pyshp
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
CHICAGO_DIR = ROOT / "Chicago-Trip"
MVP_DATA = ROOT / "mvp_experiment" / "data"
OSM_DIR = MVP_DATA / "osm_il"

GEOFABRIK_URL = "https://download.geofabrik.de/north-america/us/illinois-latest-free.shp.zip"
TARGET_PREFIXES = (
    "gis_osm_pois_free_1.",
    "gis_osm_pois_a_free_1.",
    "gis_osm_pofw_free_1.",
    "gis_osm_pofw_a_free_1.",
    "gis_osm_transport_free_1.",
    "gis_osm_transport_a_free_1.",
)

# Chicago study area bbox (from TractZonalHistogram.geojson with tiny margin)
BBOX = (-88.75, 41.18, -87.00, 42.52)

# Map OSM Geofabrik fclass -> activity-oriented POI semantic category
FCLASS_TO_GROUP = {
    # shop
    "supermarket": "shop", "convenience": "shop", "department_store": "shop", "mall": "shop",
    "clothes": "shop", "shoe_shop": "shop", "bakery": "shop", "butcher": "shop",
    "beverages": "shop", "greengrocer": "shop", "furniture_shop": "shop", "doityourself": "shop",
    "beauty_shop": "shop", "hairdresser": "shop", "chemist": "shop", "bookshop": "shop",
    "car_dealership": "shop", "bicycle_shop": "shop", "gift_shop": "shop", "jeweller": "shop",
    "mobile_phone_shop": "shop", "computer_shop": "shop", "florist": "shop", "garden_centre": "shop",
    "optician": "shop", "outdoor_shop": "shop", "sports_shop": "shop", "stationery": "shop",
    "toy_shop": "shop", "video_shop": "shop", "kiosk": "shop", "market_place": "shop",
    "vending_any": "shop", "general": "shop", "newsagent": "shop", "laundry": "shop",
    # meal
    "restaurant": "meal", "fast_food": "meal", "cafe": "meal", "bar": "meal", "pub": "meal",
    "biergarten": "meal", "food_court": "meal", "nightclub": "meal",
    # healthcare
    "hospital": "healthcare", "clinic": "healthcare", "doctors": "healthcare",
    "dentist": "healthcare", "pharmacy": "healthcare", "veterinary": "healthcare",
    "nursing_home": "healthcare", "optician": "healthcare",
    # school / education
    "school": "school", "university": "school", "college": "school", "kindergarten": "school",
    "library": "school",
    # social / recreational
    "park": "social_rec", "playground": "social_rec", "pitch": "social_rec",
    "sports_centre": "social_rec", "stadium": "social_rec", "swimming_pool": "social_rec",
    "golf_course": "social_rec", "ice_rink": "social_rec", "track": "social_rec",
    "dog_park": "social_rec", "cinema": "social_rec", "theatre": "social_rec",
    "museum": "social_rec", "arts_centre": "social_rec", "attraction": "social_rec",
    "theme_park": "social_rec", "zoo": "social_rec", "viewpoint": "social_rec",
    "picnic_site": "social_rec", "camp_site": "social_rec", "community_centre": "social_rec",
    "artwork": "social_rec", "monument": "social_rec", "memorial": "social_rec",
    "ruins": "social_rec", "castle": "social_rec", "fort": "social_rec", "fountain": "social_rec",
    "beach": "social_rec", "water_park": "social_rec", "tourist_info": "social_rec",
    # worship / civic (also social/escort/community)
    "christian": "worship_civic", "christian_catholic": "worship_civic",
    "christian_protestant": "worship_civic", "christian_lutheran": "worship_civic",
    "christian_baptist": "worship_civic", "christian_methodist": "worship_civic",
    "christian_episcopal": "worship_civic", "christian_orthodox": "worship_civic",
    "christian_presbyterian": "worship_civic", "jewish": "worship_civic",
    "muslim": "worship_civic", "muslim_sunni": "worship_civic", "buddhist": "worship_civic",
    "hindu": "worship_civic", "sikh": "worship_civic", "town_hall": "worship_civic",
    "courthouse": "worship_civic", "embassy": "worship_civic", "public_building": "worship_civic",
    "police": "worship_civic", "fire_station": "worship_civic", "prison": "worship_civic",
    "graveyard": "worship_civic",
    # errand / services
    "bank": "errand", "atm": "errand", "post_office": "errand", "post_box": "errand",
    "car_wash": "errand", "car_rental": "errand", "car_sharing": "errand",
    "travel_agent": "errand", "recycling": "errand", "recycling_glass": "errand",
    "recycling_paper": "errand", "recycling_clothes": "errand", "recycling_metal": "errand",
    "toilet": "errand", "telephone": "errand", "drinking_water": "errand",
    "bench": "errand", "waste_basket": "errand",
    # lodging / overnight
    "hotel": "lodging", "motel": "lodging", "hostel": "lodging", "guesthouse": "lodging",
    "chalet": "lodging", "alpine_hut": "lodging", "caravan_site": "lodging",
    # transport
    "bus_stop": "transport", "bus_station": "transport", "railway_station": "transport",
    "railway_halt": "transport", "tram_stop": "transport", "taxi": "transport",
    "airport": "transport", "airfield": "transport", "helipad": "transport",
    "ferry_terminal": "transport", "apron": "transport",
}

POI_GROUPS = [
    "shop", "meal", "healthcare", "school", "social_rec",
    "worship_civic", "errand", "lodging", "transport", "other_poi",
]

# Non-informative micro street furniture to exclude from top named POIs & dominant category
NOISE_FCLASSES = {"bench", "waste_basket", "post_box", "telephone", "drinking_water", "toilet", "vending_any", "bus_stop"}


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.session = requests.Session()
        r = self.session.head(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers["Content-Length"])
        self.pos = 0

    def readable(self): return True
    def seekable(self): return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.size + offset
        return self.pos

    def tell(self): return self.pos

    def read(self, n=-1):
        if n == 0 or self.pos >= self.size:
            return b""
        end = self.size - 1 if n < 0 else min(self.pos + n - 1, self.size - 1)
        r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data


def download_osm_poi_layers():
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    needed = [p + ext for p in (
        "gis_osm_pois_free_1", "gis_osm_pois_a_free_1",
        "gis_osm_pofw_free_1", "gis_osm_pofw_a_free_1",
        "gis_osm_transport_free_1", "gis_osm_transport_a_free_1",
    ) for ext in (".shp", ".dbf", ".shx")]
    if all((OSM_DIR / fn).exists() for fn in needed):
        print("[OSM] All target shapefiles already present, skipping download.")
        return

    print(f"[OSM] Opening remote zip via HTTP Range: {GEOFABRIK_URL}")
    rf = HTTPRangeFile(GEOFABRIK_URL)
    with zipfile.ZipFile(rf) as z:
        members = [m for m in z.namelist() if m.startswith(TARGET_PREFIXES)]
        print(f"[OSM] Extracting {len(members)} files to {OSM_DIR} ...")
        for m in members:
            out = OSM_DIR / Path(m).name
            if out.exists():
                continue
            data = z.read(m)
            out.write_bytes(data)
            print(f"  saved {out.name} ({len(data)/1e6:.2f} MB)")


def load_tracts():
    gj_path = CHICAGO_DIR / "Chicago" / "Tracts" / "TractZonalHistogram.geojson"
    with open(gj_path) as f:
        gj = json.load(f)
    geoids, geoms, props = [], [], []
    for ft in gj["features"]:
        g = ft.get("geometry")
        if not g:
            continue
        p = ft["properties"]
        geoids.append(str(p["GEOID"]))
        geoms.append(shape(g))
        props.append(p)
    print(f"[Tracts] Loaded {len(geoids)} tract polygons.")
    return geoids, geoms, props


def iter_osm_pois():
    """Yield (lon, lat, fclass, name) for all POIs in Chicago bbox."""
    layers = [
        "gis_osm_pois_free_1",
        "gis_osm_pois_a_free_1",
        "gis_osm_pofw_free_1",
        "gis_osm_pofw_a_free_1",
        "gis_osm_transport_free_1",
        "gis_osm_transport_a_free_1",
    ]
    minx, miny, maxx, maxy = BBOX
    total_read = 0
    total_in_bbox = 0
    for layer in layers:
        shp_path = OSM_DIR / f"{layer}.shp"
        sf = shapefile.Reader(str(shp_path), encoding="utf-8", encodingErrors="ignore")
        fields = [fld[0] for fld in sf.fields[1:]]
        idx_fclass = fields.index("fclass")
        idx_name = fields.index("name")
        is_poly = layer.endswith("_a_free_1")
        layer_in = 0
        for sr in sf.iterShapeRecords():
            total_read += 1
            shp = sr.shape
            if not shp.points:
                continue
            if is_poly:
                bx0, by0, bx1, by1 = shp.bbox
                lon = 0.5 * (bx0 + bx1)
                lat = 0.5 * (by0 + by1)
            else:
                lon, lat = shp.points[0]
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
            total_in_bbox += 1
            layer_in += 1
            rec = sr.record
            fclass = str(rec[idx_fclass] or "").strip().lower()
            name = str(rec[idx_name] or "").strip()
            yield lon, lat, fclass, name
        print(f"  layer {layer:26s}: {layer_in:7d} POIs in Chicago bbox")
    print(f"[OSM] Total read={total_read}, in Chicago bbox={total_in_bbox}")


def load_acs_summary():
    """Extract a compact, meaningful set of ACS tract features from tract_long_table.csv."""
    acs_path = CHICAGO_DIR / "Chicago" / "Tracts" / "tract_long_table.csv"
    # Selected key ACS columns:
    # P1_001N: Total population; H1_001N: Housing units;
    # DP03_0001E: Pop 16+; DP03_0002E: In labor force; DP03_0004E: Employed;
    # DP03_0019E: Commuters drove alone; DP03_0021E: Public transit commuters;
    # DP03_0024E: Worked from home; DP03_0062E: Median household income;
    # DP04_0001E: Total housing units; DP04_0046E: Owner-occupied; DP04_0047E: Renter-occupied;
    # S0101_C01_001E: Total pop; S1501_C01_006E: Pop 25+; S1501_C02_015E: % Bachelor+
    wanted = [
        "P1_001N", "H1_001N", "DP03_0001E", "DP03_0002E", "DP03_0004E",
        "DP03_0019E", "DP03_0021E", "DP03_0024E", "DP03_0062E",
        "DP04_0001E", "DP04_0046E", "DP04_0047E",
    ]
    out = {}
    with open(acs_path, encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        present = [c for c in wanted if c in r.fieldnames]
        for row in r:
            tid = str(row["tractID"]).strip()
            vals = {}
            for c in present:
                v = row.get(c, "")
                try:
                    fv = float(v)
                    vals[c] = fv if fv >= 0 else 0.0
                except (ValueError, TypeError):
                    vals[c] = 0.0
            out[tid] = vals
    print(f"[ACS] Loaded {len(present)} key ACS features for {len(out)} tracts.")
    return out, present


def main():
    download_osm_poi_layers()
    geoids, geoms, props = load_tracts()
    tree = STRtree(geoms)

    tract_group_counts = {g: collections.Counter() for g in geoids}
    tract_fclass_counts = {g: collections.Counter() for g in geoids}
    # Keep named POIs grouped by semantic category for diverse LLM prompts
    tract_named_pois = {g: collections.defaultdict(list) for g in geoids}

    matched = 0
    unmatched = 0
    for lon, lat, fclass, name in iter_osm_pois():
        pt = Point(lon, lat)
        cand_idxs = tree.query(pt, predicate="intersects")
        if len(cand_idxs) == 0:
            unmatched += 1
            continue
        idx = int(cand_idxs[0])
        gid = geoids[idx]
        matched += 1
        grp = FCLASS_TO_GROUP.get(fclass, "other_poi")
        tract_fclass_counts[gid][fclass] += 1
        if fclass not in NOISE_FCLASSES:
            tract_group_counts[gid][grp] += 1
            if name and len(tract_named_pois[gid][grp]) < 6:
                label = f"{name} ({fclass})"
                if label not in tract_named_pois[gid][grp]:
                    tract_named_pois[gid][grp].append(label)

    print(f"[Spatial Join] Matched {matched} POIs to tracts ({unmatched} outside tract polygons).")

    acs_map, acs_cols = load_acs_summary()

    summary_json = {}
    csv_rows = []
    nonzero_tracts = 0
    for gid, p in zip(geoids, props):
        area_km2 = max(float(p.get("Area") or p.get("ALAND") or 1e6) / 1e6, 0.01)
        gcnt = tract_group_counts[gid]
        total_poi = sum(gcnt.values())
        if total_poi > 0:
            nonzero_tracts += 1
        shares = {g: (gcnt[g] / total_poi if total_poi > 0 else 0.0) for g in POI_GROUPS}
        entropy = -sum(s * math.log(s) for s in shares.values() if s > 0)
        dom_cat = max(POI_GROUPS, key=lambda g: gcnt[g]) if total_poi > 0 else "none"

        # Curate up to 15 diverse named POIs (prioritizing healthcare, school, shop, meal, social_rec, worship_civic, errand)
        priority_order = ["healthcare", "school", "shop", "meal", "social_rec", "worship_civic", "errand", "lodging", "transport", "other_poi"]
        top_named = []
        for grp in priority_order:
            for item in tract_named_pois[gid][grp][:3]:
                top_named.append(item)
                if len(top_named) >= 15:
                    break
            if len(top_named) >= 15:
                break

        # Top fclasses excluding noise
        clean_fclasses = {k: v for k, v in tract_fclass_counts[gid].items() if k not in NOISE_FCLASSES}
        top_fclasses = dict(collections.Counter(clean_fclasses).most_common(10))

        summary_json[gid] = {
            "tract_id": gid,
            "area_km2": round(area_km2, 3),
            "built_up_ratio": round(float(p.get("Built-up") or 0.0), 4),
            "tree_cover_ratio": round(float(p.get("Tree_Cover") or 0.0), 4),
            "total_poi": total_poi,
            "poi_density_per_km2": round(total_poi / area_km2, 2),
            "poi_entropy": round(entropy, 4),
            "dominant_poi_group": dom_cat,
            "poi_group_counts": {g: gcnt[g] for g in POI_GROUPS if gcnt[g] > 0},
            "top_fclasses": top_fclasses,
            "representative_pois": top_named,
        }

        row = {
            "tract_id": gid,
            "area_km2": round(area_km2, 4),
            "built_up": float(p.get("Built-up") or 0.0),
            "tree_cover": float(p.get("Tree_Cover") or 0.0),
            "grassland": float(p.get("Grassland") or 0.0),
            "cropland": float(p.get("Cropland") or 0.0),
            "water": float(p.get("Permanent water bodies") or 0.0),
            "total_poi": total_poi,
            "poi_density": round(total_poi / area_km2, 4),
            "poi_entropy": round(entropy, 4),
            "dominant_poi_group": dom_cat,
            "has_hospital_or_clinic": int(
                (tract_fclass_counts[gid]["hospital"] + tract_fclass_counts[gid]["clinic"] + tract_fclass_counts[gid]["doctors"]) > 0
            ),
            "has_school_or_k12": int(
                (tract_fclass_counts[gid]["school"] + tract_fclass_counts[gid]["kindergarten"]) > 0
            ),
            "has_university": int(
                (tract_fclass_counts[gid]["university"] + tract_fclass_counts[gid]["college"]) > 0
            ),
            "has_mall_or_supermarket": int(
                (tract_fclass_counts[gid]["mall"] + tract_fclass_counts[gid]["supermarket"] + tract_fclass_counts[gid]["department_store"]) > 0
            ),
            "n_restaurants_cafes": tract_fclass_counts[gid]["restaurant"] + tract_fclass_counts[gid]["fast_food"] + tract_fclass_counts[gid]["cafe"],
            "n_healthcare_strict": tract_fclass_counts[gid]["hospital"] + tract_fclass_counts[gid]["clinic"] + tract_fclass_counts[gid]["doctors"] + tract_fclass_counts[gid]["dentist"],
        }
        for g in POI_GROUPS:
            row[f"poi_cnt_{g}"] = gcnt[g]
            row[f"poi_share_{g}"] = round(shares[g], 5)
        acs = acs_map.get(gid, {})
        for c in acs_cols:
            row[f"acs_{c}"] = acs.get(c, 0.0)
        # Derived ACS density / ratios
        pop = row.get("acs_P1_001N", 0.0)
        emp = row.get("acs_DP03_0004E", 0.0)
        row["acs_pop_density"] = round(pop / area_km2, 2)
        row["acs_emp_ratio"] = round(emp / pop, 4) if pop > 0 else 0.0
        csv_rows.append(row)

    MVP_DATA.mkdir(parents=True, exist_ok=True)
    with open(MVP_DATA / "tract_poi_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    with open(MVP_DATA / "tract_poi_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    print(f"[Done] Saved tract_poi_summary.json & tract_poi_features.csv ({nonzero_tracts}/{len(geoids)} tracts have >=1 clean POI).")


if __name__ == "__main__":
    main()
