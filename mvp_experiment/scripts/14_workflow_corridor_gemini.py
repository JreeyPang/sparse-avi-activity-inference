"""Step 14: Parallel Gemini Flash inference over strictly leak-free corridor batches (val60 + test100)
via Antigravity run_workflow (model='flash').
"""
import json
from pathlib import Path

ROOT = Path.cwd()
BATCH_DIR = ROOT / "mvp_experiment" / "data" / "corridor_llm_batches"
RES_DIR = ROOT / "mvp_experiment" / "results"
SCHEMA = json.loads((BATCH_DIR / "output_schema.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((BATCH_DIR / "manifest.json").read_text(encoding="utf-8"))

INSTRUCTIONS = """You are a transportation behavior & urban geography expert reconstructing Chicago private-vehicle daily trip chains strictly from sparse AVI camera detections and corridor micro-zone POIs.
DO NOT call any file, shell, or search tools — everything you need is in the JSON below. Directly reason and call finish with your structured JSON predictions.

For EVERY likely gap (every gap_key in every chain below), examine:
1. Full daily context (`day_of_week`, `is_weekend`, `daily_gap_skeleton`, home anchor consistency across gaps, work/school regularity, escort-before/after-work pairs).
2. Each candidate corridor micro-zone (`zone_id`, `detour_min`, `max_dwell_min`, `is_home_anchor`, `micro_named_pois`, `micro_poi_groups`, `tract_named_pois`).

Output for EVERY gap_key (do not skip any gap_key!):
- `brief_reasoning`: 1–2 concise sentences in Chinese (简洁中文) explaining WHY you think the car stopped (or did not stop), which micro-zone/POI fits best (trading off detour_min vs POI & schedule match), and what activity it most likely is.
- `stop_prob_llm`: probability in [0, 1] that a real activity stop occurred in this gap (vs pure driving/congestion).
- `gap_activity_probs`: calibrated 9-class probability distribution summing to ~1.0 over:
  - `home`: returning home mid-day (ONLY plausible if a candidate has `is_home_anchor=true`, or overnight start/end)
  - `work`: workplace / job / commercial-industrial visit (weekday long stay 2–10h, or repeated work corridor)
  - `school`: attending K-12 school / college / university / daycare (weekday multi-hour stay at a micro-zone with school/university/library POIs)
  - `shop`: supermarket, mall, retail, pharmacy retail (typically 10–75 min, shop POIs)
  - `meal`: restaurant, fast food, cafe, drive-thru food (5–90 min around breakfast/lunch 11:00–14:30 or dinner 17:00–21:00, meal POIs — DO NOT under-predict meal when dining POIs & mealtime match!)
  - `social_recreational`: park, gym, sports, cinema, museum, church/worship, visiting friends/family (often evenings/weekends, 30–240 min)
  - `escort`: pick-up / drop-off another person (VERY short stop 1–15 min, especially near schools 07:00–09:00 or 14:30–16:30, transit stations, or residential zones)
  - `healthcare`: hospital, clinic, doctor, dentist, medical campus visit (typically 25–180 min weekday daytime near hospital/clinic/doctors/dentist POIs)
  - `errand_other`: bank, post office, gas/auto service, personal errands, other
- `top_zones`: pick the top 3 most plausible candidate `zone_id`s FROM THIS GAP'S `candidates` list (use exact integer `zone_id`s from input!), assigning each a `zone_prob` (summing to <= 1.0, reflecting BOTH low `detour_min` AND strong POI/home-anchor/schedule fit!) and a zone-specific 9-class `activity_probs` distribution.
"""

all_batch_files = []
for sp in ("val60", "test100"):
    for b in MANIFEST["splits"][sp]["batches"]:
        all_batch_files.append((sp, BATCH_DIR / b["file"]))

phase("Corridor Gemini Flash inference (val60 + test100)")
log(f"Launching {len(all_batch_files)} parallel Gemini Flash workers...")

def make_worker(item):
    sp, bf = item
    data_str = bf.read_text(encoding="utf-8")
    prompt = f"{INSTRUCTIONS}\n\nINPUT BATCH ({bf.name}):\n```json\n{data_str}\n```"
    return agent(
        prompt=prompt,
        schema=SCHEMA,
        role=f"Gemini {bf.stem}",
        type_name="research",
        model="flash",
    )

results = await parallel(all_batch_files, make_worker)

merged = {"val60": {}, "test100": {}}
for (sp, bf), res in zip(all_batch_files, results):
    preds = res.get("gap_predictions", [])
    log(f"{bf.name}: returned {len(preds)} gap predictions")
    for p in preds:
        merged[sp][p["gap_key"]] = p

RES_DIR.mkdir(parents=True, exist_ok=True)
for sp in ("val60", "test100"):
    out_path = RES_DIR / f"corridor_gemini_{sp}.json"
    out_path.write_text(
        json.dumps({"model": "gemini-flash-corridor", "split": sp, "gap_predictions": merged[sp]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Saved {len(merged[sp])} {sp} gap predictions -> {out_path}")
