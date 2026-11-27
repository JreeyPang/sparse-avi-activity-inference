"""Run Gemini Flash in parallel across all 10 batches (100 pure-auto test chains)
via Antigravity run_workflow, saving merged predictions to results/gemini_100chains.json.
"""
import json
from pathlib import Path

ROOT = Path.cwd()
BATCH_DIR = ROOT / "mvp_experiment" / "data" / "batches_100"
OUT_PATH = ROOT / "mvp_experiment" / "results" / "gemini_100chains.json"
SCHEMA = json.loads((BATCH_DIR / "output_schema.json").read_text(encoding="utf-8"))

INSTRUCTIONS = """You are a transportation planning & travel behavior expert analyzing Chicago private-vehicle daily trip chains.
DO NOT call any file, shell, or search tools — everything you need is in the JSON below. Directly reason and call finish with your predictions.

Task: For EVERY stop in EVERY chain below, infer the activity type and a calibrated probability distribution over these exact 9 classes:
- home: traveler's residence (strong cue: first/last stop of day or is_day_start_or_end_tract=true, though people occasionally stay overnight elsewhere)
- work: primary workplace or work-related visits/deliveries (weekday 6:00-18:00 long stays 3-10h, or repeated visits across other_days_at_tract, or commercial/industrial/office areas)
- school: attending K-12 school, college/university, or daycare (weekday mornings/afternoons, multi-hour stay in tract with school/university POIs)
- shop: grocery, supermarket, mall, retail store (typically 10-90 min, daytime/evening, tracts with shop/supermarket/mall POIs)
- meal: dining out, cafe, fast food, drive-thru food (typically 5-90 min around breakfast/lunch 11:00-14:00 or dinner 17:00-20:30, tracts with restaurant/fast_food/cafe POIs)
- social_recreational: gym, park, sports, cinema/museum, church/worship, visiting friends/relatives (often evenings/weekends, 30-240 min, parks/sports/worship/residential tracts)
- escort: picking up or dropping off another person (VERY strong cue: very short stop 1-15 min, especially near weekday school drop-off 7:00-9:00 / pick-up 14:30-16:30, transit stations, or residential/school tracts; often paired before/after work!)
- healthcare: doctor, dentist, clinic, hospital medical visit (typically 20-180 min weekday daytime, tract containing hospital/clinic/doctors/dentist POIs)
- errand_other: bank, post office, gas station, haircut, auto service, personal errands, staying overnight away from home, or other

Use FULL daily chain context (sequence of stops, mutual exclusivity, escort-work-escort pairing, duration, time of day, day of week, trip distances, historical visits, and tract POI composition/names).
Return a prediction entry for EVERY single sample_key in the input (do not skip any stop!). Ensure probs sum to ~1.0 and predicted_label matches the highest probability class.
"""

batch_files = sorted(BATCH_DIR.glob("batch_*.json"))
phase("Gemini Flash 100-chain inference")
log(f"Launching {len(batch_files)} parallel Gemini Flash workers...")

def make_worker(bf: Path):
    data_str = bf.read_text(encoding="utf-8")
    prompt = f"{INSTRUCTIONS}\n\nINPUT CHAINS ({bf.name}):\n```json\n{data_str}\n```"
    return agent(
        prompt=prompt,
        schema=SCHEMA,
        role=f"Gemini {bf.stem}",
        type_name="research",
        model="flash",
    )

batch_results = await parallel(batch_files, make_worker)

all_preds = {}
for bf, res in zip(batch_files, batch_results):
    preds = res.get("predictions", [])
    log(f"{bf.name}: returned {len(preds)} predictions")
    for p in preds:
        all_preds[p["sample_key"]] = p

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUT_PATH.write_text(json.dumps({"model": "gemini-flash", "predictions": all_preds}, ensure_ascii=False, indent=2), encoding="utf-8")
log(f"Saved {len(all_preds)} merged Gemini predictions -> {OUT_PATH}")
