"""Run Codex CLI (`codex exec`) in parallel across all 10 batches (100 pure-auto test chains),
saving merged predictions to results/codex_100chains.json with per-batch resume caching.
"""
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "mvp_experiment" / "data" / "batches_100"
CACHE_DIR = ROOT / "mvp_experiment" / "results" / "codex_batch_cache"
OUT_PATH = ROOT / "mvp_experiment" / "results" / "codex_100chains.json"
SCHEMA_PATH = BATCH_DIR / "output_schema.json"

INSTRUCTIONS = """You are a transportation planning & travel behavior expert analyzing Chicago private-vehicle daily trip chains.
DO NOT run any shell commands or read any local files — everything you need is in the JSON below. Directly reason and output the final JSON matching the required schema.

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


def run_one(bf: Path):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_file = CACHE_DIR / f"{bf.stem}.json"
    if out_file.exists():
        data = json.loads(out_file.read_text(encoding="utf-8"))
        if data.get("predictions"):
            print(f"[Codex] {bf.name}: cached ({len(data['predictions'])} preds)", flush=True)
            return bf.name, data

    t0 = time.time()
    prompt = f"{INSTRUCTIONS}\n\nINPUT CHAINS ({bf.name}):\n```json\n{bf.read_text(encoding='utf-8')}\n```"
    cmd = [
        "codex", "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "-s", "read-only",
        "--output-schema", str(SCHEMA_PATH),
        "-o", str(out_file),
        "-",
    ]
    proc = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=240)
    if proc.returncode != 0 or not out_file.exists():
        raise RuntimeError(f"Codex failed on {bf.name}: rc={proc.returncode}\nSTDERR:\n{proc.stderr[-1500:]}\nSTDOUT:\n{proc.stdout[-800:]}")
    data = json.loads(out_file.read_text(encoding="utf-8"))
    print(f"[Codex] {bf.name}: done in {time.time()-t0:.1f}s ({len(data.get('predictions', []))} preds)", flush=True)
    return bf.name, data


def main():
    batch_files = sorted(BATCH_DIR.glob("batch_*.json"))
    all_preds = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(run_one, bf) for bf in batch_files]
        for fut in concurrent.futures.as_completed(futs):
            bname, data = fut.result()
            for p in data.get("predictions", []):
                all_preds[p["sample_key"]] = p

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"model": "codex-cli-default", "predictions": all_preds}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Codex] Saved {len(all_preds)} merged predictions -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
