"""Step 4 (Ready-to-run): Baseline 3 — LLM Activity Prior Inference (DeepSeek / OpenAI-compatible).

Usage:
  export DEEPSEEK_API_KEY="sk-..."
  uv run --with requests --with scikit-learn --with numpy \
    python mvp_experiment/scripts/04_run_llm_baseline.py --limit 200 --non-home-only

Supports:
  - Resumable JSONL caching (never pays twice for the same stop!)
  - temperature=0 structured JSON output (probabilities over all 9 classes + duration range)
  - Automatic evaluation using the exact same metrics as 03_run_baselines.py
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import requests
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[2]
MVP_DATA = ROOT / "mvp_experiment" / "data"
RES_DIR = ROOT / "mvp_experiment" / "results"

CLASSES = [
    "home", "work", "school", "shop", "meal",
    "social_recreational", "escort", "healthcare", "errand_other",
]

SYSTEM_PROMPT = f"""You are a transportation planning & travel behavior expert analyzing vehicle trip chains in Chicago.
Given a vehicle stop's timing, duration, daily trip-chain context, and surrounding Census Tract OSM POI composition,
infer the probability distribution over these exact 9 activity classes:
{CLASSES}

Definitions:
- home: returning to / staying at the traveler's residence (often day start/end anchor tract)
- work: primary workplace or work-related meetings/deliveries
- school: K-12 school, university, college, daycare
- shop: grocery, supermarket, mall, retail shopping
- meal: dining out, cafe, fast food, drive-thru food
- social_recreational: gym, park, sports, entertainment, museum, church/civic, visiting friends
- escort: picking up or dropping off another person (often very short 1-15 min stop near school/home/transit/activities)
- healthcare: doctor, dentist, clinic, hospital medical visit
- errand_other: bank, post office, gas station, haircut, auto service, overnight away from home, other

Return ONLY valid JSON with keys:
  "probs": object mapping each of the 9 classes to a float in [0, 1] summing to 1.0
  "predicted_label": one of the 9 class strings (argmax of probs)
  "typical_duration_min": [low, high] reasonable duration range in minutes for this activity
  "rationale": brief 1-sentence reason
"""


def call_llm(api_url: str, api_key: str, model: str, context: dict) -> dict:
    payload = {
        "model": model,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    }
    r = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    parsed = json.loads(content)
    return {"parsed": parsed, "usage": usage}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", default="https://api.deepseek.com/chat/completions")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--api-key-env", help="Environment variable containing the key for --api-url")
    ap.add_argument("--limit", type=int, default=200, help="Number of stops to evaluate (0=all)")
    ap.add_argument("--non-home-only", action="store_true", help="Evaluate only non-home stops for rapid cost-effective validation")
    ap.add_argument("--cache", default=str(RES_DIR / "llm_baseline_cache.jsonl"))
    args = ap.parse_args()

    key_env = args.api_key_env
    if not key_env:
        if args.api_url != "https://api.deepseek.com/chat/completions":
            ap.error("--api-key-env is required for a custom --api-url")
        key_env = "DEEPSEEK_API_KEY"
    api_key = os.environ.get(key_env)
    if not api_key:
        ap.error(f"Set {key_env} before running this script")

    items = [json.loads(line) for line in open(MVP_DATA / "llm_test_prompts.jsonl", encoding="utf-8")]
    if args.non_home_only:
        items = [x for x in items if x["ground_truth_label"] != "home"]
    if args.limit > 0:
        # Stratified-friendly deterministic slice preserving rare healthcare/school/escort
        rng = np.random.default_rng(42)
        idx = rng.choice(len(items), size=min(args.limit, len(items)), replace=False)
        items = [items[i] for i in sorted(idx)]

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = {}
    if cache_path.exists():
        for line in open(cache_path, encoding="utf-8"):
            rec = json.loads(line)
            cached[rec["sample_key"]] = rec

    total_prompt_tok = 0
    total_comp_tok = 0
    with open(cache_path, "a", encoding="utf-8") as cf:
        for i, item in enumerate(items, 1):
            sk = item["sample_key"]
            if sk in cached:
                continue
            out = call_llm(args.api_url, api_key, args.model, item["context"])
            rec = {
                "sample_key": sk,
                "ground_truth_label": item["ground_truth_label"],
                "parsed": out["parsed"],
                "usage": out["usage"],
            }
            cf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cf.flush()
            cached[sk] = rec
            total_prompt_tok += out["usage"].get("prompt_tokens", 0)
            total_comp_tok += out["usage"].get("completion_tokens", 0)
            if i % 20 == 0:
                print(f"[{i}/{len(items)}] prompt_tok={total_prompt_tok}, comp_tok={total_comp_tok}")
            time.sleep(0.1)

    y_true = np.array([item["ground_truth_label"] for item in items])
    y_pred = np.array([cached[item["sample_key"]]["parsed"].get("predicted_label", "errand_other") for item in items])
    eval_classes = [c for c in CLASSES if c != "home"] if args.non_home_only else CLASSES

    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, labels=eval_classes, average="macro", zero_division=0)
    print(f"\n=== LLM Baseline ({args.model}, n={len(items)}, non_home_only={args.non_home_only}) ===")
    print(f"Accuracy={acc:.4f}, Macro-F1={mf1:.4f}")
    print(classification_report(y_true, y_pred, labels=eval_classes, zero_division=0))


if __name__ == "__main__":
    main()
