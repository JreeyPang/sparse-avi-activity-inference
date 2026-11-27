# End-to-end sparse AVI reconstruction

The household-disjoint split has 1,351 training days, 276 validation days, and 324 held-out days. The 100-day cohort uses the fixed step 05 selection. Predictions use AVI tokens and unwrapped timestamps, start/end tracts, and weekday/weekend from the survey calendar join on day ID. Training labels and schedule bands are from the canonical step 02 activities.

The validation stop probabilities come from 5-fold GroupKFold grouped by household_id across 115 households. Each day is scored by a stop classifier trained on the other four folds. The M1 threshold, M3 separation oracle and checkpoint selection all use those out-of-fold probabilities. A separate stop classifier fit on all 276 validation days scores the test candidates. Threshold and checkpoint metrics summarize development-set selection using OOF inputs.

M1 selected a threshold of **0.67** on the OOF validation probabilities (stop F1 **86.20%**).

| Stop fold | Fit households | Held-out households | Fit days | Held-out days | Fit active gaps | Held-out active gaps |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 92 | 23 | 220 | 56 | 2330 | 662 |
| 2 | 92 | 23 | 221 | 55 | 2420 | 572 |
| 3 | 92 | 23 | 221 | 55 | 2388 | 604 |
| 4 | 92 | 23 | 221 | 55 | 2367 | 625 |
| 5 | 92 | 23 | 221 | 55 | 2463 | 529 |

Step 11 fits activity probabilities on observable candidate features in the labeled training tract. It learns 20th–80th percentile preferred time bands and 1st–99th percentile hard dwell bounds. Interior candidate dwell is at least 1.5 minutes. Boundaries use a 15-minute road-access budget and assumed day bounds. The top 20 candidates include at most two zones per tract. Shortest-path travel time is in minutes. Euclidean chord distance in metres is an additional straight-line path-length proxy.

M1 applies the calibrated physical gap threshold, picks the top physical zone and assigns the raw activity argmax. M2 jointly selects skip or one stop per AVI interval, with continuous arrival and class-gated dwell, train-derived soft schedule penalties, consistent work/school tracts and daily caps. This includes intervals over 120 minutes. M3 uses the same model with validation-calibrated weights. The joint models prefer longer dwell among equal-cost schedule vertices, preserving the primary MILP objective. M1 uses the feasible dwell up to the upper preferred duration band.

Routing chooses a continuous sequence of orientations along the observed road tokens and uses exact shortest graph legs through each selected stop. Among orientations it first minimizes disconnections and schedule conflicts, then travel time. The graph permits virtual road-access links. Only physical tokens enter link scores. The route audit counts virtual/unmapped segments, disconnections and timing conflicts between specific oriented road endpoints and the optimistic prism.

Stop detection counts occupied AVI gaps. Spatial and triple scores use one-to-one exact keys with all intermediate stops in the denominator. Chain-order tract F1 uses the longest common subsequence. Schedule, centroid and purpose metrics pair stops in the same interval, choosing the exact tract when available and the first surveyed stop when a tie remains. Non-home classification metrics are conditioned on non-home truth. Sequence similarity uses the original start and end slots, with an explicit `<missing>` token for each unavailable anchor in its all-day metric. The valid-anchor-only metric uses days with both anchor labels available. Unlabeled intermediate activities are excluded from stop and sequence metrics and counted below. Path PRF pools within-day link-set counts. Withheld scoring subtracts all given tokens from both predicted and true sets.

## All 324 days

985 scored intermediate stops, 0 without an assignable AVI interval. The 3,966 active intervals average 9.05 candidate tracts.

Missing survey labels affected 12 start anchors and 0 end anchors. 12 days had at least one missing anchor and 0 had both missing. The valid-anchor-only sequence denominator is 312 days. Excluded intermediate activities with missing or non-activity labels numbered 3.

| Candidate tract recall in true interval | Hits / true stops | Recall |
|---|---:|---:|
| Overall | 948 / 985 | 96.24% |
| home | 140 / 141 | 99.29% |
| work | 162 / 168 | 96.43% |
| school | 22 / 26 | 84.62% |
| shop | 180 / 185 | 97.30% |
| meal | 88 / 91 | 96.70% |
| social_recreational | 103 / 104 | 99.04% |
| escort | 121 / 135 | 89.63% |
| healthcare | 33 / 33 | 100.00% |
| errand_other | 99 / 102 | 97.06% |

The prior step 10 Top-20 audit recovered 90/135 escort tracts (66.67%) in the true gap and averaged 4.82 tracts per active gap. Here the corresponding escort change is +22.96 percentage points.

| Metric | Model | Precision | Recall | F1 | TP / predicted / true |
|---|---|---:|---:|---:|---:|
| Stop detection | M1 | 79.10% | 77.98% | 78.54% | 719 / 909 / 922 |
| Stop detection | M2 | 61.56% | 86.33% | 71.87% | 796 / 1,293 / 922 |
| Stop detection | M3 | 75.82% | 80.26% | 77.98% | 740 / 976 / 922 |
| Exact interval + tract | M1 | 38.61% | 35.63% | 37.06% | 351 / 909 / 985 |
| Exact interval + tract | M2 | 23.74% | 31.17% | 26.95% | 307 / 1,293 / 985 |
| Exact interval + tract | M3 | 41.80% | 41.42% | 41.61% | 408 / 976 / 985 |
| Chain-order tract | M1 | 42.02% | 38.78% | 40.34% | 382 / 909 / 985 |
| Chain-order tract | M2 | 28.00% | 36.75% | 31.78% | 362 / 1,293 / 985 |
| Chain-order tract | M3 | 45.29% | 44.87% | 45.08% | 442 / 976 / 985 |
| Interval + tract + activity | M1 | 20.68% | 19.09% | 19.85% | 188 / 909 / 985 |
| Interval + tract + activity | M2 | 12.30% | 16.14% | 13.96% | 159 / 1,293 / 985 |
| Interval + tract + activity | M3 | 19.88% | 19.70% | 19.79% | 194 / 976 / 985 |
| Physical link set | M1 | 72.44% | 73.08% | 72.76% | 49,626 / 68,505 / 67,907 |
| Physical link set | M2 | 67.70% | 72.52% | 70.03% | 49,248 / 72,745 / 67,907 |
| Physical link set | M3 | 73.60% | 73.11% | 73.36% | 49,648 / 67,455 / 67,907 |
| Withheld physical link set | M1 | 68.03% | 68.73% | 68.38% | 40,181 / 59,060 / 58,462 |
| Withheld physical link set | M2 | 62.88% | 68.08% | 65.38% | 39,803 / 63,300 / 58,462 |
| Withheld physical link set | M3 | 69.30% | 68.77% | 69.03% | 40,203 / 58,010 / 58,462 |

M2 changes stop F1 by -6.66 pp, strict triple F1 by -5.89 pp, and withheld physical F1 by -3.00 pp relative to M1. Validation calibration changes those three scores by +6.10 pp, +5.83 pp, and +3.66 pp relative to M2.

| Matched-stop and day metrics | M1 | M2 | M3 |
|---|---:|---:|---:|
| Daily stop count MAE | 1.210 | 1.753 | 1.269 |
| Median tract centroid error (km) | 0.611 | 1.126 | 0.000 |
| Mean tract centroid error (km) | 1.096 | 1.384 | 0.924 |
| P75 tract centroid error (km) | 1.838 | 2.321 | 1.617 |
| Median tract centroid error, wrong tract only (km) | 1.826 | 1.986 | 1.785 |
| Arrival MAE (min) | 15.092 | 17.704 | 17.271 |
| Duration MAE (min) | 26.139 | 30.803 | 27.990 |
| Daily activity sequence similarity, all days | 57.01% | 52.28% | 56.12% |
| Daily activity sequence similarity, both anchor labels valid | 58.04% | 53.25% | 56.95% |
| Matched activity accuracy | 42.70% | 38.19% | 40.14% |
| Matched non-home activity accuracy | 35.01% | 33.93% | 33.76% |
| Matched non-home macro F1 | 24.45% | 30.29% | 26.70% |

| Matched activity class F1 (true support) | M1 | M2 | M3 |
|---|---:|---:|---:|
| home | 59.04% (122) | 53.47% (127) | 62.86% (121) |
| work | 66.90% (153) | 61.67% (157) | 59.61% (152) |
| school | 8.33% (18) | 22.22% (20) | 5.71% (19) |
| shop | 45.08% (127) | 41.99% (148) | 42.40% (135) |
| meal | 6.52% (53) | 20.65% (67) | 10.81% (56) |
| social_recreational | 32.47% (89) | 33.93% (94) | 34.44% (90) |
| escort | 12.35% (64) | 25.17% (84) | 28.37% (71) |
| healthcare | 5.13% (28) | 11.11% (28) | 9.30% (28) |
| errand_other | 15.00% (65) | 17.54% (71) | 17.11% (68) |

From M2 to M3, matched F1 for school fell (22.22% → 5.71%) and for healthcare fell (11.11% → 9.30%). Tail weighting changed the validation loss used to fit shared model weights. Class-specific evidence came from observable activity scores and POI features, without LLM priors. Matched class F1 also depends on which stops each model recovers.

### Bidirectional coupling

M3 added 67 and removed 0 stops relative to M1 across 3,966 active gaps. The two-stop comparison isolates tract changes. Within the ambiguous gaps, M3 added 67 stops and removed 0.

| Gaps | Count | Corrected | Harmed | Net | Withheld gap F1 M1 → M3 | Δ gap F1 | Δ whole-day F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Both stop, changed tract | 232 | 82 (35.34%) | 39 (16.81%) | +43 | 26.78% → 36.37% | +9.59 pp | +1.28 pp |
| At least two candidate tracts | 3,841 | 96 (2.50%) | 39 (1.02%) | +57 | 56.06% → 57.18% | +1.13 pp | +0.65 pp |
| Ambiguous and disagree on stop existence | 67 | 14 (20.90%) | 0 (0.00%) | +14 | 51.85% → 40.21% | -11.63 pp | +0.34 pp |

HiGHS timing percentiles use solved calls only. Days with trivially pruned MILPs are counted separately.

| Model | HiGHS median / p90 / max (ms) | Solved | Pruned | Calls >50 ms | Broken gaps | Virtual/unmapped segments | Travel + dwell over gap | Route–schedule conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | — / — / — | 0 | 0 | 0 | 0 | 25230 | 547 | 281 |
| M2 | 3.614 / 4.751 / 13.450 | 313 | 11 | 0 | 0 | 31139 | 648 | 409 |
| M3 | 3.767 / 4.768 / 5.234 | 300 | 24 | 0 | 0 | 22556 | 544 | 304 |

## Fixed 100-day MVP subset

302 scored intermediate stops, 0 without an assignable AVI interval. The 1,282 active intervals average 9.15 candidate tracts.

Missing survey labels affected 3 start anchors and 0 end anchors. 3 days had at least one missing anchor and 0 had both missing. The valid-anchor-only sequence denominator is 97 days. Excluded intermediate activities with missing or non-activity labels numbered 1.

| Candidate tract recall in true interval | Hits / true stops | Recall |
|---|---:|---:|
| Overall | 290 / 302 | 96.03% |
| home | 42 / 42 | 100.00% |
| work | 58 / 61 | 95.08% |
| school | 7 / 8 | 87.50% |
| shop | 54 / 55 | 98.18% |
| meal | 24 / 25 | 96.00% |
| social_recreational | 32 / 33 | 96.97% |
| escort | 36 / 41 | 87.80% |
| healthcare | 10 / 10 | 100.00% |
| errand_other | 27 / 27 | 100.00% |

| Metric | Model | Precision | Recall | F1 | TP / predicted / true |
|---|---|---:|---:|---:|---:|
| Stop detection | M1 | 81.43% | 79.72% | 80.57% | 228 / 280 / 286 |
| Stop detection | M2 | 60.19% | 86.71% | 71.06% | 248 / 412 / 286 |
| Stop detection | M3 | 76.47% | 81.82% | 79.05% | 234 / 306 / 286 |
| Exact interval + tract | M1 | 38.57% | 35.76% | 37.11% | 108 / 280 / 302 |
| Exact interval + tract | M2 | 23.06% | 31.46% | 26.61% | 95 / 412 / 302 |
| Exact interval + tract | M3 | 38.24% | 38.74% | 38.49% | 117 / 306 / 302 |
| Chain-order tract | M1 | 42.50% | 39.40% | 40.89% | 119 / 280 / 302 |
| Chain-order tract | M2 | 27.43% | 37.42% | 31.65% | 113 / 412 / 302 |
| Chain-order tract | M3 | 41.83% | 42.38% | 42.11% | 128 / 306 / 302 |
| Interval + tract + activity | M1 | 19.29% | 17.88% | 18.56% | 54 / 280 / 302 |
| Interval + tract + activity | M2 | 11.65% | 15.89% | 13.45% | 48 / 412 / 302 |
| Interval + tract + activity | M3 | 17.97% | 18.21% | 18.09% | 55 / 306 / 302 |
| Physical link set | M1 | 70.74% | 72.14% | 71.43% | 15,801 / 22,337 / 21,904 |
| Physical link set | M2 | 65.25% | 71.47% | 68.22% | 15,655 / 23,992 / 21,904 |
| Physical link set | M3 | 71.94% | 72.31% | 72.13% | 15,839 / 22,016 / 21,904 |
| Withheld physical link set | M1 | 66.09% | 67.61% | 66.84% | 12,737 / 19,273 / 18,840 |
| Withheld physical link set | M2 | 60.16% | 66.83% | 63.32% | 12,591 / 20,928 / 18,840 |
| Withheld physical link set | M3 | 67.41% | 67.81% | 67.61% | 12,775 / 18,952 / 18,840 |

M2 changes stop F1 by -9.51 pp, strict triple F1 by -5.11 pp, and withheld physical F1 by -3.52 pp relative to M1. Validation calibration changes those three scores by +7.99 pp, +4.65 pp, and +4.28 pp relative to M2.

| Matched-stop and day metrics | M1 | M2 | M3 |
|---|---:|---:|---:|
| Daily stop count MAE | 1.060 | 1.800 | 1.120 |
| Median tract centroid error (km) | 0.668 | 1.141 | 0.211 |
| Mean tract centroid error (km) | 1.103 | 1.451 | 0.983 |
| P75 tract centroid error (km) | 1.847 | 2.441 | 1.740 |
| Median tract centroid error, wrong tract only (km) | 1.817 | 2.016 | 1.740 |
| Arrival MAE (min) | 16.676 | 17.532 | 20.227 |
| Duration MAE (min) | 25.120 | 30.536 | 30.446 |
| Daily activity sequence similarity, all days | 57.14% | 51.27% | 55.87% |
| Daily activity sequence similarity, both anchor labels valid | 58.14% | 52.28% | 56.84% |
| Matched activity accuracy | 40.35% | 35.48% | 38.89% |
| Matched non-home activity accuracy | 31.22% | 30.14% | 29.74% |
| Matched non-home macro F1 | 20.74% | 27.63% | 23.46% |

| Matched activity class F1 (true support) | M1 | M2 | M3 |
|---|---:|---:|---:|
| home | 57.89% (39) | 53.76% (39) | 67.35% (39) |
| work | 60.42% (57) | 53.01% (58) | 52.38% (56) |
| school | 0.00% (7) | 20.00% (7) | 0.00% (7) |
| shop | 38.64% (35) | 36.36% (44) | 39.51% (39) |
| meal | 6.67% (16) | 9.09% (20) | 6.25% (17) |
| social_recreational | 33.33% (27) | 30.99% (28) | 30.19% (27) |
| escort | 8.33% (21) | 22.73% (26) | 26.09% (23) |
| healthcare | 0.00% (9) | 23.53% (9) | 12.50% (9) |
| errand_other | 15.79% (17) | 17.39% (17) | 17.39% (17) |

From M2 to M3, matched F1 for school fell (20.00% → 0.00%) and for healthcare fell (23.53% → 12.50%). Tail weighting changed the validation loss used to fit shared model weights. Class-specific evidence came from observable activity scores and POI features, without LLM priors. Matched class F1 also depends on which stops each model recovers.

### Bidirectional coupling

M3 added 26 and removed 0 stops relative to M1 across 1,282 active gaps. The two-stop comparison isolates tract changes. Within the ambiguous gaps, M3 added 26 stops and removed 0.

| Gaps | Count | Corrected | Harmed | Net | Withheld gap F1 M1 → M3 | Δ gap F1 | Δ whole-day F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Both stop, changed tract | 60 | 15 (25.00%) | 11 (18.33%) | +4 | 24.05% → 36.28% | +12.23 pp | +1.57 pp |
| At least two candidate tracts | 1,253 | 20 (1.60%) | 11 (0.88%) | +9 | 54.79% → 56.23% | +1.44 pp | +0.77 pp |
| Ambiguous and disagree on stop existence | 26 | 5 (19.23%) | 0 (0.00%) | +5 | 45.58% → 43.32% | -2.27 pp | +0.85 pp |

HiGHS timing percentiles use solved calls only. Days with trivially pruned MILPs are counted separately.

| Model | HiGHS median / p90 / max (ms) | Solved | Pruned | Calls >50 ms | Broken gaps | Virtual/unmapped segments | Travel + dwell over gap | Route–schedule conflicts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 | — / — / — | 0 | 0 | 0 | 0 | 8420 | 203 | 98 |
| M2 | 3.543 / 4.725 / 6.647 | 96 | 4 | 0 | 0 | 10149 | 222 | 129 |
| M3 | 3.695 / 4.790 / 5.234 | 94 | 6 | 0 | 0 | 7489 | 200 | 106 |

## Validation-only inverse optimization

The margin-rescaled oracle minimizes reconstruction cost minus weighted task loss in HiGHS. Precomputed per-choice schedule vertices give the exact minimum schedule cost for each weight vector. The fixed feasible reference surrogate solves the same day constraints with the objective C(θ₀) + 1,000 × task loss. Each projected proximal subgradient update targets the mean structured hinge with a ridge penalty centered on θ₀. The OOF-input validation checkpoint score averages occupied-gap F1, strict triple F1 and end-to-end non-home interval/activity macro F1, with regularized loss breaking ties. Selected iteration **5**. Detour is fixed at one equivalent minute per minute.

| Cost term | M2 | M3 |
|---|---:|---:|
| detour_per_minute | 1.0000 | 1.0000 |
| stop_log_odds | 4.0000 | 6.6593 |
| log_poi_count | 0.7500 | 0.2408 |
| day_anchor | 1.5000 | 0.8863 |
| log_activity_potential | 3.0000 | 0.1484 |
| activity_poi_match | 0.7500 | 0.0000 |
| early_start_per_30min | 0.9000 | 0.3473 |
| late_start_per_30min | 0.9000 | 0.0200 |
| short_dwell_per_30min | 3.0000 | 2.7029 |
| long_dwell_per_30min | 1.5000 | 1.5000 |

| Validation class | Count | Tail weight |
|---|---:|---:|
| home | 99 | 0.900 |
| work | 141 | 0.754 |
| school | 26 | 1.755 |
| shop | 123 | 0.807 |
| meal | 87 | 0.960 |
| social_recreational | 88 | 0.954 |
| escort | 68 | 1.085 |
| healthcare | 13 | 2.482 |
| errand_other | 76 | 1.027 |

| Iteration | Mean hinge | Ridge objective | Val macro F1 | Val stop F1 | Val triple F1 |
|---:|---:|---:|---:|---:|---:|
| 0 | 52.7508 | 52.7508 | 22.52% | 79.22% | 14.66% |
| 1 | 41.3336 | 44.0262 | 13.35% | 85.94% | 8.25% |
| 2 | 41.5543 | 43.0824 | 22.60% | 84.91% | 16.32% |
| 3 | 38.3525 | 39.7330 | 24.14% | 86.25% | 18.84% |
| 4 | 37.0356 | 38.2579 | 22.80% | 86.05% | 15.01% |
| 5 | 36.2292 | 37.2834 | 24.95% | 86.14% | 18.74% |
| 6 | 35.6996 | 36.6631 | 14.08% | 86.07% | 9.89% |
| 7 | 36.0271 | 36.7034 | 24.50% | 85.94% | 19.05% |
| 8 | 35.0193 | 35.7931 | 24.95% | 86.25% | 18.54% |
| 9 | 34.6523 | 35.4540 | 12.78% | 85.96% | 10.07% |
| 10 | 35.5563 | 36.1045 | 23.67% | 85.76% | 19.13% |

## Runtime

| Stage | Seconds |
|---|---:|
| stage 11 build | 258.313 |
| stop cross fit and full fit | 0.798 |
| stage 12 initial load | 0.637 |
| validation calibration | 28.845 |
| test stop schedule decisions | 4.612 |
| network load and path recovery | 30.818 |
| evaluation and case studies | 0.811 |
| stage 12 before report write | 65.722 |
| stage 12 through json write | 66.025 |
