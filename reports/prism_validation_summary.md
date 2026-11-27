# Chicago time-space prism validation

The corridor cache contains 1,961,895 graph nodes, 2,780,181 undirected CSR edges and 84,837 micro-zones across 2,079 road-served tracts of 2,099. The 126,892 snapped and assigned POIs average 1.50 per zone and 60.45 per tract.
The bounding box yielded 129,434 POIs, of which 126,994 snapped within 300 m and 126,892 landed in a tract micro-zone. Step 01 assigned 126,854 by joining the POI coordinates directly to tract polygons.

The last full corridor cache build took 18.840 s through the bulk array save.

Interior intervals use unwrapped AVI time and require 5 minutes of dwell. Start, end and zero-observation intervals use an assumed 15-minute reachability window because day-anchor times are unobserved. Boundaries are ineligible for transit pruning.

The bidirectional travel-time graph uses optimistic class speeds, retained shared OSM vertices and virtual tract access links. A detector hit exposes the retained nodes on its edge as possible positions. Physical edge coverage requires every retained segment of a physical token to be feasible in the tested prism or union. Interval coverage uses the aligned target path for each interior gap. The primary measure uses the withheld physical edge occurrences strictly between the AVI hits. The endpoint-inclusive measure also tests the complete AVI boundary edges. Virtual links are excluded throughout.

Top-B cuts are applied to each interval's ranked micro-zones before converting them to parent tracts. Interval-matched recall requires the whole stop to fit in one AVI gap, with target trip order resolving rounded timestamp ties. An unmatched stop remains in the denominator as an interval miss. Chain-union recall tests the tract anywhere in the chain, so it can count a hit even if the correct interval was pruned. The first and last activities are excluded from the intermediate-stop audit.

## 324 pure-auto chains

985 intermediate stops, 844 non-home. AVI interval matched 100.00% (985/985).
Excluded non-activity labels 3 (not_imputable 3). Stop-matching statuses were time_contained 975, trip_order_tie_break 10.

| Stop group | Matching | Top-10 | Top-20 | Top-30 | Top-50 | Full prism |
|---|---|---:|---:|---:|---:|---:|
| Overall | Interval | 77.26% (761/985) | 84.37% (831/985) | 87.31% (860/985) | 89.95% (886/985) | 91.78% (904/985) |
| Overall | Chain | 86.40% (851/985) | 92.49% (911/985) | 94.62% (932/985) | 97.16% (957/985) | 100.00% (985/985) |
| Non-home | Interval | 74.05% (625/844) | 82.11% (693/844) | 85.55% (722/844) | 88.63% (748/844) | 90.64% (765/844) |
| Non-home | Chain | 84.12% (710/844) | 91.23% (770/844) | 93.72% (791/844) | 96.68% (816/844) | 100.00% (844/844) |
| home | Interval | 96.45% (136/141) | 97.87% (138/141) | 97.87% (138/141) | 97.87% (138/141) | 98.58% (139/141) |
| home | Chain | 100.00% (141/141) | 100.00% (141/141) | 100.00% (141/141) | 100.00% (141/141) | 100.00% (141/141) |
| work | Interval | 82.74% (139/168) | 88.69% (149/168) | 91.07% (153/168) | 94.05% (158/168) | 97.02% (163/168) |
| work | Chain | 88.69% (149/168) | 94.05% (158/168) | 95.83% (161/168) | 98.21% (165/168) | 100.00% (168/168) |
| school | Interval | 53.85% (14/26) | 76.92% (20/26) | 76.92% (20/26) | 80.77% (21/26) | 80.77% (21/26) |
| school | Chain | 73.08% (19/26) | 92.31% (24/26) | 92.31% (24/26) | 96.15% (25/26) | 100.00% (26/26) |
| shop | Interval | 77.84% (144/185) | 85.95% (159/185) | 88.65% (164/185) | 91.89% (170/185) | 94.59% (175/185) |
| shop | Chain | 85.95% (159/185) | 91.89% (170/185) | 95.14% (176/185) | 96.76% (179/185) | 100.00% (185/185) |
| meal | Interval | 75.82% (69/91) | 86.81% (79/91) | 89.01% (81/91) | 91.21% (83/91) | 91.21% (83/91) |
| meal | Chain | 83.52% (76/91) | 93.41% (85/91) | 93.41% (85/91) | 97.80% (89/91) | 100.00% (91/91) |
| social_recreational | Interval | 76.92% (80/104) | 82.69% (86/104) | 90.38% (94/104) | 94.23% (98/104) | 97.12% (101/104) |
| social_recreational | Chain | 82.69% (86/104) | 87.50% (91/104) | 93.27% (97/104) | 97.12% (101/104) | 100.00% (104/104) |
| escort | Interval | 59.26% (80/135) | 66.67% (90/135) | 68.89% (93/135) | 70.37% (95/135) | 71.85% (97/135) |
| escort | Chain | 83.70% (113/135) | 90.37% (122/135) | 92.59% (125/135) | 94.07% (127/135) | 100.00% (135/135) |
| healthcare | Interval | 72.73% (24/33) | 90.91% (30/33) | 90.91% (30/33) | 93.94% (31/33) | 93.94% (31/33) |
| healthcare | Chain | 78.79% (26/33) | 93.94% (31/33) | 93.94% (31/33) | 96.97% (32/33) | 100.00% (33/33) |
| errand_other | Interval | 73.53% (75/102) | 78.43% (80/102) | 85.29% (87/102) | 90.20% (92/102) | 92.16% (94/102) |
| errand_other | Chain | 80.39% (82/102) | 87.25% (89/102) | 90.20% (92/102) | 96.08% (98/102) | 100.00% (102/102) |

- Transit-only intervals among all gaps 77.78% (7,839/10,079). Among interior gaps 83.11% (7,839/9,432).
- Transit-pruning false negatives 8.12% (80/985) of all true intermediate stops and 8.12% (80/985) among matched stops.
- Interval statuses were reachable 9,918, travel_budget_infeasible 161.
- Chain-union physical target-edge coverage 100.00% (97,468/97,468). Withheld target-edge coverage 100.00% (87,713/87,713).
- Mean chain union retained 1,599,071 nodes (81.51% of the graph), 2,279,508 CSR edges (81.99%) and 1,697.7 candidate tracts (80.88%).
- Mean chain union retained 144,077 of the 171,575 vocabulary edge tokens (83.97%).
- Per-chain prediction runtime in seconds was 2.363 median, 5.972 p90 and 21.345 maximum with 8 workers.

### Infeasible interval audit

Travel-budget infeasible gaps account for 1.71% (161/9,432) of interior intervals and contain 0.00% (0/985) of intermediate stops. Boundary infeasibilities total 0. These gaps are tracked separately from reachable, transit-only gaps.

| Infeasible interior gaps | Count | Span min, median / p90 / max | Shortest minus span min, median / p90 / max |
|---|---:|---:|---:|
| all infeasible | 161 | 0.0500 / 0.8333 / 2.7000 | 0.2622 / 1.1250 / 4.8165 |
| zero span | 66 | 0.0000 / 0.0000 / 0.0000 | 0.3786 / 1.1155 / 4.8165 |
| positive span | 95 | 0.2834 / 1.1433 / 2.7000 | 0.2016 / 1.2009 / 3.4881 |

### Per-interval edge coverage

| Interior gaps | Aligned | Withheld edge coverage | Mean per-gap withheld coverage | With AVI boundary edges |
|---|---:|---:|---:|---:|
| overall | 9,432/9,432 | 97.29% (80,239/82,473) | 95.87% (8,498 gaps) | 96.45% (97,741/101,337) |
| excluding broad component | 9,242/9,242 | 97.17% (76,748/78,982) | 95.77% (8,309 gaps) | 96.31% (93,870/97,466) |

### Per-interval compression

Of intervals with at least one stop candidate, 9.15% (190/2,076) cover the entire main component and 90.85% (1,886/2,076) use localized prisms. Broad gaps occur in 50.93% (165/324) of chains. Their span in minutes is 373.8 / 586.5 / 878.5 at median / p90 / max.

Broad multi-hour gaps contribute to the high chain-union full-prism recall and edge coverage by making the entire main component available to those chains. Removing the flagged broad intervals gives 97.26% (958/985) chain-union full-prism stop recall and 99.93% (97,403/97,468) chain-union physical edge coverage. Unflagged prisms can also span large areas.

| Localized active interval | Median / p75 / p90 |
|---|---:|
| full candidate micro zones | 1,842.0 / 8,247.8 / 67,097.5 |
| full candidate tracts | 94.0 / 413.0 / 2,001.0 |
| prism nodes | 153,334.5 / 465,244.5 / 1,841,956.0 |
| prism edges | 226,628.0 / 684,064.0 / 2,631,893.5 |

| Cutoff | Mean distinct tracts per active gap | Mean distinct tracts per chain | Share of 2,099 tracts, per gap / per chain |
|---|---:|---:|---:|
| Top-10 | 3.11 | 11.32 | 0.15% / 0.54% |
| Top-20 | 4.82 | 16.60 | 0.23% / 0.79% |
| Top-30 | 6.28 | 20.63 | 0.30% / 0.98% |
| Top-50 | 8.72 | 26.82 | 0.42% / 1.28% |

| Pruned stop audit | All true stops in group |
|---|---:|
| home | 1.42% (2/141) |
| work | 2.98% (5/168) |
| school | 15.38% (4/26) |
| shop | 5.41% (10/185) |
| meal | 8.79% (8/91) |
| social_recreational | 2.88% (3/104) |
| escort | 28.15% (38/135) |
| healthcare | 6.06% (2/33) |
| errand_other | 7.84% (8/102) |
| <1 min | 49.64% (69/139) |
| 1–<5 min | 19.30% (11/57) |
| 5–<15 min | 0.00% (0/143) |
| 15–<60 min | 0.00% (0/242) |
| 60+ min | 0.00% (0/404) |
| unknown min | — |

## 100-chain MVP subset

302 intermediate stops, 260 non-home. AVI interval matched 100.00% (302/302).
Excluded non-activity labels 1 (not_imputable 1). Stop-matching statuses were time_contained 299, trip_order_tie_break 3.

| Stop group | Matching | Top-10 | Top-20 | Top-30 | Top-50 | Full prism |
|---|---|---:|---:|---:|---:|---:|
| Overall | Interval | 77.81% (235/302) | 85.43% (258/302) | 89.07% (269/302) | 90.40% (273/302) | 92.05% (278/302) |
| Overall | Chain | 86.75% (262/302) | 93.71% (283/302) | 96.03% (290/302) | 97.02% (293/302) | 100.00% (302/302) |
| Non-home | Interval | 74.62% (194/260) | 83.46% (217/260) | 87.69% (228/260) | 89.23% (232/260) | 91.15% (237/260) |
| Non-home | Chain | 84.62% (220/260) | 92.69% (241/260) | 95.38% (248/260) | 96.54% (251/260) | 100.00% (260/260) |
| home | Interval | 97.62% (41/42) | 97.62% (41/42) | 97.62% (41/42) | 97.62% (41/42) | 97.62% (41/42) |
| home | Chain | 100.00% (42/42) | 100.00% (42/42) | 100.00% (42/42) | 100.00% (42/42) | 100.00% (42/42) |
| work | Interval | 80.33% (49/61) | 88.52% (54/61) | 91.80% (56/61) | 93.44% (57/61) | 98.36% (60/61) |
| work | Chain | 85.25% (52/61) | 91.80% (56/61) | 95.08% (58/61) | 96.72% (59/61) | 100.00% (61/61) |
| school | Interval | 62.50% (5/8) | 87.50% (7/8) | 87.50% (7/8) | 87.50% (7/8) | 87.50% (7/8) |
| school | Chain | 87.50% (7/8) | 100.00% (8/8) | 100.00% (8/8) | 100.00% (8/8) | 100.00% (8/8) |
| shop | Interval | 80.00% (44/55) | 89.09% (49/55) | 92.73% (51/55) | 94.55% (52/55) | 96.36% (53/55) |
| shop | Chain | 89.09% (49/55) | 94.55% (52/55) | 98.18% (54/55) | 98.18% (54/55) | 100.00% (55/55) |
| meal | Interval | 76.00% (19/25) | 88.00% (22/25) | 92.00% (23/25) | 92.00% (23/25) | 92.00% (23/25) |
| meal | Chain | 84.00% (21/25) | 96.00% (24/25) | 96.00% (24/25) | 96.00% (24/25) | 100.00% (25/25) |
| social_recreational | Interval | 69.70% (23/33) | 75.76% (25/33) | 90.91% (30/33) | 93.94% (31/33) | 96.97% (32/33) |
| social_recreational | Chain | 78.79% (26/33) | 84.85% (28/33) | 93.94% (31/33) | 96.97% (32/33) | 100.00% (33/33) |
| escort | Interval | 60.98% (25/41) | 68.29% (28/41) | 68.29% (28/41) | 68.29% (28/41) | 68.29% (28/41) |
| escort | Chain | 78.05% (32/41) | 90.24% (37/41) | 90.24% (37/41) | 90.24% (37/41) | 100.00% (41/41) |
| healthcare | Interval | 70.00% (7/10) | 90.00% (9/10) | 90.00% (9/10) | 100.00% (10/10) | 100.00% (10/10) |
| healthcare | Chain | 80.00% (8/10) | 90.00% (9/10) | 90.00% (9/10) | 100.00% (10/10) | 100.00% (10/10) |
| errand_other | Interval | 81.48% (22/27) | 85.19% (23/27) | 88.89% (24/27) | 88.89% (24/27) | 88.89% (24/27) |
| errand_other | Chain | 92.59% (25/27) | 100.00% (27/27) | 100.00% (27/27) | 100.00% (27/27) | 100.00% (27/27) |

- Transit-only intervals among all gaps 77.76% (2,545/3,273). Among interior gaps 82.82% (2,545/3,073).
- Transit-pruning false negatives 7.95% (24/302) of all true intermediate stops and 7.95% (24/302) among matched stops.
- Interval statuses were reachable 3,202, travel_budget_infeasible 71.
- Chain-union physical target-edge coverage 100.00% (31,702/31,702). Withheld target-edge coverage 100.00% (28,529/28,529).
- Mean chain union retained 1,586,051 nodes (80.84% of the graph), 2,260,254 CSR edges (81.30%) and 1,680.1 candidate tracts (80.04%).
- Mean chain union retained 142,306 of the 171,575 vocabulary edge tokens (82.94%).
- Per-chain prediction runtime in seconds was 2.449 median, 5.821 p90 and 15.852 maximum with 8 workers.

### Infeasible interval audit

Travel-budget infeasible gaps account for 2.31% (71/3,073) of interior intervals and contain 0.00% (0/302) of intermediate stops. Boundary infeasibilities total 0. These gaps are tracked separately from reachable, transit-only gaps.

| Infeasible interior gaps | Count | Span min, median / p90 / max | Shortest minus span min, median / p90 / max |
|---|---:|---:|---:|
| all infeasible | 71 | 0.0500 / 1.1500 / 2.7000 | 0.3998 / 1.4612 / 4.8165 |
| zero span | 33 | 0.0000 / 0.0000 / 0.0000 | 0.4229 / 1.3902 / 4.8165 |
| positive span | 38 | 0.4833 / 2.2150 / 2.7000 | 0.3599 / 1.4014 / 3.4881 |

### Per-interval edge coverage

| Interior gaps | Aligned | Withheld edge coverage | Mean per-gap withheld coverage | With AVI boundary edges |
|---|---:|---:|---:|---:|
| overall | 3,073/3,073 | 96.39% (25,914/26,885) | 95.27% (2,764 gaps) | 95.62% (31,585/33,031) |
| excluding broad component | 3,008/3,008 | 96.24% (24,828/25,799) | 95.15% (2,699 gaps) | 95.45% (30,369/31,815) |

### Per-interval compression

Of intervals with at least one stop candidate, 9.91% (65/656) cover the entire main component and 90.09% (591/656) use localized prisms. Broad gaps occur in 56.00% (56/100) of chains. Their span in minutes is 324.0 / 564.1 / 708.1 at median / p90 / max.

Broad multi-hour gaps contribute to the high chain-union full-prism recall and edge coverage by making the entire main component available to those chains. Removing the flagged broad intervals gives 96.69% (292/302) chain-union full-prism stop recall and 99.93% (31,680/31,702) chain-union physical edge coverage. Unflagged prisms can also span large areas.

| Localized active interval | Median / p75 / p90 |
|---|---:|
| full candidate micro zones | 1,753.0 / 6,444.5 / 61,008.0 |
| full candidate tracts | 82.0 / 390.0 / 1,911.0 |
| prism nodes | 150,438.0 / 410,847.5 / 1,764,622.0 |
| prism edges | 223,310.0 / 598,232.0 / 2,525,822.0 |

| Cutoff | Mean distinct tracts per active gap | Mean distinct tracts per chain | Share of 2,099 tracts, per gap / per chain |
|---|---:|---:|---:|
| Top-10 | 3.25 | 12.43 | 0.15% / 0.59% |
| Top-20 | 5.10 | 18.21 | 0.24% / 0.87% |
| Top-30 | 6.73 | 22.79 | 0.32% / 1.09% |
| Top-50 | 9.21 | 29.60 | 0.44% / 1.41% |

| Pruned stop audit | All true stops in group |
|---|---:|
| home | 2.38% (1/42) |
| work | 1.64% (1/61) |
| school | 12.50% (1/8) |
| shop | 3.64% (2/55) |
| meal | 8.00% (2/25) |
| social_recreational | 3.03% (1/33) |
| escort | 31.71% (13/41) |
| healthcare | 0.00% (0/10) |
| errand_other | 11.11% (3/27) |
| <1 min | 62.50% (20/32) |
| 1–<5 min | 26.67% (4/15) |
| 5–<15 min | 0.00% (0/49) |
| 15–<60 min | 0.00% (0/82) |
| 60+ min | 0.00% (0/124) |
| unknown min | — |

Total script time before writing the summary 128.142 s.
