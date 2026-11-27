# 稀疏卡口下的车辆出行活动推断

本仓库记录一项硕士毕设预实验。给定少量模拟卡口的路段与时间、全天起终点的已知普查区，以及芝加哥路网和兴趣点（POI），目标是重构车辆的行驶路径、中途停留、时长和活动类型。实验用时空棱柱生成候选地点，用活动安排模型（HAPP）联合判断停留和活动，再检验引入 Gemini 的活动与地点先验后，各环节的表现如何变化。

## 实验设置与结果

卡口观测由芝加哥家庭出行调查的匹配轨迹按 10% 物理路段保留协议模拟。出行调查提供活动目的和时间参考，匹配轨迹提供路径参考。纯小汽车样本按家庭分开，含 1,351 条训练链、276 条验证链和 324 条测试链，其中固定 100 条用于端到端 Gemini 对比。

| 100 条测试链 | 分步基线 M1 | 联合模型 M3 | M3 加校准后的 Gemini |
| --- | ---: | ---: | ---: |
| 停留检测 F1 | 80.57% | 79.05% | 77.70% |
| 停留间隔与普查区都正确的 F1 | 37.11% | 38.49% | 39.30% |
| 匹配停留上的非居家活动宏 F1 | 20.74% | 23.46% | 27.60% |
| 停留间隔、普查区与活动都正确的 F1 | 18.56% | 18.09% | 18.53% |
| 隐藏物理路段集合 F1 | 66.84% | 67.61% | 67.37% |

Gemini 版本在匹配停留上的活动分类表现更好。该指标取决于各模型成功匹配的停留集合。无 LLM 的 M3 使用 276 条验证链标定，加入 LLM 的版本使用 60 条验证链标定。这一比较衡量的是两套流程在当前设置下的差异。模拟布点、已知日首尾片区以及每个卡口间隔至多一次停留的设定，限制了结果向真实卡口网络和连续短停留场景的推广。

- [时空棱柱候选覆盖率](reports/prism_validation_summary.md)
- [324 条测试链的端到端重构](reports/e2e_reconstruction_summary.md)
- [100 条测试链的 Gemini 对比](reports/e2e_llm_comparison_summary.md)及[汇总数据](reports/e2e_llm_comparison_100chains.json)

## 运行条件

使用 Python 3.12 或更新版本，依赖见 `requirements.txt`。在 macOS 上，LightGBM 可能还需要 OpenMP 运行库。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

运行需要有权限访问的芝加哥调查与预处理资产。准备数据时保留以下相对路径。脚本 01 会补齐兴趣点特征。精确复跑原结果时需要原实验留存的同版本 POI 数据与模型响应。

```text
Chicago-Trip/
├── Chicago/Tracts/
│   ├── TractZonalHistogram.geojson
│   ├── tract_long_table.csv
│   └── tract_virtual_links.geojsonl
└── data/
    ├── Rawdata/Chicago/
    │   ├── MyDailyTravelData/                 # 包含 trip_linked.csv 和 day.csv
    │   └── Transportation Network/road_tract_network.geojsonl
    └── processed_fixed/strict_chicago_all_usable_10pct/
        ├── masked_train.jsonl
        ├── masked_val.jsonl
        ├── masked_test.jsonl
        ├── edge_vocab.json
        └── household_split.json
```

已有初版处理结果时，可以用 `python Chicago-Trip/tools/fix_processed.py --root Chicago-Trip` 生成修订版。该工具需要既有的 `Chicago-Trip/data/processed/` 和 `Chicago-Trip/data/processed_strict/`，上游轨迹匹配和观测遮挡流程需要另行获取。

## 运行顺序

从仓库根目录运行。生成物写入 `mvp_experiment/data/` 与 `mvp_experiment/results/`，已归档的汇总保存在 `reports/`。

```sh
python mvp_experiment/scripts/01_fetch_chicago_poi.py
python mvp_experiment/scripts/02_build_mvp_dataset.py
python mvp_experiment/scripts/05_prepare_100chain_eval.py
python mvp_experiment/scripts/09_build_corridor_network.py
python mvp_experiment/scripts/10_build_reduced_prism_graphs.py
python mvp_experiment/scripts/11_train_observable_scorer_and_val_prisms.py
python mvp_experiment/scripts/12_reconstruct_trip_chains.py
python mvp_experiment/scripts/13_prepare_corridor_llm_batches.py
```

- 脚本 03–08 是给定调查停留特征的活动分类预实验。脚本 06 和 14 是 Antigravity workflow 源码，请从仓库根目录在对应运行器中执行。脚本 07 需要 Codex CLI，脚本 04 需要所选模型 API 的密钥。
- 端到端模型对比用脚本 15，取脚本 13 的批次清单和 `mvp_experiment/results/corridor_gemini_{val60,test100}.json` 预测缓存。脚本 14 生成初始预测，原实验另有缺失条目的补跑。完整复跑所需的调查记录、样本级预测和原运行缓存由本地保管。

## 研究来源

- 稀疏卡口下的路径、停留地点与时长推断参考 [Cao 等，2024](https://www.sciencedirect.com/science/article/pii/S0965856424003124)。
- 大模型辅助兴趣点语义和已识别停留的活动推断见 [Liu 等，2024](https://arxiv.org/abs/2405.11715)。
- POI 来自 [OpenStreetMap 贡献者](https://www.openstreetmap.org/copyright)，通过 [Geofabrik 伊利诺伊州下载页](https://download.geofabrik.de/north-america/us/illinois.html)获取。
