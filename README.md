# RA-FogNet

**Explainable Reliability-Aware Multimodal Deep Sensor Fusion for Predictive
Fog Risk Assessment in Intelligent Transportation Systems**

This repository contains a full PyTorch reference implementation of
**RA-FogNet**, together with every baseline, ablation variant, robustness
test, and the SHAP-based explainability module reported in the manuscript.
It is intended to accompany the paper submission so that reviewers and
readers can inspect, run, and extend the exact method described in the
text.

## What's implemented

| Manuscript section | Code |
|---|---|
| §3.3 Multimodal sensor representation (Eq. 5-7) | `ra_fognet/modules.py :: MultiModalEncoder` |
| §3.4 Local uncertainty & reliability (Eq. 8-12) | `ra_fognet/modules.py :: ReliabilityEstimator` |
| §3.5 Reliability-aware attention fusion (Eq. 13-18) | `ra_fognet/modules.py :: ReliabilityAttentionFusion` |
| §3.6 Sensor-interaction graph learning (Eq. 19-24) | `ra_fognet/modules.py :: SensorGraphLayer` |
| §3.7-3.8 Temporal Transformer + classifier (Eq. 25-39) | `ra_fognet/modules.py :: TemporalTransformerEncoder`, `ra_fognet/model.py` |
| §3.9 SHAP explainability (Eq. 40-44) | `ra_fognet/explainability.py` |
| §3.12 Training objective (Eq. 52-56) | `ra_fognet/losses.py` |
| Table 6/7 baselines (CNN, CNN-LSTM, GRU, BiLSTM, TCN, Transformer, GCN, GAT, Graph Transformer, Attention Fusion Network) | `ra_fognet/baselines.py` |
| Table 9 ablation study | `ra_fognet/ablation.py` |
| Table 12 robustness study (Gaussian noise + missing sensor) | `ra_fognet/robustness.py` |
| §4.1 Session-level train/val/test protocol | `ra_fognet/data.py :: session_level_split` |

A single `RAFogNet` class (in `ra_fognet/model.py`) implements the full
model and exposes `use_reliability` / `use_graph` / `use_transformer`
boolean flags so that every row of the ablation table (Table 9) is produced
by the *same* implementation with components switched on or off, rather
than by separate hand-written variants.

## Repository layout

```
ra-fognet/
├── ra_fognet/
│   ├── data.py            # synthetic data generator, CSV loader, windowing, session split
│   ├── modules.py         # encoders, reliability estimator, attention fusion, graph layer, Transformer
│   ├── model.py            # full RA-FogNet assembly + ablation flags
│   ├── baselines.py        # CNN / CNN-LSTM / GRU / BiLSTM / TCN / Transformer / GCN / GAT / Graph-Transformer / Attention-Fusion
│   ├── losses.py            # classification + visibility + attention-entropy loss (Eq. 52-56)
│   ├── train.py             # shared training / evaluation loop
│   ├── robustness.py        # Gaussian-noise and missing-sensor degradation tests
│   ├── ablation.py           # ablation-suite runner (Table 9)
│   ├── explainability.py     # SHAP-based global/local modality attribution
│   └── utils.py               # seeding, metrics, early stopping
├── data/
│   └── ra_fognet_dataset.csv  # prepared dataset (45,600 rows / 24 sessions) — see "Data" below
├── scripts/
│   ├── generate_dataset.py   # prepare + validate the persisted dataset CSV (run this first)
│   ├── run_train.py          # train RA-FogNet or any baseline, report test metrics
│   ├── run_robustness.py     # reproduce Table 12
│   ├── run_ablation.py       # reproduce Table 9
│   └── run_explain.py        # reproduce Fig. 2 / Fig. 3 SHAP analysis
├── requirements.txt
└── README.md
```

`scripts/generate_dataset.py` and the data-generation/loading/splitting
functions in `ra_fognet/data.py` only require `numpy` and `pandas` — they
do **not** require PyTorch — so the dataset can be prepared or inspected
in a minimal environment before installing the full training stack.

## Installation

```bash
git clone <this-repo-url>
cd ra-fognet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11, PyTorch 2.4.1 (see Table 2 of the manuscript for
the original hardware/software configuration: NVIDIA RTX 4090, Intel Core
i9-13900K, 64 GB RAM, Windows 11).

## Data

The manuscript's dataset was collected from a physical sensing rig
(Texas Instruments AWR1843 radar, Campbell Scientific CS125 visibility
sensor, MQ-7/MQ-135 gas sensors, DHT22 humidity sensor, OBD-II vehicle
interface, u-blox NEO-M8N GNSS) and is not redistributed in this
repository.

### Step 1 — Prepare the dataset (run this once)

To let reviewers run the **entire pipeline end-to-end** without the raw
field logs, `scripts/generate_dataset.py` builds and validates a
**persisted** CSV dataset that matches the statistics reported in the
manuscript exactly:

```bash
python scripts/generate_dataset.py --out data/ra_fognet_dataset.csv \
    --n_sessions 24 --seconds_per_session 1900 --seed 42
```

This produces `data/ra_fognet_dataset.csv` (already included in this
repository, ~12.6 MB) with:

* **45,600** synchronized 1 Hz observations across **24 driving sessions**
  (`session_id`, `t`, `timestamp` spanning Nov 2025 – Jan 2026, as in
  Section 4.1),
* the exact reported class composition: **18,240 Clear / 13,680 Mild /
  9,120 Moderate / 4,560 Severe** (40/30/20/10 %), derived by construction
  via a class-quota dwell-segment schedule rather than by chance,
* all sensing-modality columns (`radar_0..radar_3`, `optical_0`, `gas_0`,
  `gas_1`, `humidity_0`, `vehicle_0`, `geo_0..geo_2`) generated from a
  shared, physically-motivated fog-intensity latent so that cross-modal
  correlations are realistic (e.g. the optical and humidity channels are
  strongly anti-correlated with visibility; vehicle speed is strongly
  correlated with it — consistent with drivers slowing down in fog),
* a `label` column derived **from** `visibility` via the exact thresholds
  in Eq. `eq:fog_severity_class` (never assigned independently), so
  visibility and label are always mutually consistent.

The script also runs a validation pass (column completeness, no NaNs,
reported class distribution) and prints a summary before saving.

**Results obtained on this synthetic dataset are for pipeline validation
only and will not reproduce the exact numbers in the manuscript's tables.**
Swap in your own real sensor logs (see "Using real data" below) to get
numbers that reflect your physical rig.

### Step 2 — Point the code at it

All four scripts in `scripts/` default to reading
`data/ra_fognet_dataset.csv`. If that file exists, it is loaded directly;
if it doesn't (e.g. you skipped Step 1), each script prints a warning and
falls back to generating a fresh synthetic dataset on the fly for that run
only, so nothing ever hard-fails. Pass `--data_csv path/to/file.csv` to
point at a different prepared dataset or your own real logs.

### Using real data

Provide a CSV with columns:

```
session_id, t, visibility,
radar_0..radar_3, optical_0, gas_0, gas_1, humidity_0, vehicle_0, geo_0..geo_2
```

and pass `--data_csv path/to/your_log.csv` to any script in `scripts/`.
The `label` column is derived automatically from `visibility` if not
already present.

## Usage

Train RA-FogNet on the prepared dataset:
```bash
python scripts/run_train.py --model ra_fognet --epochs 100
```

Train a baseline for comparison (Table 7):
```bash
python scripts/run_train.py --model graph_transformer --epochs 100
# other --model options: cnn, cnn_lstm, gru, bilstm, tcn, transformer,
#                        gcn, gat, graph_transformer, attention_fusion
```

Run the ablation study (Table 9):
```bash
python scripts/run_ablation.py --epochs 100
```

Run the robustness / sensor-degradation study (Table 12):
```bash
python scripts/run_robustness.py --epochs 100
```

Run the SHAP-based explainability analysis (Fig. 2 / Fig. 3):
```bash
python scripts/run_explain.py --epochs 100
```

All scripts write their metrics to `outputs/*.json`.

## Reproducibility

* Random seeds `{42, 52, 62, 72, 82}` are used for the repeated-run
  statistical analysis (Table 3/4/5); pass `--seed` to any script.
* Session-level partitioning (17 train / 3 val / 4 test sessions) is
  enforced in `session_level_split` so that no temporal window crosses a
  train/val/test boundary.
* Optimizer/training hyperparameters mirror Table 2 of the manuscript
  (Adam, lr = 1e-4, batch size = 64, weight decay = 1e-5, dropout = 0.2,
  max 100 epochs, early-stopping patience = 15).

## Citation

If you use this code, please cite the accompanying manuscript:

```bibtex
@article{rafognet2026,
  title   = {Explainable Reliability-Aware Multimodal Deep Sensor Fusion
             for Predictive Fog Risk Assessment in Intelligent
             Transportation Systems},
  author  = {Palani, S and Babu, Nirmal Varghese and Jemima, Darling D
             and Arumugam, Sajeev Ram},
  journal = {Journal of XYZ},
  year    = {2026}
}
```

## License

Add your preferred license (e.g., MIT or Apache-2.0) before publishing the
repository publicly.
