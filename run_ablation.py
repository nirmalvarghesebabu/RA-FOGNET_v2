#!/usr/bin/env python
"""Run the full ablation study reproducing Table 9."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ra_fognet.data import (
    generate_synthetic_dataset, load_csv, load_dataset, normalize_dataframe,
    session_level_split, FogSequenceDataset, MODALITY_DIMS,
)
from ra_fognet.ablation import run_ablation_suite
from ra_fognet.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="data/ra_fognet_dataset.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/ablation.json")
    args = ap.parse_args()

    set_seed(args.seed)
    df = load_dataset(args.data_csv, seed=args.seed)
    train_df, val_df, test_df = session_level_split(df, seed=args.seed)
    train_df, stats = normalize_dataframe(train_df)
    val_df, _ = normalize_dataframe(val_df, stats)
    test_df, _ = normalize_dataframe(test_df, stats)

    train_ds = FogSequenceDataset(train_df, seq_len=30, stride=1)
    val_ds = FogSequenceDataset(val_df, seq_len=30, stride=30)
    test_ds = FogSequenceDataset(test_df, seq_len=30, stride=30)

    results = run_ablation_suite(
        MODALITY_DIMS, train_ds, val_ds, test_ds,
        train_config=dict(max_epochs=args.epochs),
    )
    print(json.dumps(results, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
