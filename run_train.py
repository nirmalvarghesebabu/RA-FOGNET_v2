#!/usr/bin/env python
"""
Train RA-FogNet (or a baseline) end-to-end: data loading/generation,
session-level split, sequence windowing, training, and test-set evaluation.

Examples
--------
Synthetic data, RA-FogNet:
    python scripts/run_train.py --model ra_fognet --epochs 100

Real data, CNN-LSTM baseline:
    python scripts/run_train.py --model cnn_lstm --data_csv path/to/log.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ra_fognet.data import (
    generate_synthetic_dataset, load_csv, load_dataset, normalize_dataframe,
    session_level_split, FogSequenceDataset, MODALITY_DIMS,
)
from ra_fognet.model import RAFogNet
from ra_fognet.baselines import BASELINE_REGISTRY, AttentionFusionNetwork
from ra_fognet.train import train_model, evaluate_dataset
from ra_fognet.utils import set_seed


def build_model(name: str, modality_dims):
    if name == "ra_fognet":
        return RAFogNet(modality_dims=modality_dims), "ra_fognet"
    if name == "attention_fusion":
        return AttentionFusionNetwork(modality_dims=modality_dims), "ra_fognet"  # dict-input model
    if name in BASELINE_REGISTRY:
        in_dim = sum(modality_dims.values())
        return BASELINE_REGISTRY[name](in_dim=in_dim), "baseline"
    raise ValueError(f"Unknown model '{name}'. Options: ra_fognet, attention_fusion, "
                      f"{list(BASELINE_REGISTRY.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ra_fognet")
    ap.add_argument("--data_csv", default="data/ra_fognet_dataset.csv",
                     help="Path to the prepared dataset CSV (run scripts/generate_dataset.py first, "
                          "or point this at your own real sensor log). Pass --data_csv '' to fall back "
                          "to on-the-fly synthetic generation instead.")
    ap.add_argument("--seq_len", type=int, default=30)
    ap.add_argument("--train_stride", type=int, default=1)
    ap.add_argument("--eval_stride", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/results.json")
    args = ap.parse_args()

    set_seed(args.seed)

    df = load_dataset(args.data_csv, seed=args.seed)
    train_df, val_df, test_df = session_level_split(df, seed=args.seed)
    train_df, stats = normalize_dataframe(train_df)
    val_df, _ = normalize_dataframe(val_df, stats)
    test_df, _ = normalize_dataframe(test_df, stats)

    train_ds = FogSequenceDataset(train_df, seq_len=args.seq_len, stride=args.train_stride)
    val_ds = FogSequenceDataset(val_df, seq_len=args.seq_len, stride=args.eval_stride)
    test_ds = FogSequenceDataset(test_df, seq_len=args.seq_len, stride=args.eval_stride)
    print(f"Train/Val/Test sequences: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    model, model_kind = build_model(args.model, MODALITY_DIMS)
    train_cfg = dict(lr=args.lr, batch_size=args.batch_size, max_epochs=args.epochs, patience=args.patience)
    train_model(model, train_ds, val_ds, model_kind=model_kind, config=train_cfg)

    test_metrics = evaluate_dataset(model, test_ds, model_kind=model_kind, batch_size=args.batch_size)
    print("Test metrics:", json.dumps(test_metrics, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "test_metrics": test_metrics}, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
