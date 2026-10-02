#!/usr/bin/env python
"""Run post-hoc SHAP-based explainability (global + local), Section 4.9."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ra_fognet.data import (
    generate_synthetic_dataset, load_csv, load_dataset, normalize_dataframe,
    session_level_split, FogSequenceDataset, MODALITY_DIMS, collate_fn,
)
from ra_fognet.model import RAFogNet
from ra_fognet.train import train_model
from ra_fognet.explainability import global_modality_importance, local_explanation
from ra_fognet.utils import set_seed, get_device
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default="data/ra_fognet_dataset.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_instances", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/explanations.json")
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

    model = RAFogNet(modality_dims=MODALITY_DIMS)
    train_model(model, train_ds, val_ds, model_kind="ra_fognet", config=dict(max_epochs=args.epochs))

    device = get_device()
    model.to(device)

    print("Computing global modality importance (SHAP)...")
    global_importance = global_modality_importance(model, test_ds, device, n_instances=args.n_instances)
    print(json.dumps(global_importance, indent=2))

    loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    modality_batch, label, _ = next(iter(loader))
    local_phi = local_explanation(model, modality_batch, int(label.item()), device)
    print("Local explanation for one test instance:")
    print(json.dumps(local_phi, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"global_importance": global_importance, "local_example": local_phi}, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
