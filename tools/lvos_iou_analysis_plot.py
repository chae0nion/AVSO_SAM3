#!/usr/bin/env python3
"""
두 가지 질문 분석:
1. iou_score_std 클 때 gt_iou도 낮은가?
2. iou_score 편차 클 때 후보 마스크 위치(cand_iou_mean)가 얼마나 다른가?

Usage:
  python tools/lvos_iou_analysis_plot.py \
    --scores_root outputs/lvos_v1/low_jf_rerun \
    --out_dir outputs/lvos_v1/low_jf_rerun/analysis_plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOW_JF_TARGETS = [
    ("49TNsJzk", 1), ("9mBuSvT2", 1), ("D4AgqLQL", 1), ("EWCZAcdt", 2),
    ("HNrCxhwd", 1), ("K3OUeINk", 1), ("N6CONZUW", 1), ("ScFTYisJ", 1),
    ("aFytsETk", 1), ("cUD1dwuP", 2), ("xpI7xRWN", 1),
]


def load_target_rows(scores_root: Path) -> pd.DataFrame:
    dfs = []
    for seq, obj_id in LOW_JF_TARGETS:
        csv_path = scores_root / seq / "scores.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[df["obj_id"] == obj_id].copy()
        df["seq"] = seq
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No scores.csv found")
    return pd.concat(dfs, ignore_index=True)


def plot_range_vs_gt_iou(df: pd.DataFrame, out_dir: Path) -> None:
    """Q1: iou_score_range (max-min) 클 때 gt_iou가 낮은가?"""
    d = df[["iou_score_range", "gt_iou", "seq"]].dropna()
    d = d.astype({"iou_score_range": float, "gt_iou": float})

    corr = d["iou_score_range"].corr(d["gt_iou"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(d["iou_score_range"], d["gt_iou"], alpha=0.15, s=5, color="steelblue")
    ax.set_xlabel("iou_score_range (max - min of 3 candidate IOU scores)")
    ax.set_ylabel("gt_iou")
    ax.set_title(f"iou_score_range vs gt_iou  (r={corr:.3f})")
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, label="gt_iou=0.5")
    ax.legend(fontsize=8)

    ax = axes[1]
    bins = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    bin_labels = ["0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0"]
    d["gt_bin"] = pd.cut(d["gt_iou"], bins=bins, labels=bin_labels)
    groups = [d.loc[d["gt_bin"] == lb, "iou_score_range"].values for lb in bin_labels]
    ax.boxplot(groups, tick_labels=bin_labels, showfliers=False)
    ax.set_xlabel("gt_iou range")
    ax.set_ylabel("iou_score_range")
    ax.set_title("iou_score_range distribution per gt_iou bin")

    plt.tight_layout()
    out_path = out_dir / "q1_range_vs_gt_iou.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Q1] corr(iou_score_range, gt_iou)={corr:.3f} -> {out_path.name}")


def plot_std_vs_cand_position(df: pd.DataFrame, out_dir: Path) -> None:
    """Q2: iou_score 편차 클 때 후보 마스크 위치(cand_iou_mean)가 다른가?"""
    d = df[["iou_score_range", "cand_iou_mean", "gt_iou", "seq"]].dropna()
    d = d.astype({"iou_score_range": float, "cand_iou_mean": float, "gt_iou": float})

    corr = d["iou_score_range"].corr(d["cand_iou_mean"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sc = ax.scatter(d["iou_score_range"], d["cand_iou_mean"],
                    c=d["gt_iou"], cmap="RdYlGn", alpha=0.3, s=5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="gt_iou")
    ax.set_xlabel("iou_score_range (max - min of 3 candidates)")
    ax.set_ylabel("cand_iou_mean (positional overlap between candidates)")
    ax.set_title(f"iou_score_range vs cand_iou_mean  (r={corr:.3f})")

    ax = axes[1]
    bins = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
    bin_labels = ["0-0.05", "0.05-0.1", "0.1-0.2", "0.2-0.5", "0.5-1.0"]
    d["range_bin"] = pd.cut(d["iou_score_range"], bins=bins, labels=bin_labels)
    groups = [d.loc[d["range_bin"] == lb, "cand_iou_mean"].values for lb in bin_labels]
    ax.boxplot(groups, tick_labels=bin_labels, showfliers=False)
    ax.set_xlabel("iou_score_range bin")
    ax.set_ylabel("cand_iou_mean (low = different positions)")
    ax.set_title("cand_iou_mean distribution per iou_score_range bin")

    plt.tight_layout()
    out_path = out_dir / "q2_range_vs_cand_position.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Q2] corr(iou_score_range, cand_iou_mean)={corr:.3f} -> {out_path.name}")


def print_summary(df: pd.DataFrame) -> None:
    d = df[["iou_score_std", "iou_score_range", "cand_iou_mean", "gt_iou"]].dropna()
    d = d.astype(float)
    print("\n=== Correlation summary ===")
    for col in ["iou_score_std", "iou_score_range", "cand_iou_mean"]:
        r = d[col].corr(d["gt_iou"])
        print(f"  corr({col}, gt_iou) = {r:.4f}")
    r2 = d["iou_score_range"].corr(d["cand_iou_mean"])
    print(f"  corr(iou_score_range, cand_iou_mean) = {r2:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_target_rows(args.scores_root)
    print(f"Total {len(df)} rows loaded")

    plot_range_vs_gt_iou(df, args.out_dir)
    plot_std_vs_cand_position(df, args.out_dir)
    print_summary(df)


if __name__ == "__main__":
    main()
