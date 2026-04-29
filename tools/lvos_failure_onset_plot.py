#!/usr/bin/env python3
"""
Failure onset analysis: plot average signal trajectories around gt_iou drop events.

Failure onset = first frame where gt_iou drops below threshold after being above it
                for at least min_ok_run consecutive frames.

Usage:
  python tools/lvos_failure_onset_plot.py \
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

SIGNALS = [
    "object_score_logit",
    "iou_score_max",
    "iou_score_range",
    "cand_iou_mean",
    "blur_laplacian",
    "blur_fft_energy",
]


def load_data(scores_root: Path) -> dict[str, pd.DataFrame]:
    """Load per-sequence DataFrames sorted by frame."""
    result = {}
    for seq, obj_id in LOW_JF_TARGETS:
        csv_path = scores_root / seq / "scores.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df[df["obj_id"] == obj_id].copy()
        num_cols = SIGNALS + ["gt_iou"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
        df = df.sort_values("frame").reset_index(drop=True)
        result[seq] = df
    return result


def find_failure_onsets(gt_iou: pd.Series, fail_thr: float, min_ok_run: int) -> list[int]:
    """Return indices where gt_iou drops below fail_thr after min_ok_run good frames."""
    onsets = []
    ok_run = 0
    for i, val in enumerate(gt_iou):
        if np.isnan(val):
            ok_run = 0
            continue
        if val >= fail_thr:
            ok_run += 1
        else:
            if ok_run >= min_ok_run:
                onsets.append(i)
            ok_run = 0
    return onsets


def extract_windows(seq_data: dict[str, pd.DataFrame],
                    win_before: int, win_after: int,
                    fail_thr: float, min_ok_run: int) -> list[pd.DataFrame]:
    """Extract signal windows centered on each failure onset."""
    windows = []
    for seq, df in seq_data.items():
        onsets = find_failure_onsets(df["gt_iou"], fail_thr, min_ok_run)
        for onset_idx in onsets:
            start = onset_idx - win_before
            end = onset_idx + win_after + 1
            if start < 0 or end > len(df):
                continue
            win = df.iloc[start:end][SIGNALS + ["gt_iou"]].copy()
            win["t"] = np.arange(-win_before, win_after + 1)
            windows.append(win)
    return windows


def plot_onset_windows(windows: list[pd.DataFrame], out_dir: Path,
                       win_before: int, win_after: int) -> None:
    if not windows:
        print("No failure onset windows found.")
        return

    combined = pd.concat(windows, ignore_index=True)
    grouped = combined.groupby("t")

    t_vals = np.arange(-win_before, win_after + 1)
    means = grouped.mean(numeric_only=True)
    sems = grouped.sem(numeric_only=True)
    n = len(windows)

    fig, axes = plt.subplots(len(SIGNALS) + 1, 1, figsize=(12, 3 * (len(SIGNALS) + 1)), sharex=True)

    # gt_iou (ground truth reference)
    ax = axes[0]
    m = means["gt_iou"].reindex(t_vals)
    s = sems["gt_iou"].reindex(t_vals)
    ax.plot(t_vals, m, color="black", linewidth=2, label="gt_iou (mean)")
    ax.fill_between(t_vals, m - s, m + s, alpha=0.2, color="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="failure onset")
    ax.set_ylabel("gt_iou")
    ax.set_title(f"Signal trajectories around failure onset (n={n} events, thr=0.5)")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)

    colors = ["steelblue", "darkorange", "green", "purple", "brown", "teal"]
    for i, (sig, color) in enumerate(zip(SIGNALS, colors)):
        ax = axes[i + 1]
        if sig not in means.columns:
            ax.set_ylabel(sig)
            continue
        m = means[sig].reindex(t_vals)
        s = sems[sig].reindex(t_vals)
        ax.plot(t_vals, m, color=color, linewidth=1.5)
        ax.fill_between(t_vals, m - s, m + s, alpha=0.2, color=color)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_ylabel(sig, fontsize=8)

    axes[-1].set_xlabel("frames relative to failure onset (0 = first fail frame)")
    plt.tight_layout()
    out_path = out_dir / "failure_onset_signals.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}  ({n} onset events)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--win_before", type=int, default=50)
    parser.add_argument("--win_after", type=int, default=30)
    parser.add_argument("--fail_thr", type=float, default=0.5)
    parser.add_argument("--min_ok_run", type=int, default=10,
                        help="Min consecutive OK frames before counting as onset")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seq_data = load_data(args.scores_root)
    windows = extract_windows(seq_data, args.win_before, args.win_after,
                              args.fail_thr, args.min_ok_run)
    print(f"Found {len(windows)} failure onset events across {len(seq_data)} sequences")
    plot_onset_windows(windows, args.out_dir, args.win_before, args.win_after)


if __name__ == "__main__":
    main()
