#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class FrameStats:
    frame: str
    iou: float
    tp: int
    fp: int
    fn: int
    tn: int
    pred_area: int
    gt_area: int
    area_ratio_pred_over_gt: float


def load_mask_bool(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    return arr > 0


def make_overlay(base_img: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, alpha: float) -> np.ndarray:
    img = base_img.astype(np.float32)

    pred_only = np.logical_and(pred_mask, ~gt_mask)
    gt_only = np.logical_and(gt_mask, ~pred_mask)
    overlap = np.logical_and(pred_mask, gt_mask)

    red = np.array([255, 0, 0], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    yellow = np.array([255, 255, 0], dtype=np.float32)

    out = img.copy()
    for mask, color in ((pred_only, red), (gt_only, green), (overlap, yellow)):
        if np.any(mask):
            out[mask] = (1.0 - alpha) * out[mask] + alpha * color

    return np.clip(out, 0, 255).astype(np.uint8)


def frame_stats(pred_mask: np.ndarray, gt_mask: np.ndarray, frame: str) -> FrameStats:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = float(inter / union) if union > 0 else 1.0

    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    tp = int(inter)
    tn = int(np.logical_and(~pred_mask, ~gt_mask).sum())

    pred_area = int(pred_mask.sum())
    gt_area = int(gt_mask.sum())
    area_ratio = float(pred_area / gt_area) if gt_area > 0 else float("nan")

    return FrameStats(
        frame=frame,
        iou=iou,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        pred_area=pred_area,
        gt_area=gt_area,
        area_ratio_pred_over_gt=area_ratio,
    )


def write_frame_csv(path: Path, rows: Iterable[FrameStats]) -> None:
    rows = list(rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "iou",
                "tp",
                "fp",
                "fn",
                "tn",
                "pred_area",
                "gt_area",
                "area_ratio_pred_over_gt",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "frame": r.frame,
                    "iou": f"{r.iou:.6f}",
                    "tp": r.tp,
                    "fp": r.fp,
                    "fn": r.fn,
                    "tn": r.tn,
                    "pred_area": r.pred_area,
                    "gt_area": r.gt_area,
                    "area_ratio_pred_over_gt": f"{r.area_ratio_pred_over_gt:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch overlay and framewise metrics for LVOS predictions.")
    parser.add_argument("--pred_root", required=True, type=Path)
    parser.add_argument("--gt_root", required=True, type=Path)
    parser.add_argument("--img_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--limit_sequences", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--sequence_list_file",
        type=Path,
        default=None,
        help="Optional text file with one sequence name per line. If provided, only these sequences are analyzed.",
    )
    args = parser.parse_args()

    pred_root = args.pred_root
    gt_root = args.gt_root
    img_root = args.img_root
    out_root = args.out_root

    overlay_root = out_root / "overlays"
    metrics_root = out_root / "metrics_per_sequence"
    overlay_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    sequences = sorted([p.name for p in pred_root.iterdir() if p.is_dir()])
    if args.sequence_list_file is not None:
        requested = {
            line.strip()
            for line in args.sequence_list_file.read_text().splitlines()
            if line.strip()
        }
        sequences = [seq for seq in sequences if seq in requested]
    if args.limit_sequences > 0:
        sequences = sequences[: args.limit_sequences]

    summary_rows = []

    for idx, seq in enumerate(sequences, 1):
        pred_seq = pred_root / seq
        gt_seq = gt_root / seq
        img_seq = img_root / seq

        if not gt_seq.exists() or not img_seq.exists():
            summary_rows.append(
                {
                    "sequence": seq,
                    "frames_common": 0,
                    "mean_iou": "nan",
                    "zero_iou_frames": 0,
                    "pred_only": len(list(pred_seq.glob("*.png"))),
                    "gt_only": 0,
                    "status": "missing_gt_or_img",
                }
            )
            continue

        pred_files = sorted(p.name for p in pred_seq.glob("*.png"))
        gt_files = sorted(p.name for p in gt_seq.glob("*.png"))
        common = sorted(set(pred_files) & set(gt_files))
        pred_only = len(set(pred_files) - set(gt_files))
        gt_only = len(set(gt_files) - set(pred_files))

        seq_overlay_dir = overlay_root / seq
        seq_overlay_dir.mkdir(parents=True, exist_ok=True)

        rows: list[FrameStats] = []
        for frame in common:
            pred_mask = load_mask_bool(pred_seq / frame)
            gt_mask = load_mask_bool(gt_seq / frame)

            row = frame_stats(pred_mask, gt_mask, frame)
            rows.append(row)

            # input images are jpg in LVOS1
            jpg_name = frame.replace(".png", ".jpg")
            img_path = img_seq / jpg_name
            if img_path.exists():
                base_img = np.array(Image.open(img_path).convert("RGB"))
                ov = make_overlay(base_img, pred_mask, gt_mask, args.alpha)
                Image.fromarray(ov).save(seq_overlay_dir / frame)

        metrics_csv = metrics_root / f"{seq}_framewise_analysis.csv"
        if rows:
            write_frame_csv(metrics_csv, rows)
            mean_iou = float(np.mean([r.iou for r in rows]))
            zero_iou = int(sum(1 for r in rows if r.iou == 0.0))
            status = "ok"
        else:
            mean_iou = float("nan")
            zero_iou = 0
            status = "no_common_frames"

        summary_rows.append(
            {
                "sequence": seq,
                "frames_common": len(common),
                "mean_iou": f"{mean_iou:.6f}" if rows else "nan",
                "zero_iou_frames": zero_iou,
                "pred_only": pred_only,
                "gt_only": gt_only,
                "status": status,
            }
        )

        print(f"[{idx}/{len(sequences)}] {seq}: frames={len(common)} mean_iou={mean_iou:.4f} zero_iou={zero_iou}")

    summary_csv = out_root / "summary_metrics.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sequence",
                "frames_common",
                "mean_iou",
                "zero_iou_frames",
                "pred_only",
                "gt_only",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"done: summary -> {summary_csv}")
    print(f"done: overlays -> {overlay_root}")
    print(f"done: per-seq metrics -> {metrics_root}")


if __name__ == "__main__":
    main()
