#!/usr/bin/env python3
"""
Per-object frame-level analysis: GT IOU (from pred/GT masks) + model scores (from scores.csv).
Usage:
  python tools/lvos_obj_analysis.py \
    --pred_root outputs/lvos_v1/lvos1_debug \
    --gt_root LVOS1/Annotations \
    --out_root outputs/lvos_v1/lvos1_debug/lvos1_analysis/obj_analysis
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


# J-Mean < 0.7 sequences: (video, obj_id)
LOW_JF_TARGETS = [
    ("49TNsJzk", 1),
    ("9mBuSvT2", 1),
    ("D4AgqLQL", 1),
    ("EWCZAcdt", 2),
    ("HNrCxhwd", 1),
    ("K3OUeINk", 1),
    ("N6CONZUW", 1),
    ("ScFTYisJ", 1),
    ("aFytsETk", 1),
    ("cUD1dwuP", 2),
    ("xpI7xRWN", 1),
]


def load_obj_mask(path: Path, obj_id: int) -> np.ndarray:
    arr = np.array(Image.open(path))
    return arr == obj_id


def frame_stats(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = float(inter / union) if union > 0 else 1.0

    tp = int(inter)
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(~pred_mask, gt_mask).sum())
    tn = int(np.logical_and(~pred_mask, ~gt_mask).sum())
    pred_area = int(pred_mask.sum())
    gt_area = int(gt_mask.sum())
    area_ratio = float(pred_area / gt_area) if gt_area > 0 else float("nan")

    return dict(gt_iou=iou, tp=tp, fp=fp, fn=fn, tn=tn,
                pred_area=pred_area, gt_area=gt_area,
                area_ratio_pred_over_gt=area_ratio)


def load_scores_csv(path: Path, obj_id: int) -> dict[str, dict]:
    result = {}
    if not path.exists():
        return result
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["obj_id"] == str(obj_id):
                result[row["frame"]] = row
    return result


def analyze(pred_root: Path, gt_root: Path, seq: str, obj_id: int, out_root: Path) -> None:
    pred_seq = pred_root / seq
    gt_seq = gt_root / seq
    scores_csv = pred_root / seq / "scores.csv"

    if not pred_seq.exists():
        print(f"[SKIP] pred not found: {pred_seq}")
        return
    if not gt_seq.exists():
        print(f"[SKIP] gt not found: {gt_seq}")
        return

    scores = load_scores_csv(scores_csv, obj_id)

    pred_files = {p.name for p in pred_seq.glob("*.png")}
    gt_files = {p.name for p in gt_seq.glob("*.png")}
    common = sorted(pred_files & gt_files)

    rows = []
    for frame in common:
        pred_mask = load_obj_mask(pred_seq / frame, obj_id)
        gt_mask = load_obj_mask(gt_seq / frame, obj_id)
        stats = frame_stats(pred_mask, gt_mask)

        frame_key = frame.replace(".png", "")
        sc = scores.get(frame_key, {})
        row = {
            "frame": frame,
            "gt_iou": f"{stats['gt_iou']:.6f}",
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "tn": stats["tn"],
            "pred_area": stats["pred_area"],
            "gt_area": stats["gt_area"],
            "area_ratio_pred_over_gt": f"{stats['area_ratio_pred_over_gt']:.6f}",
            "object_score_logit": sc.get("object_score_logit", ""),
            "iou_score_max": sc.get("iou_score_max", ""),
            "iou_score_0": sc.get("iou_score_0", ""),
            "iou_score_1": sc.get("iou_score_1", ""),
            "iou_score_2": sc.get("iou_score_2", ""),
            "blur_laplacian": sc.get("blur_laplacian", ""),
            "blur_fft_energy": sc.get("blur_fft_energy", ""),
            "blur_fft_angle": sc.get("blur_fft_angle", ""),
        }
        rows.append(row)

    if not rows:
        print(f"[SKIP] no common frames: {seq} obj{obj_id}")
        return

    out_root.mkdir(parents=True, exist_ok=True)
    out_file = out_root / f"{seq}_{obj_id}_analysis.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_iou = float(np.mean([float(r["gt_iou"]) for r in rows]))
    zero_iou = sum(1 for r in rows if float(r["gt_iou"]) == 0.0)
    print(f"{seq}_obj{obj_id}: frames={len(rows)} mean_gt_iou={mean_iou:.4f} zero_iou={zero_iou} -> {out_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_root", required=True, type=Path)
    parser.add_argument("--gt_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    args = parser.parse_args()

    for seq, obj_id in LOW_JF_TARGETS:
        analyze(args.pred_root, args.gt_root, seq, obj_id, args.out_root)


if __name__ == "__main__":
    main()
