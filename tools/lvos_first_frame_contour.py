"""
LVOS 첫 annotation 프레임에 GT mask contour를 그려서 저장.

사용법:
  python tools/lvos_first_frame_contour.py \
      --video_dir /data/LVOS/JPEGImages \
      --ann_dir   /data/LVOS/Annotations \
      --output_dir ./output/lvos_contours
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# object id별 색상 (BGR)
COLORS = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
]


def load_ann_png(ann_path):
    ann = np.array(Image.open(ann_path).convert("P"))
    obj_ids = np.unique(ann)
    obj_ids = obj_ids[obj_ids != 0]
    return {int(obj_id): (ann == obj_id) for obj_id in obj_ids}


def get_sorted_frame_names(video_path):
    frames = sorted(f for f in os.listdir(video_path) if f.endswith(".jpg") or f.endswith(".png"))
    return frames


def get_sorted_ann_names(ann_dir):
    if not os.path.isdir(ann_dir):
        return []
    return sorted(f for f in os.listdir(ann_dir) if f.endswith(".png"))


def draw_contours_on_image(img_bgr, mask_dict):
    out = img_bgr.copy()
    for i, (obj_id, mask) in enumerate(sorted(mask_dict.items())):
        color = COLORS[i % len(COLORS)]
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)
    return out


def process_video(video_name, video_dir, ann_dir, output_dir):
    video_path = os.path.join(video_dir, video_name)
    ann_video_dir = os.path.join(ann_dir, video_name)

    frame_names = get_sorted_frame_names(video_path)
    ann_names = get_sorted_ann_names(ann_video_dir)

    if not frame_names or not ann_names:
        return

    stem_to_idx = {os.path.splitext(n)[0]: i for i, n in enumerate(frame_names)}

    # 첫 번째 annotation 파일 찾기
    first_ann = None
    for ann_name in ann_names:
        ann_stem = os.path.splitext(ann_name)[0]
        if ann_stem in stem_to_idx:
            first_ann = ann_name
            break

    if first_ann is None:
        return

    ann_stem = os.path.splitext(first_ann)[0]
    frame_idx = stem_to_idx[ann_stem]
    img_path = os.path.join(video_path, frame_names[frame_idx])
    ann_path = os.path.join(ann_video_dir, first_ann)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return

    mask_dict = load_ann_png(ann_path)
    if not mask_dict:
        return

    out_video_dir = os.path.join(output_dir, video_name)
    os.makedirs(out_video_dir, exist_ok=True)

    # 객체별 개별 이미지
    for obj_id, mask in sorted(mask_dict.items()):
        out_obj = draw_contours_on_image(img_bgr, {obj_id: mask})
        cv2.imwrite(os.path.join(out_video_dir, f"{ann_stem}_obj{obj_id:03d}_contour.jpg"), out_obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    video_names = sorted(
        v for v in os.listdir(args.ann_dir)
        if os.path.isdir(os.path.join(args.ann_dir, v))
    )

    for video_name in tqdm(video_names, desc="Processing videos"):
        process_video(video_name, args.video_dir, args.ann_dir, args.output_dir)

    print(f"\nDone. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
