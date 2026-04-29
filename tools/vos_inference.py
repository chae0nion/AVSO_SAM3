"""
SAM3 Semi-supervised Video Object Segmentation (VOS) Inference
SAM2의 vos_inference.py에 대응하는 SAM3용 스크립트.

동작 방식:
  - build_sam3_video_model()로 Detector + Tracker 로드
  - predictor.backbone = detector.backbone 으로 SAM3 ViT backbone 공유
  - Frame 0의 GT mask → add_new_mask() → propagate_in_video()
  - SAM2 vos_inference.py와 동일한 평가 방식

지원 데이터셋:
  - DAVIS: JPEGImages/480p/<video>/*.jpg  /  Annotations/480p/<video>/*.png
  - YouTube-VOS: JPEGImages/<video>/*.jpg  /  Annotations/<split>/<video>/*.png
  - LVOS: JPEGImages/<video>/*.jpg  /  Annotations/<video>/*.png
    (일부 object가 비디오 중간에 등장 → --track_object_appearing_later_in_video 필요)

사용법:
  python tools/vos_inference.py \
      --video_dir /data/DAVIS/JPEGImages/480p \
      --ann_dir   /data/DAVIS/Annotations/480p \
      --output_dir ./output/davis \
      --dataset davis

  python tools/vos_inference.py \
      --video_dir /data/ytvos/train/JPEGImages \
      --ann_dir   /data/ytvos/train/Annotations \
      --output_dir ./output/ytvos \
      --dataset ytvos

  python tools/vos_inference.py \
      --video_dir /data/LVOS/JPEGImages \
      --ann_dir   /data/LVOS/Annotations \
      --video_list /data/LVOS/ImageSets/val.txt \
      --output_dir ./output/lvos \
      --dataset lvos \
      --track_object_appearing_later_in_video
"""

import argparse
import csv
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_model


# ─────────────────────────────────────────────
# 블러 측정 유틸
# ─────────────────────────────────────────────

def compute_blur_laplacian(img_gray: np.ndarray) -> float:
    """Laplacian 분산 — 값이 낮을수록 블러."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def compute_blur_fft(img_gray: np.ndarray, size: int = 60) -> tuple[float, float]:
    """
    FFT 기반 블러 분석 — 고주파 에너지와 지배적 블러 방향 반환.

    Returns:
        fft_energy: 고주파 에너지 평균 (낮을수록 블러)
        fft_angle:  지배적 블러 방향 (degree, 모션 블러 분석용)
    """
    h, w = img_gray.shape
    cx, cy = w // 2, h // 2
    fft_shift = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float32)))

    # 저주파 마스킹 후 고주파 에너지
    mask = np.ones((h, w), dtype=np.float32)
    mask[cy - size:cy + size, cx - size:cx + size] = 0
    magnitude = np.log(np.abs(fft_shift) + 1) * mask
    fft_energy = float(magnitude.mean())

    # 지배적 방향: magnitude map에서 무게중심 방향
    y_idx, x_idx = np.where(magnitude > magnitude.mean())
    if len(x_idx) > 0:
        dx = float(x_idx.mean()) - cx
        dy = float(y_idx.mean()) - cy
        fft_angle = float(np.degrees(np.arctan2(dy, dx)))
    else:
        fft_angle = 0.0

    return fft_energy, fft_angle


def analyze_blur(img_path: str, fft_size: int = 60) -> tuple[float, float, float]:
    """
    Laplacian 분산 + FFT 에너지/방향을 모두 계산.

    Returns:
        laplacian:  float (낮을수록 블러)
        fft_energy: float (낮을수록 블러)
        fft_angle:  float (지배적 블러 방향, degree)
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    laplacian = compute_blur_laplacian(img)
    fft_energy, fft_angle = compute_blur_fft(img, size=fft_size)
    return laplacian, fft_energy, fft_angle


# ─────────────────────────────────────────────
# 데이터셋 유틸
# ─────────────────────────────────────────────

def load_ann_png(ann_path):
    """
    Annotation PNG를 읽어 object별 binary mask dict를 반환한다.

    DAVIS / YouTube-VOS는 모두 palette PNG 형식을 사용하며,
    각 픽셀값이 object id를 나타낸다 (0 = background).

    Args:
        ann_path: annotation PNG 파일 경로

    Returns:
        {obj_id (int): binary mask (H, W) bool} 형태의 dict.
        background(0)는 포함하지 않는다.
    """
    # palette 모드로 열면 픽셀값이 그대로 object id가 된다
    ann = np.array(Image.open(ann_path).convert("P"))

    obj_ids = np.unique(ann)
    obj_ids = obj_ids[obj_ids != 0]  # 0 = background 제외

    return {int(obj_id): (ann == obj_id) for obj_id in obj_ids}


def save_ann_png(mask_dict, video_height, video_width, save_path):
    """
    object별 binary mask dict를 하나의 palette PNG로 저장한다.

    여러 object mask를 하나의 array에 합친 뒤 DAVIS 표준 palette를 적용한다.
    object가 겹치는 경우 obj_id가 큰 것이 덮어쓴다.

    Args:
        mask_dict: {obj_id (int): binary mask (H, W) bool}
        video_height: 원본 비디오 높이
        video_width: 원본 비디오 너비
        save_path: 저장할 PNG 경로
    """
    # 모든 object mask를 하나의 array에 합침 (픽셀값 = obj_id)
    combined = np.zeros((video_height, video_width), dtype=np.uint8)
    for obj_id, mask in mask_dict.items():
        combined[mask] = obj_id

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = Image.fromarray(combined).convert("P")
    img.putpalette(_get_davis_palette())
    img.save(save_path)


def _get_davis_palette():
    """
    DAVIS 데이터셋 표준 color palette 생성 (256색).

    각 object id에 대해 시각적으로 구별되는 색상을 bit-interleaving으로 생성한다.
    평가 도구(davis-evaluation 등)가 이 palette를 기대하므로 맞춰줘야 한다.
    """
    palette = []
    for i in range(256):
        r = (i & 1) << 7 | (i & 8) << 3 | (i & 64) >> 1
        g = (i & 2) << 6 | (i & 16) << 2 | (i & 128) >> 2
        b = (i & 4) << 5 | (i & 32) << 1
        palette.extend([r, g, b])
    return palette


def get_sorted_frame_names(video_path):
    """
    비디오 폴더 내 JPEG 프레임 파일명을 숫자 순으로 정렬해 반환한다.

    파일명이 "00000.jpg" 같은 숫자 형식임을 전제한다.
    """
    frames = [
        f for f in os.listdir(video_path)
        if os.path.splitext(f)[-1].lower() in (".jpg", ".jpeg")
    ]
    frames.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return frames


def get_sorted_ann_names(ann_video_dir):
    """
    Annotation 폴더 내 PNG 파일명을 숫자 순으로 정렬해 반환한다.

    폴더가 존재하지 않으면 빈 리스트를 반환한다.
    """
    if not os.path.isdir(ann_video_dir):
        return []
    anns = [f for f in os.listdir(ann_video_dir) if f.endswith(".png")]
    anns.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return anns


# ─────────────────────────────────────────────
# 핵심 추론 함수
# ─────────────────────────────────────────────

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def run_vos_on_video(
    predictor,
    video_dir,
    ann_dir,
    video_name,
    output_dir,
    score_thresh=0.0,
    use_all_masks=False,
    offload_video_to_cpu=False,
):
    """
    단일 비디오에 대해 semi-supervised VOS 추론을 수행한다.
    모든 object가 첫 프레임에 등장하는 경우 (DAVIS 등) 사용.

    Args:
        predictor: Sam3TrackerPredictor 인스턴스
        video_dir: 비디오 JPEG 프레임 루트 경로
        ann_dir: GT annotation PNG 루트 경로
        video_name: 비디오 이름 (하위 폴더명)
        output_dir: 결과 저장 루트 경로
        score_thresh: mask logit threshold (default: 0.0)
        use_all_masks: True이면 모든 annotation 프레임 사용, False이면 첫 프레임만
        offload_video_to_cpu: True이면 프레임을 CPU에 올려 GPU 메모리 절약
    """
    video_path = os.path.join(video_dir, video_name)
    ann_video_dir = os.path.join(ann_dir, video_name)

    frame_names = get_sorted_frame_names(video_path)
    ann_names = get_sorted_ann_names(ann_video_dir)

    if not frame_names:
        print(f"  [skip] No JPEG frames in {video_path}")
        return
    if not ann_names:
        print(f"  [skip] No annotations in {ann_video_dir}")
        return

    stem_to_idx = {os.path.splitext(name)[0]: idx for idx, name in enumerate(frame_names)}

    inference_state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=False,
    )
    video_height = inference_state["video_height"]
    video_width = inference_state["video_width"]

    # 입력 mask 프레임 결정
    if not use_all_masks:
        # 첫 annotation 프레임만 사용
        input_ann_names = ann_names[:1]
    else:
        input_ann_names = ann_names

    first_ann_frame_idx = None
    for ann_name in input_ann_names:
        ann_stem = os.path.splitext(ann_name)[0]
        if ann_stem not in stem_to_idx:
            print(f"  [warn] annotation {ann_name} has no matching JPEG in {video_name}")
            continue

        frame_idx = stem_to_idx[ann_stem]
        if first_ann_frame_idx is None:
            first_ann_frame_idx = frame_idx

        obj_masks = load_ann_png(os.path.join(ann_video_dir, ann_name))
        for obj_id, mask in obj_masks.items():
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=torch.from_numpy(mask).float(),
            )

    if first_ann_frame_idx is None:
        print(f"  [skip] No valid annotation frames for {video_name}")
        return

    # 전체 비디오에 mask 전파
    video_segments = {}
    score_log = []
    blur_cache = {}  # {frame_idx: (laplacian, fft_energy, fft_angle)}
    for frame_idx, obj_ids, _, video_res_masks, obj_scores, iou_scores, iou_scores_all, mask_areas_all, cand_pairwise_iou, _multimasks in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=first_ann_frame_idx,
        max_frame_num_to_track=inference_state["num_frames"],
        reverse=False,
        propagate_preflight=True,
    ):
        video_segments[frame_idx] = {
            obj_id: (video_res_masks[i] > score_thresh).cpu().numpy().squeeze()
            for i, obj_id in enumerate(obj_ids)
        }
        frame_name = os.path.splitext(frame_names[frame_idx])[0]
        if frame_idx not in blur_cache:
            img_path = os.path.join(video_path, frame_names[frame_idx])
            blur_cache[frame_idx] = analyze_blur(img_path)
        laplacian, fft_energy, fft_angle = blur_cache[frame_idx]
        for i, obj_id in enumerate(obj_ids):
            obj_score = float(obj_scores[i].item()) if obj_scores is not None else None
            iou_score_max = float(iou_scores[i].item()) if iou_scores is not None else None
            iou_all = iou_scores_all[i].tolist() if iou_scores_all is not None else [None, None, None]
            areas = mask_areas_all[i].tolist() if mask_areas_all is not None else [None, None, None]
            piou = cand_pairwise_iou[i] if cand_pairwise_iou is not None else None
            ciou_01 = float(piou[0, 1].item()) if piou is not None else None
            ciou_02 = float(piou[0, 2].item()) if piou is not None else None
            ciou_12 = float(piou[1, 2].item()) if piou is not None else None
            score_log.append((frame_name, int(obj_id), obj_score, iou_score_max, *iou_all, *areas, ciou_01, ciou_02, ciou_12, laplacian, fft_energy, fft_angle))

    # 결과를 프레임별 palette PNG로 저장
    output_video_dir = os.path.join(output_dir, video_name)
    for frame_idx, frame_name in enumerate(frame_names):
        mask_dict = video_segments.get(frame_idx, {})
        stem = os.path.splitext(frame_name)[0]
        save_ann_png(mask_dict, video_height, video_width,
                     os.path.join(output_video_dir, f"{stem}.png"))

    # iou_score / object_score_logit / blur 로그 저장
    os.makedirs(output_video_dir, exist_ok=True)
    log_path = os.path.join(output_video_dir, "scores.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "obj_id", "object_score_logit", "iou_score_max", "iou_score_0", "iou_score_1", "iou_score_2", "mask_area_0", "mask_area_1", "mask_area_2", "cand_iou_01", "cand_iou_02", "cand_iou_12", "blur_laplacian", "blur_fft_energy", "blur_fft_angle"])
        writer.writerows(score_log)

    print(f"  [done] {video_name}: {len(video_segments)} frames → {output_video_dir}")


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    video_dir,
    ann_dir,
    video_name,
    output_dir,
    score_thresh=0.0,
    use_all_masks=False,
    offload_video_to_cpu=False,
):
    """
    object별로 분리 추론하는 VOS. LVOS나 YouTube-VOS처럼
    모든 object가 첫 프레임에 나타나지 않는 데이터셋용.

    각 object마다 독립적으로 propagate한 뒤,
    non-overlapping constraint로 최종 mask를 합산한다.
    """
    video_path = os.path.join(video_dir, video_name)
    ann_video_dir = os.path.join(ann_dir, video_name)

    frame_names = get_sorted_frame_names(video_path)
    ann_names = get_sorted_ann_names(ann_video_dir)

    if not frame_names:
        print(f"  [skip] No JPEG frames in {video_path}")
        return
    if not ann_names:
        print(f"  [skip] No annotations in {ann_video_dir}")
        return

    stem_to_idx = {os.path.splitext(name)[0]: idx for idx, name in enumerate(frame_names)}

    inference_state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=False,
    )
    video_height = inference_state["video_height"]
    video_width = inference_state["video_width"]

    # ── Step 1: 모든 annotation에서 object별 입력 mask 수집 ──
    inputs_per_object = defaultdict(dict)  # {obj_id: {frame_idx: mask_tensor}}
    for ann_name in ann_names:
        ann_stem = os.path.splitext(ann_name)[0]
        if ann_stem not in stem_to_idx:
            continue
        frame_idx = stem_to_idx[ann_stem]
        obj_masks = load_ann_png(os.path.join(ann_video_dir, ann_name))
        for obj_id, mask in obj_masks.items():
            if not np.any(mask):
                continue
            # use_all_masks=False이면 object당 첫 번째 등장 프레임만 사용
            if len(inputs_per_object[obj_id]) > 0 and not use_all_masks:
                continue
            inputs_per_object[obj_id][frame_idx] = torch.from_numpy(mask).float()

    if not inputs_per_object:
        print(f"  [skip] No valid annotation frames for {video_name}")
        return

    # ── Step 2: object별 독립 추론 ──
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)  # {obj_id: {frame_idx: scores}}
    iou_log_per_object = defaultdict(dict)    # {obj_id: {frame_idx: (obj_score, iou_score_max, iou_all)}}
    blur_cache = {}  # {frame_idx: (laplacian, fft_energy, fft_angle)} — 객체 간 공유

    for obj_id in object_ids:
        input_frame_inds = sorted(inputs_per_object[obj_id])
        predictor.clear_all_points_in_video(inference_state)

        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=obj_id,
                mask=inputs_per_object[obj_id][input_frame_idx],
            )

        for out_frame_idx, _, _, video_res_masks, obj_scores, iou_scores, iou_scores_all, mask_areas_all, cand_pairwise_iou, _multimasks in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            max_frame_num_to_track=inference_state["num_frames"],
            reverse=False,
            propagate_preflight=True,
        ):
            output_scores_per_object[obj_id][out_frame_idx] = (
                video_res_masks.cpu()
            )
            obj_score = float(obj_scores[0].item()) if obj_scores is not None else None
            iou_score_max = float(iou_scores[0].item()) if iou_scores is not None else None
            iou_all = iou_scores_all[0].tolist() if iou_scores_all is not None else [None, None, None]
            areas = mask_areas_all[0].tolist() if mask_areas_all is not None else [None, None, None]
            piou = cand_pairwise_iou[0] if cand_pairwise_iou is not None else None
            ciou_01 = float(piou[0, 1].item()) if piou is not None else None
            ciou_02 = float(piou[0, 2].item()) if piou is not None else None
            ciou_12 = float(piou[1, 2].item()) if piou is not None else None
            if out_frame_idx not in blur_cache:
                img_path = os.path.join(video_path, frame_names[out_frame_idx])
                blur_cache[out_frame_idx] = analyze_blur(img_path)
            iou_log_per_object[obj_id][out_frame_idx] = (obj_score, iou_score_max, iou_all, areas, ciou_01, ciou_02, ciou_12)

    # ── Step 3: per-object score를 합산 + non-overlapping constraint ──
    output_video_dir = os.path.join(output_dir, video_name)
    for frame_idx, frame_name in enumerate(frame_names):
        # 모든 object의 score를 하나의 tensor로 합침
        scores = torch.full(
            size=(len(object_ids), 1, video_height, video_width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, obj_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[obj_id]:
                scores[i] = output_scores_per_object[obj_id][frame_idx]

        # non-overlapping constraint: 겹치는 영역에서 가장 높은 score만 유지
        scores = predictor._apply_non_overlapping_constraints(scores)

        mask_dict = {
            obj_id: (scores[i] > score_thresh).numpy().squeeze()
            for i, obj_id in enumerate(object_ids)
        }

        stem = os.path.splitext(frame_name)[0]
        save_ann_png(mask_dict, video_height, video_width,
                     os.path.join(output_video_dir, f"{stem}.png"))

    # iou_score / object_score_logit / blur 로그 저장
    os.makedirs(output_video_dir, exist_ok=True)
    log_path = os.path.join(output_video_dir, "scores.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "obj_id", "object_score_logit", "iou_score_max", "iou_score_0", "iou_score_1", "iou_score_2", "mask_area_0", "mask_area_1", "mask_area_2", "cand_iou_01", "cand_iou_02", "cand_iou_12", "blur_laplacian", "blur_fft_energy", "blur_fft_angle"])
        for obj_id in object_ids:
            for frame_idx, (obj_score, iou_score_max, iou_all, areas, ciou_01, ciou_02, ciou_12) in sorted(iou_log_per_object[obj_id].items()):
                frame_name = os.path.splitext(frame_names[frame_idx])[0]
                laplacian, fft_energy, fft_angle = blur_cache.get(frame_idx, (None, None, None))
                writer.writerow([frame_name, int(obj_id), obj_score, iou_score_max, *iou_all, *areas, ciou_01, ciou_02, ciou_12, laplacian, fft_energy, fft_angle])

    print(f"  [done] {video_name}: {len(frame_names)} frames (per-obj) → {output_video_dir}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 VOS Inference")
    parser.add_argument("--video_dir", required=True,
                        help="JPEG 프레임 루트 경로 (하위에 <video_name>/ 폴더)")
    parser.add_argument("--ann_dir", required=True,
                        help="Annotation PNG 루트 경로 (하위에 <video_name>/ 폴더)")
    parser.add_argument("--output_dir", required=True,
                        help="결과 PNG 저장 루트 경로")
    parser.add_argument("--dataset", default="davis", choices=["davis", "ytvos", "lvos"])
    parser.add_argument("--checkpoint", default=None,
                        help="SAM3 checkpoint 경로. None이면 HuggingFace 자동 다운로드")
    parser.add_argument("--video_list", default=None,
                        help="처리할 비디오 이름 목록 텍스트 파일 (한 줄에 하나). "
                             "None이면 ann_dir 하위 모든 비디오 처리")
    parser.add_argument("--score_thresh", type=float, default=0.0,
                        help="mask logit threshold (default: 0.0)")
    parser.add_argument("--use_all_masks", action="store_true",
                        help="모든 annotation 프레임을 입력으로 사용 "
                             "(기본: 첫 프레임 또는 object별 첫 등장 프레임만 사용)")
    parser.add_argument("--track_object_appearing_later_in_video", action="store_true",
                        help="비디오 중간에 등장하는 object 처리 (LVOS, YouTube-VOS 등). "
                             "object별 독립 추론 후 non-overlapping constraint 적용")
    parser.add_argument("--offload_video_to_cpu", action="store_true",
                        help="긴 비디오 GPU 메모리 절약용 CPU offload")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 모델 로드 ──
    # build_sam3_video_model()은 Detector(Sam3ImageOnVideoMultiGPU)와
    # Tracker(Sam3TrackerPredictor)를 함께 생성한다.
    #
    # VOS에서는 Tracker만 사용하지만, SAM3의 ViT backbone은 Detector 쪽에 있으므로
    # predictor.backbone = detector.backbone 으로 교체해야
    # SAM3의 강력한 ViT feature를 Tracker에서도 활용할 수 있다.
    # (교체하지 않으면 Tracker는 backbone=None 상태로, 이미 캐시된 feature만 사용 가능)
    print("Loading SAM3 model...")
    sam3_model = build_sam3_video_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=(args.checkpoint is None),
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone  # SAM3 ViT backbone을 tracker에 연결

    # ── VOS용: detector의 객체 추가/삭제/재조건화 비활성화 ──
    sam3_model.new_det_thresh = 999.0           # 새 객체 추가 안 함
    sam3_model.hotstart_unmatch_thresh = 99999  # unmatched 객체 삭제 안 함
    sam3_model.hotstart_dup_thresh = 99999      # 중복 객체 삭제 안 함
    sam3_model.suppress_overlapping_based_on_recent_occlusion_threshold = 0.0
    sam3_model.min_trk_keep_alive = -99999      # keep_alive로 인한 삭제 안 함
    sam3_model.recondition_every_nth_frame = -1 # 주기적 cond_frame 갱신 안 함
    print("Model loaded. (detector add/remove/recondition disabled for VOS)")

    # ── 비디오 목록 결정 ──
    # --video_list 파일이 주어지면 그 목록만 처리하고,
    # 없으면 ann_dir 하위의 모든 비디오 폴더를 처리한다.
    if args.video_list is not None:
        with open(args.video_list) as f:
            video_names = [l.strip() for l in f if l.strip()]
    else:
        video_names = sorted([
            v for v in os.listdir(args.ann_dir)
            if os.path.isdir(os.path.join(args.ann_dir, v))
        ])
    print(f"Total videos: {len(video_names)}")

    # ── 비디오별 추론 ──
    for video_name in tqdm(video_names, desc="VOS"):
        try:
            if not args.track_object_appearing_later_in_video:
                run_vos_on_video(
                    predictor=predictor,
                    video_dir=args.video_dir,
                    ann_dir=args.ann_dir,
                    video_name=video_name,
                    output_dir=args.output_dir,
                    score_thresh=args.score_thresh,
                    use_all_masks=args.use_all_masks,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                )
            else:
                vos_separate_inference_per_object(
                    predictor=predictor,
                    video_dir=args.video_dir,
                    ann_dir=args.ann_dir,
                    video_name=video_name,
                    output_dir=args.output_dir,
                    score_thresh=args.score_thresh,
                    use_all_masks=args.use_all_masks,
                    offload_video_to_cpu=args.offload_video_to_cpu,
                )
        except Exception as e:
            print(f"  [error] {video_name}: {e}")
        finally:
            torch.cuda.empty_cache()

    print(f"\nDone. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
