"""
SAM3 VOS with Detector-Assisted Cond-Frame Injection

두 가지 독립 컴포넌트:

Component2 (Post-occlusion memory protection):
  - obj_score < 0 이 OCC_MIN_FRAMES 프레임 연속 → is_occluded 선언
  - 복구(obj_score >= 0) 후 POST_OCC_SKIP 프레임 동안 remove_from_non_cond
    (복구 직후 불안정한 프레임이 non_cond 메모리에 쌓이는 것을 방지)

Component1 (Detector-assisted cond-frame):
  - 트리거 조건:
      1) cand_pairwise_iou의 임의 쌍이 PAIR_IOU_THRESH 미만 (후보 mask가 발산)
      2) iou_score_max >= IOU_MAX_THRESH (트래커가 최선 후보를 확신)
      3) detector가 iou_max 후보 위치에서 객체를 확인 (best_hit_idx)
      4) 다른 후보 위치를 다른 detector box가 hit (other_hit_idx != best_hit_idx)
         → 같은 box가 두 후보를 모두 hit하면 동일 객체(포함관계/스케일 차이)로 간주, skip
  - 액션: 현재 프레임을 cond_frame으로 승격

사용법:
  python tools/vos_det_inference_v3.py \\
      --video_dir /data/LVOS2/JPEGImages \\
      --ann_dir   /data/LVOS2/Annotations \\
      --output_dir ./output/lvos2_full \\
      --dataset lvos \\
      --track_object_appearing_later_in_video

사용법:
  python tools/vos_det_inference.py \\
      --video_dir /data/LVOS/JPEGImages \\
      --ann_dir   /data/LVOS/Annotations \\
      --output_dir ./output/lvos_det \\
      --dataset lvos \\
      --track_object_appearing_later_in_video
"""

import argparse
import csv
import os
import sys

# --device 값을 임포트 전에 미리 읽어서 CUDA_VISIBLE_DEVICES 설정
# (CUDA는 첫 번째 CUDA 호출 전에 env var를 읽으므로 임포트 전에 설정해야 함)
_device_val = None
for _i, _arg in enumerate(sys.argv):
    if _arg == '--device' and _i + 1 < len(sys.argv):
        _device_val = sys.argv[_i + 1]
        break
if _device_val is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _device_val

from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_model
from sam3.model.data_misc import BatchedDatapoint, FindStage, convert_my_tensors
from sam3.model.geometry_encoders import Prompt


TEXT_ID_FOR_VISUAL = 1  # sam3_video_inference.py와 동일


# ─────────────────────────────────────────────
# 블러 측정 유틸
# ─────────────────────────────────────────────

def compute_blur_laplacian(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def compute_blur_fft(img_gray: np.ndarray, size: int = 60):
    h, w = img_gray.shape
    cx, cy = w // 2, h // 2
    fft_shift = np.fft.fftshift(np.fft.fft2(img_gray.astype(np.float32)))
    mask = np.ones((h, w), dtype=np.float32)
    mask[cy - size:cy + size, cx - size:cx + size] = 0
    magnitude = np.log(np.abs(fft_shift) + 1) * mask
    fft_energy = float(magnitude.mean())
    y_idx, x_idx = np.where(magnitude > magnitude.mean())
    if len(x_idx) > 0:
        dx = float(x_idx.mean()) - cx
        dy = float(y_idx.mean()) - cy
        fft_angle = float(np.degrees(np.arctan2(dy, dx)))
    else:
        fft_angle = 0.0
    return fft_energy, fft_angle


def analyze_blur(img_path: str, fft_size: int = 60):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    laplacian = compute_blur_laplacian(img)
    fft_energy, fft_angle = compute_blur_fft(img, size=fft_size)
    return laplacian, fft_energy, fft_angle


# ─────────────────────────────────────────────
# 데이터셋 유틸
# ─────────────────────────────────────────────

def load_ann_png(ann_path):
    ann = np.array(Image.open(ann_path).convert("P"))
    obj_ids = np.unique(ann)
    obj_ids = obj_ids[obj_ids != 0]
    return {int(obj_id): (ann == obj_id) for obj_id in obj_ids}


def save_ann_png(mask_dict, video_height, video_width, save_path):
    combined = np.zeros((video_height, video_width), dtype=np.uint8)
    for obj_id, mask in mask_dict.items():
        combined[mask] = obj_id
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img = Image.fromarray(combined).convert("P")
    img.putpalette(_get_davis_palette())
    img.save(save_path)


def _get_davis_palette():
    palette = []
    for i in range(256):
        r = (i & 1) << 7 | (i & 8) << 3 | (i & 64) >> 1
        g = (i & 2) << 6 | (i & 16) << 2 | (i & 128) >> 2
        b = (i & 4) << 5 | (i & 32) << 1
        palette.extend([r, g, b])
    return palette


def get_sorted_frame_names(video_path):
    frames = [
        f for f in os.listdir(video_path)
        if os.path.splitext(f)[-1].lower() in (".jpg", ".jpeg")
    ]
    frames.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return frames


def get_sorted_ann_names(ann_video_dir):
    if not os.path.isdir(ann_video_dir):
        return []
    anns = [f for f in os.listdir(ann_video_dir) if f.endswith(".png")]
    anns.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return anns


# ─────────────────────────────────────────────
# Detector 헬퍼
# ─────────────────────────────────────────────

def mask_to_bbox_xywh_norm(mask_hw: np.ndarray, video_height: int, video_width: int):
    """Binary mask → [x_min, y_min, w, h] (normalized 0~1). None if empty."""
    ys, xs = np.where(mask_hw)
    if len(xs) == 0:
        return None
    x_min = float(xs.min()) / video_width
    x_max = float(xs.max()) / video_width
    y_min = float(ys.min()) / video_height
    y_max = float(ys.max()) / video_height
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def tensor_mask_to_bbox_xyxy_norm(mask_hw: torch.Tensor):
    """[H, W] float tensor (logits or binary) → [x1_n, y1_n, x2_n, y2_n]. None if empty."""
    H, W = mask_hw.shape
    ys, xs = torch.where(mask_hw > 0)
    if len(xs) == 0:
        return None
    return torch.tensor([
        float(xs.min()) / W,
        float(ys.min()) / H,
        float(xs.max()) / W,
        float(ys.max()) / H,
    ], dtype=torch.float32, device=mask_hw.device)


def build_detector_input_batch(inference_state, device):
    """
    inference_state["images"]로부터 detector용 BatchedDatapoint 생성.
    text_ids = TEXT_ID_FOR_VISUAL(1) 로 전부 설정.
    """
    images = inference_state["images"]
    num_frames = inference_state["num_frames"]

    # [num_frames, C, H, W] tensor 만들기
    if isinstance(images, (list, tuple)):
        img_batch = torch.stack([img for img in images])
    else:
        img_batch = images
    img_batch = img_batch.to(device).detach().requires_grad_(False)

    find_text_batch = ["<text placeholder>", "visual"]
    input_box_dim = 258
    input_pts_dim = 257
    stages = [
        FindStage(
            img_ids=[t],
            text_ids=[TEXT_ID_FOR_VISUAL],
            input_boxes=[torch.zeros(input_box_dim)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, input_pts_dim)],
            input_points_mask=[torch.empty(0)],
            object_ids=[],
        )
        for t in range(num_frames)
    ]
    for i in range(len(stages)):
        stages[i] = convert_my_tensors(stages[i])

    input_batch = BatchedDatapoint(
        img_batch=img_batch,
        find_text_batch=find_text_batch,
        find_inputs=stages,
        find_targets=[None] * num_frames,
        find_metadatas=[None] * num_frames,
    )
    return input_batch


def build_visual_prompt(bbox_xywh_norm, device):
    """
    xywh normalized bbox → visual exemplar Prompt (cxcywh format, shape [1,1,4]).
    """
    x, y, w, h = bbox_xywh_norm
    cx, cy = x + w / 2, y + h / 2
    cxcywh = torch.tensor([[cx, cy, w, h]], dtype=torch.float32, device=device)
    box_embeddings = cxcywh.unsqueeze(0)           # [1, 1, 4]  (seq, batch, 4)
    box_labels = torch.ones(1, 1, dtype=torch.long, device=device)  # [1, 1] positive
    return Prompt(
        box_embeddings=box_embeddings,
        box_mask=None,
        box_labels=box_labels,
    )


def get_det_hit_idx(det_boxes_xyxy_norm: torch.Tensor,
                    region_xyxy_norm: torch.Tensor,
                    iou_thresh: float = 0.3):
    """
    det_boxes_xyxy_norm: [N, 4] normalized xyxy (from det_out["bbox"])
    region_xyxy_norm:    [4]    normalized xyxy bbox of candidate mask
    Returns index of the best-matching det box if IoU ≥ iou_thresh, else None.
    """
    if det_boxes_xyxy_norm.numel() == 0:
        return None
    region = region_xyxy_norm.to(det_boxes_xyxy_norm.device)
    x1 = torch.max(det_boxes_xyxy_norm[:, 0], region[0])
    y1 = torch.max(det_boxes_xyxy_norm[:, 1], region[1])
    x2 = torch.min(det_boxes_xyxy_norm[:, 2], region[2])
    y2 = torch.min(det_boxes_xyxy_norm[:, 3], region[3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area_det = (det_boxes_xyxy_norm[:, 2] - det_boxes_xyxy_norm[:, 0]) * \
               (det_boxes_xyxy_norm[:, 3] - det_boxes_xyxy_norm[:, 1])
    area_reg = (region[2] - region[0]) * (region[3] - region[1])
    union = area_det + area_reg - inter
    iou = inter / (union + 1e-6)
    best_idx = int(iou.argmax().item())
    return best_idx if iou[best_idx].item() >= iou_thresh else None


# ─────────────────────────────────────────────
# cond_frame 관리 헬퍼
# ─────────────────────────────────────────────

def promote_to_cond_frame(inference_state, frame_idx):
    """
    non_cond_frame_outputs[frame_idx] → cond_frame_outputs[frame_idx] 이동.
    propagate_in_video가 run_mem_encoder=True로 실행되므로 maskmem_features는 이미 있음.
    Returns True if promotion succeeded, False if frame not in non_cond.
    """
    output_dict = inference_state["output_dict"]
    entry = output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
    if entry is None:
        return False
    output_dict["cond_frame_outputs"][frame_idx] = entry
    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        per_obj = obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        if per_obj is not None:
            obj_output_dict["cond_frame_outputs"][frame_idx] = per_obj
    return True


def remove_from_non_cond(inference_state, frame_idx):
    """non_cond memory에서 해당 프레임 제거 (post-occ cooldown 중 호출)."""
    output_dict = inference_state["output_dict"]
    output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
    for obj_output_dict in inference_state["output_dict_per_obj"].values():
        obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)


# ─────────────────────────────────────────────
# 핵심 추론 함수
# ─────────────────────────────────────────────

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def save_mask_png(mask_hw: np.ndarray, save_path: str, contour_color=(0, 255, 0), thickness=2):
    """Binary mask → RGB PNG with filled region (semi-transparent green) + contour."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    h, w = mask_hw.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # filled region: semi-transparent tint (mix 40%)
    tint = np.array(contour_color, dtype=np.uint8)
    canvas[mask_hw > 0] = (tint * 0.4).astype(np.uint8)
    # contour
    binary = (mask_hw > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, contour_color, thickness)
    Image.fromarray(canvas).save(save_path)


def vos_det_per_object(
    sam3_model,
    predictor,
    video_dir,
    ann_dir,
    video_name,
    output_dir,
    score_thresh=0.0,
    use_all_masks=False,
    offload_video_to_cpu=False,
    save_det_masks=False,
    enable_component2=True,
    enable_component1=True,
    compute_blur=False,
    post_occ_skip=None,
):
    """
    Detector-assisted VOS: object별 순차 추론.
    Component2: post-occ 복구 직후 remove_from_non_cond
    Component1: detector distractor 검증 → non_cond → cond 승격
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
    num_frames = inference_state["num_frames"]
    device = sam3_model.device

    # ── 모든 annotation에서 object별 입력 mask 수집 ──
    inputs_per_object = defaultdict(dict)
    for ann_name in ann_names:
        ann_stem = os.path.splitext(ann_name)[0]
        if ann_stem not in stem_to_idx:
            continue
        frame_idx = stem_to_idx[ann_stem]
        obj_masks = load_ann_png(os.path.join(ann_video_dir, ann_name))
        for obj_id, mask in obj_masks.items():
            if not np.any(mask):
                continue
            if len(inputs_per_object[obj_id]) > 0 and not use_all_masks:
                continue
            inputs_per_object[obj_id][frame_idx] = torch.from_numpy(mask).float()

    if not inputs_per_object:
        print(f"  [skip] No valid annotation frames for {video_name}")
        return

    # ── Detector input_batch 빌드 (Component1 또는 save_det_masks 사용 시에만) ──
    detector_input_batch = (
        build_detector_input_batch(inference_state, device)
        if (enable_component1 or save_det_masks) else None
    )

    # ── 활성화 플래그 ──
    ENABLE_COMPONENT2 = enable_component2   # post-occ 복구 직후 remove_from_non_cond
    ENABLE_COMPONENT1 = enable_component1   # detector distractor → cond 승격
    ENABLE_OCC   = ENABLE_COMPONENT2   # occlusion 상태 머신 실행 여부

    # ── 하이퍼파라미터 ──
    # Part 1: proactive cond-frame promotion
    OCC_SCORE_THRESH   = 0      # obj_score < 0 → occluded
    OCC_MIN_FRAMES     = 5    # N 프레임 연속 occluded → is_occluded 선언
    CONF_OBJ_THRESH    = 3.0    # "고신뢰" obj_score logit 기준
    CONF_IOU_THRESH    = 0.75   # "고신뢰" iou_score_max 기준
    CONF_STREAK_THRESH = 5      # N 프레임 연속 고신뢰 → cond_frame 승격
    COND_MIN_INTERVAL  = 5   # 승격 최소 간격 (과잉 cond 방지)
    POST_OCC_SKIP      = post_occ_skip if post_occ_skip is not None else 5
    # Part 2: detector-assisted distractor detection
    PAIR_IOU_THRESH    = 0.7    # 이 값 미만인 쌍이 있으면 "후보 위치 발산"
    IOU_MAX_THRESH     = 0.7    # tracker 최선 후보 iou ≥ 이 값
    DET_IOU_THRESH     = 0.3    # detector 박스 ↔ 후보 영역 IoU 임계
    DET_COOLDOWN       = 5     # 주입 후 최소 간격

    # ── object별 순차 추론 ──
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    iou_log_per_object = defaultdict(dict)
    blur_cache = {}
    event_log_rows = []
    frame_debug_rows = []

    for obj_id in object_ids:
        input_frame_inds = sorted(inputs_per_object[obj_id])
        first_frame_idx = input_frame_inds[0]

        predictor.clear_all_points_in_video(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=obj_id,
                mask=inputs_per_object[obj_id][input_frame_idx],
            )

        # ── visual exemplar 준비 (첫 등장 프레임 GT mask → bbox) ──
        first_gt_mask_np = inputs_per_object[obj_id][first_frame_idx].numpy() > 0.5
        exemplar_bbox = mask_to_bbox_xywh_norm(first_gt_mask_np, video_height, video_width)
        if exemplar_bbox is not None:
            visual_prompt = build_visual_prompt(exemplar_bbox, device)
        else:
            visual_prompt = None

        # Detector feature cache (객체별로 독립 유지; text feature 재사용을 위해)
        det_feature_cache = {}

        # Part 1 상태
        occ_frame_count      = 0
        is_occluded          = False
        high_conf_streak     = 0      # 연속 고신뢰 프레임 수
        last_cond_frame_idx  = first_frame_idx  # 마지막 cond 승격 프레임 (초기 GT 프레임)
        post_occ_skip        = 0      # occlusion 복구 후 non_cond 스킵 카운터

        # Part 2 상태
        det_cooldown = 0

        for (out_frame_idx, _, _, video_res_masks,
             obj_scores, iou_scores, iou_scores_all,
             mask_areas_all, cand_pairwise_iou,
             video_res_multimasks) in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=first_frame_idx,
            max_frame_num_to_track=num_frames,
            reverse=False,
            propagate_preflight=True,
        ):
            output_scores_per_object[obj_id][out_frame_idx] = video_res_masks.cpu()

            frame_stem = os.path.splitext(frame_names[out_frame_idx])[0]

            # ── detector mask 저장 (빨강 contour) ──
            if save_det_masks and visual_prompt is not None:
                with torch.inference_mode(False), torch.no_grad():
                    det_out = sam3_model.run_backbone_and_detection(
                        frame_idx=out_frame_idx,
                        num_frames=num_frames,
                        input_batch=detector_input_batch,
                        geometric_prompt=visual_prompt,
                        feature_cache=det_feature_cache,
                        reverse=False,
                        allow_new_detections=True,
                    )
                det_masks_lowres = det_out["mask"]   # (N, H_low, W_low)
                det_scores_det   = det_out["scores"]  # (N,)
                if det_masks_lowres.shape[0] > 0:
                    det_masks_hires = F.interpolate(
                        det_masks_lowres.unsqueeze(1).float(),
                        size=(video_height, video_width),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)  # (N, H, W)
                    for det_i, (det_mask, det_score) in enumerate(
                        zip(det_masks_hires, det_scores_det)
                    ):
                        det_mask_np = (det_mask > 0).cpu().numpy()
                        save_mask_png(
                            det_mask_np,
                            os.path.join(
                                output_dir, video_name,
                                "detector", f"obj{obj_id:03d}",
                                f"{frame_stem}_det{det_i:02d}_s{float(det_score):.2f}.png",
                            ),
                            contour_color=(255, 0, 0),
                        )

            # 스코어 로깅
            obj_score     = float(obj_scores[0].item()) if obj_scores is not None else None
            iou_score_max = float(iou_scores[0].item()) if iou_scores is not None else None
            iou_all   = iou_scores_all[0].tolist() if iou_scores_all is not None else [None, None, None]
            areas     = mask_areas_all[0].tolist() if mask_areas_all is not None else [None, None, None]
            piou_mat  = cand_pairwise_iou[0] if cand_pairwise_iou is not None else None
            if piou_mat is not None:
                ciou_01, ciou_02, ciou_12 = piou_mat[[0, 0, 1], [1, 2, 2]].tolist()
            else:
                ciou_01, ciou_02, ciou_12 = None, None, None
            if compute_blur and out_frame_idx not in blur_cache:
                img_path = os.path.join(video_path, frame_names[out_frame_idx])
                blur_cache[out_frame_idx] = analyze_blur(img_path)
            iou_log_per_object[obj_id][out_frame_idx] = (
                obj_score, iou_score_max, iou_all, areas, ciou_01, ciou_02, ciou_12
            )

            # ────────────────────────────────────────────────────
            # Component2: occlusion 상태 머신
            #   Component2: post-occ 복구 직후 → remove_from_non_cond
            # ────────────────────────────────────────────────────
            p1_action = "none"

            if obj_score is not None and obj_score <= 0:
                remove_from_non_cond(inference_state, out_frame_idx)

            if ENABLE_OCC and obj_score is not None:
                if obj_score < OCC_SCORE_THRESH:
                    occ_frame_count += 1
                    high_conf_streak = 0
                    if not is_occluded and occ_frame_count >= OCC_MIN_FRAMES:
                        is_occluded = True
                        p1_action = "occ_start"
                        print(f"  [Component2] {video_name} obj={obj_id} frame={frame_names[out_frame_idx]} "
                              f"→ occlusion declared (count={occ_frame_count})")
                    elif is_occluded:
                        p1_action = "occluded"
                    else:
                        p1_action = "occ_accumulating"
                else:
                    # visible 상태
                    if is_occluded:
                        # 복구: post_occ_skip 설정 (Component2 활성 시에만)
                        post_occ_skip = POST_OCC_SKIP if ENABLE_COMPONENT2 else 0
                        is_occluded = False
                        occ_frame_count = 0
                        high_conf_streak = 0
                        event_log_rows.append([
                            frame_stem, int(obj_id), "Component2", "occ_recovered",
                            f"post_occ_skip={post_occ_skip}",
                        ])
                        print(f"  [Component2] {video_name} obj={obj_id} frame={frame_names[out_frame_idx]} "
                              f"→ occlusion recovered, post_occ_skip={post_occ_skip}")
                    else:
                        occ_frame_count = 0

                    if post_occ_skip > 0:
                        # Component2: 복구 직후 cooldown → remove_from_non_cond + streak 차단
                        post_occ_skip -= 1
                        high_conf_streak = 0
                        remove_from_non_cond(inference_state, out_frame_idx)
                        p1_action = "post_occ_skip"
                    else:
                        high_conf_streak = 0
                        p1_action = "low_conf"

            # ────────────────────────────────────────────────────
            # Component1: detector distractor 감지 → cond_frame 승격
            # ────────────────────────────────────────────────────
            p2_action = "none" if ENABLE_COMPONENT1 else "disabled"
            p2_min_pair_iou = None
            p2_det_called = False
            p2_det_n_boxes = None
            p2_det_at_iou_max = None
            p2_distractor_found = None

            if (ENABLE_COMPONENT1
                    and det_cooldown == 0
                    and post_occ_skip == 0
                    and visual_prompt is not None
                    and obj_score is not None and obj_score > 0
                    and iou_score_max is not None
                    and piou_mat is not None
                    and iou_scores_all is not None
                    and video_res_multimasks is not None):

                # Step 1: iou_max 충분 AND 후보 위치 발산
                M = piou_mat.shape[0]
                pair_ious = [float(piou_mat[i, j].item())
                             for i in range(M) for j in range(i + 1, M)]
                min_pair_iou = min(pair_ious)
                p2_min_pair_iou = min_pair_iou

                if iou_score_max >= IOU_MAX_THRESH and min_pair_iou < PAIR_IOU_THRESH:
                    best_cand_idx = int(torch.argmax(iou_scores_all[0]).item())
                    iou_max_mask_hw = video_res_multimasks[0, best_cand_idx]
                    iou_max_bbox = tensor_mask_to_bbox_xyxy_norm(iou_max_mask_hw)

                    if iou_max_bbox is not None:
                        # Step 2: detector 호출
                        print(f"  [Part2-Step1] {video_name} obj={obj_id} frame={frame_names[out_frame_idx]} "
                              f"iou_max={iou_score_max:.3f} min_pair_iou={min_pair_iou:.3f} → calling detector")
                        with torch.inference_mode(False), torch.no_grad():
                            det_out = sam3_model.run_backbone_and_detection(
                                frame_idx=out_frame_idx,
                                num_frames=num_frames,
                                input_batch=detector_input_batch,
                                geometric_prompt=visual_prompt,
                                feature_cache=det_feature_cache,
                                reverse=False,
                                allow_new_detections=True,
                            )
                        det_boxes = det_out["bbox"]
                        p2_det_called = True
                        p2_det_n_boxes = int(det_boxes.shape[0])

                        best_hit_idx = get_det_hit_idx(det_boxes, iou_max_bbox, DET_IOU_THRESH)
                        if best_hit_idx is None:
                            p2_det_at_iou_max = False
                            p2_action = "no_det_at_iou_max"
                            event_log_rows.append([
                                frame_stem, int(obj_id), "Part2", "no_det_at_iou_max",
                                f"min_pair_iou={min_pair_iou:.4f}",
                                f"iou_max={iou_score_max:.4f}",
                                f"det_n_boxes={p2_det_n_boxes}",
                            ])
                        else:
                            p2_det_at_iou_max = True
                            # Step 3: 다른 후보 위치에도 객체 있는지 확인 (distractor)
                            # best와 다른 detector box가 hit해야 진짜 방해꾼
                            distractor_found = False
                            for cand_idx in range(M):
                                if cand_idx == best_cand_idx:
                                    continue
                                other_bbox = tensor_mask_to_bbox_xyxy_norm(
                                    video_res_multimasks[0, cand_idx])
                                if other_bbox is None:
                                    continue
                                other_hit_idx = get_det_hit_idx(det_boxes, other_bbox, DET_IOU_THRESH)
                                if (float(piou_mat[best_cand_idx, cand_idx].item()) < PAIR_IOU_THRESH
                                        and other_hit_idx is not None
                                        and other_hit_idx != best_hit_idx):
                                    distractor_found = True
                                    break
                            p2_distractor_found = distractor_found

                            if distractor_found:
                                if promote_to_cond_frame(inference_state, out_frame_idx):
                                    det_cooldown = DET_COOLDOWN
                                    p2_action = "cond_promoted"
                                    n_cond = len(inference_state["output_dict"]["cond_frame_outputs"])
                                    print(f"  [Part2] {video_name} obj={obj_id} "
                                          f"frame={frame_names[out_frame_idx]} "
                                          f"min_pair_iou={min_pair_iou:.2f} iou_max={iou_score_max:.2f} "
                                          f"→ distractor confirmed, promoted to cond (total_cond={n_cond})")
                                    event_log_rows.append([
                                        frame_stem, int(obj_id), "Part2", "cond_promoted",
                                        f"min_pair_iou={min_pair_iou:.4f}",
                                        f"iou_max={iou_score_max:.4f}",
                                        f"det_n_boxes={p2_det_n_boxes}",
                                        f"best_cand_idx={best_cand_idx}",
                                        f"total_cond={n_cond}",
                                    ])
                                else:
                                    p2_action = "promote_skipped_not_in_non_cond"
                            else:
                                p2_action = "no_distractor"
                                event_log_rows.append([
                                    frame_stem, int(obj_id), "Part2", "no_distractor",
                                    f"min_pair_iou={min_pair_iou:.4f}",
                                    f"iou_max={iou_score_max:.4f}",
                                    f"det_n_boxes={p2_det_n_boxes}",
                                ])
                    else:
                        p2_action = "no_bbox"
                else:
                    p2_action = "cond_not_met"
            elif det_cooldown > 0:
                p2_action = "cooldown"

            # det_cooldown 카운트다운 (블록 이후에 감소 → DET_COOLDOWN=5이면 정확히 5프레임 차단)
            if det_cooldown > 0:
                det_cooldown -= 1

            # ── per-frame debug 기록 ──
            output_dict = inference_state["output_dict"]
            in_cond     = out_frame_idx in output_dict["cond_frame_outputs"]
            in_non_cond = out_frame_idx in output_dict["non_cond_frame_outputs"]
            mem_key     = "cond" if in_cond else ("non_cond" if in_non_cond else "none")
            n_cond_frames = len(output_dict["cond_frame_outputs"])

            frame_debug_rows.append([
                frame_stem, int(obj_id),
                f"{obj_score:.4f}" if obj_score is not None else None,
                occ_frame_count, is_occluded, post_occ_skip,
                high_conf_streak, last_cond_frame_idx, p1_action,
                f"{iou_score_max:.4f}" if iou_score_max is not None else None,
                f"{p2_min_pair_iou:.4f}" if p2_min_pair_iou is not None else None,
                p2_det_called, p2_det_n_boxes, p2_det_at_iou_max, p2_distractor_found, p2_action,
                det_cooldown, mem_key, n_cond_frames,
            ])

    # ── Step 3: per-object score 합산 + non-overlapping constraint ──
    output_video_dir = os.path.join(output_dir, video_name)
    for frame_idx, frame_name in enumerate(frame_names):
        scores = torch.full(
            (len(object_ids), 1, video_height, video_width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, obj_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[obj_id]:
                scores[i] = output_scores_per_object[obj_id][frame_idx]

        scores = predictor._apply_non_overlapping_constraints(scores)
        mask_dict = {
            obj_id: (scores[i] > score_thresh).numpy().squeeze()
            for i, obj_id in enumerate(object_ids)
        }
        stem = os.path.splitext(frame_name)[0]
        save_ann_png(mask_dict, video_height, video_width,
                     os.path.join(output_video_dir, f"{stem}.png"))

    # ── 로그 저장 ──
    os.makedirs(output_video_dir, exist_ok=True)
    log_path = os.path.join(output_video_dir, "scores.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "obj_id", "object_score_logit", "iou_score_max",
            "iou_score_0", "iou_score_1", "iou_score_2",
            "mask_area_0", "mask_area_1", "mask_area_2",
            "cand_iou_01", "cand_iou_02", "cand_iou_12",
            "blur_laplacian", "blur_fft_energy", "blur_fft_angle",
        ])
        for obj_id in object_ids:
            for frame_idx, (obj_score, iou_score_max, iou_all, areas, ciou_01, ciou_02, ciou_12) \
                    in sorted(iou_log_per_object[obj_id].items()):
                frame_name_stem = os.path.splitext(frame_names[frame_idx])[0]
                lap, fft_e, fft_a = blur_cache.get(frame_idx, (None, None, None))
                writer.writerow([
                    frame_name_stem, int(obj_id), obj_score, iou_score_max,
                    *iou_all, *areas, ciou_01, ciou_02, ciou_12,
                    lap, fft_e, fft_a,
                ])

    # ── events.txt 저장 ──
    events_path = os.path.join(output_video_dir, "events.txt")
    with open(events_path, "w", encoding="utf-8") as f:
        f.write(f"# events log: {video_name}\n")
        f.write(f"# columns: frame  obj_id  part  action  details...\n\n")
        if event_log_rows:
            for row in event_log_rows:
                f.write("  ".join(str(x) for x in row) + "\n")
        else:
            f.write("(no events triggered)\n")

    # ── frame_debug.txt 저장 ──
    debug_path = os.path.join(output_video_dir, "frame_debug.txt")
    header = (
        "frame  obj_id  "
        "obj_score  occ_count  is_occluded  post_occ_skip  "
        "streak  last_cond_idx  p1_action  "
        "iou_max  p2_min_pair_iou  "
        "p2_det_called  p2_det_n_boxes  p2_det_at_iou_max  p2_distractor  p2_action  det_cooldown  "
        "mem_key  n_cond_frames"
    )
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(f"# frame debug log: {video_name}\n")
        f.write(f"# {header}\n\n")
        for row in frame_debug_rows:
            f.write("  ".join(str(x) for x in row) + "\n")

    print(f"  [done] {video_name}: {len(frame_names)} frames (det-assisted) → {output_video_dir}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SAM3 VOS with Detector-Assisted Cond-Frame")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--ann_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="lvos", choices=["davis", "ytvos", "lvos"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--video_list", default=None)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    parser.add_argument("--use_all_masks", action="store_true")
    parser.add_argument("--track_object_appearing_later_in_video", action="store_true")
    parser.add_argument("--offload_video_to_cpu", action="store_true")
    parser.add_argument("--save_det_masks", action="store_true",
                        help="매 프레임 detector mask도 빨강 contour PNG로 저장")
    parser.add_argument("--no_component2", action="store_true", help="Component2 비활성화 (post-occ 복구 직후 저장 방지)")
    parser.add_argument("--no_component1", action="store_true", help="Component1 비활성화 (detector distractor → cond 승격)")
    parser.add_argument("--post_occ_skip", type=int, default=None, help="post-occ 스킵 프레임 수 (미지정 시 기본값 5)")
    parser.add_argument("--blur", action="store_true", help="blur 분석 활성화 (scores.csv에 laplacian/fft 추가, 느림)")
    parser.add_argument("--device", type=int, default=None,
                        help="사용할 GPU 인덱스 (예: --device 4). 미지정 시 CUDA_VISIBLE_DEVICES 환경변수 또는 기본값 사용")
    return parser.parse_args()


def main():
    args = parse_args()

    # GPU 선택: --device 지정 시 해당 GPU만 visible하게 설정
    if args.device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    print("Loading SAM3 model...")
    sam3_model = build_sam3_video_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=(args.checkpoint is None),
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    # VOS용: detector 객체 추가/삭제/재조건화 비활성화
    sam3_model.new_det_thresh = 999.0
    sam3_model.hotstart_unmatch_thresh = 99999
    sam3_model.hotstart_dup_thresh = 99999
    sam3_model.suppress_overlapping_based_on_recent_occlusion_threshold = 0.0
    sam3_model.min_trk_keep_alive = -99999
    sam3_model.recondition_every_nth_frame = -1
    print("Model loaded. (detector add/remove/recondition disabled for VOS)")

    if args.video_list is not None:
        with open(args.video_list) as f:
            video_names = [l.strip() for l in f if l.strip()]
    else:
        video_names = sorted([
            v for v in os.listdir(args.ann_dir)
            if os.path.isdir(os.path.join(args.ann_dir, v))
        ])
    print(f"Total videos: {len(video_names)}")

    for video_name in tqdm(video_names, desc="VOS-det"):
        try:
            vos_det_per_object(
                sam3_model=sam3_model,
                predictor=predictor,
                video_dir=args.video_dir,
                ann_dir=args.ann_dir,
                video_name=video_name,
                output_dir=args.output_dir,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                offload_video_to_cpu=args.offload_video_to_cpu,
                save_det_masks=args.save_det_masks,
                enable_component2=not args.no_component2,
                enable_component1=not args.no_component1,
                compute_blur=args.blur,
                post_occ_skip=args.post_occ_skip,
            )
        except Exception as e:
            import traceback
            print(f"  [error] {video_name}: {e}")
            traceback.print_exc()
        finally:
            # 비디오 하나가 끝날 때마다 CUDA 메모리를 명시적으로 해제
            # (inference_state / detector_input_batch가 GC되더라도
            #  empty_cache() 전까지 GPU 메모리는 반환되지 않음)
            torch.cuda.empty_cache()

    print(f"\nDone. Results: {args.output_dir}")


if __name__ == "__main__":
    main()
