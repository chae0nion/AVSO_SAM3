# vos_det_inference.py 수정 사항

## 1. 로그 파일 추가 (events.txt, frame_debug.txt)

### 추가 위치
- `vos_det_per_object` 함수 내부

### 내용
- `event_log_rows`, `frame_debug_rows` 리스트를 object별 루프 전에 초기화
- Part 1 로직: `p1_action`, `p1_anchor_frame` 상태 변수 추가, anchor_not_found 분기 추가
  - injected / anchor_not_found 이벤트 → `event_log_rows` 기록
- Part 2 로직: `p2_action`, `p2_min_pair_iou`, `p2_stable`, `p2_det_called`,
  `p2_det_n_boxes`, `p2_det_at_iou_max`, `p2_distractor_found`, `p2_score_spread` 상태 변수 추가
  - no_det_at_iou_max / no_distractor / injected 이벤트 → `event_log_rows` 기록
- 매 프레임 끝에 `frame_debug_rows.append(...)` 로 상태 기록
- 비디오 완료 후 두 파일 저장:
  - `{output_dir}/{video_name}/events.txt`
  - `{output_dir}/{video_name}/frame_debug.txt`

---

## 2. torch.inference_mode 충돌 수정 (미완료 - 반려됨)

### 문제
`propagate_in_video` 루프 내에서 `run_backbone_and_detection` 호출 시
`RuntimeError: Inference tensors cannot be saved for backward` 발생.

### 원인
`inference_state["images"]`가 inference tensor로 저장되어 있고,
`run_backbone_and_detection` 내부 activation checkpointing이 backward 저장을 시도.

### 시도한 수정 (현재 코드에 반영됨)
- `run_backbone_and_detection` 호출 2곳을 `with torch.inference_mode(False):` 로 감쌈
  - line ~411 (save_det_masks 경로)
  - line ~583 (Part 2 경로)

### 추가 수정 (반영됨)
- `build_detector_input_batch` 내 `img_batch.to(device)` → `img_batch.to(device).clone()`
  으로 inference tensor 표시 제거

---

## 3. Part 2 Step 2 완전 비활성화 + part2_history 조건 제거

### 파일
- `tools/vos_det_inference.py`

### 내용
Step 2 안정성 체크 전체 주석 처리 → Step 1 통과 시 즉시 Step 3(detector)으로 진행.
`len(part2_history) == PART2_STABLE_FRAMES` 전제 조건도 주석 처리.

```python
# Step 2 비활성화
# stable = all(h_iou_max >= IOU_MAX_THRESH for (h_iou_max, _, _, _) in part2_history)
# if stable:
if True:  # Step 1 통과 시 바로 detector 호출

# 전제 조건에서 제거
# and len(part2_history) == PART2_STABLE_FRAMES
```

---

## 4. pre_occ_history 저장 조건 추가

### 파일
- `tools/vos_det_inference.py`

### 내용
`pre_occ_history`에 obj_score > 0 (객체 보이는 프레임)인 것만 저장하도록 조건 추가.

```python
# 변경 전
if obj_score is not None and iou_score_max is not None:

# 변경 후
if obj_score is not None and iou_score_max is not None and obj_score > 0:
```

---

## 5. non_cond_frame_outputs 저장 조건 추가

### 파일
- `sam3/model/sam3_tracking_predictor.py`

### 내용
propagate_in_video 내 non-cond 프레임 저장 시, obj_score > 0 (객체 보이는 프레임)인 경우에만
`non_cond_frame_outputs`에 저장. 저장되지 않은 슬롯은 memory attention 시 자동 skip됨.

```python
# 변경 전
output_dict[storage_key][frame_idx] = current_out

# 변경 후
if (obj_scores > 0).any():
    output_dict[storage_key][frame_idx] = current_out
```
