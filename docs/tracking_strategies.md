# SAM3 Tracking Strategies — 구현 위치 정리

논문 C.3절에 기술된 6가지 전략이 코드 어디에 구현되어 있는지 정리.

---

## 전제: `allow_new_detections` 플래그

모든 전략은 detector 결과를 사용한다.
**현재 `vos_inference.py`에서는 이 플래그가 `False`라서 아래 전략들이 전부 비활성화된다.**

```python
# sam3_video_inference.py:394
allow_new_detections = has_text_prompt or has_geometric_prompt

# sam3_video_base.py:368
if not allow_new_detections:
    pred_probs = pred_probs - 1e8  # 모든 detection 무효화
```

---

## 매 프레임 실행 흐름

```
_det_track_one_frame()  ← sam3_video_base.py:152
  ├── Step 1: run_backbone_and_detection()       :313  → detector 실행, feature cache 채움
  ├── Step 2: run_tracker_propagation()          :402  → tracker memory attention → mask/score
  ├── Step 3: run_tracker_update_planning_phase():506  → 전략 ③④⑤⑥ 판단
  ├── Step 4: run_tracker_update_execution_phase()    → 판단 실행 (_recondition_masklets 등)
  └── Step 5: build_outputs()                         → 최종 mask 생성
```

---

## 전략 1: Track Confirmation Delay (hotstart)

**논문:** 프레임 τ 출력을 τ+T 관찰 후까지 지연. 기본 T=15.

**구현:** `sam3_video_inference.py` — `propagate_in_video()` 내부

```python
# :282
unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1

# :295-315
if self.hotstart_delay > 0:
    hotstart_buffer.append([frame_idx, out])
    if len(hotstart_buffer) >= self.hotstart_delay:
        yield_list = hotstart_buffer[:1]   # 가장 오래된 프레임 출력
        hotstart_buffer = hotstart_buffer[1:]
```

**관련 파라미터 (model_builder.py):**
```python
hotstart_delay=15
hotstart_unmatch_thresh=8
hotstart_dup_thresh=8
```

---

## 전략 2: Removal of Unconfirmed Masklets

**논문:** 확인 창 [t, t+T] 내에서 MDS < V이고 t_first ≥ t이면 제거.

**구현:** `sam3_video_base.py` — `update_masklet_confirmation_status()` :1644

```python
# :1671
status: MaskletConfirmationStatus  # UNCONFIRMED / CONFIRMED

# :1674-1691
if match:
    consecutive_det_num += 1
    if consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh:
        status = CONFIRMED
else:
    consecutive_det_num = 0
```

출력 시 unconfirmed masklet 숨김:
```python
# sam3_video_inference.py:421
out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][is_unconfirmed]
```

**관련 파라미터:**
```python
masklet_confirmation_enable=False          # 기본 비활성
masklet_confirmation_consecutive_det_thresh=3
```

---

## 전략 3: Removal of Duplicate Masklets

**논문:** 두 masklet이 같은 detection에 매칭되면 나중에 생긴 것 제거.

**구현:** `sam3_video_base.py` — `_process_hotstart()` :1403

```python
# :1406-1414
for det_idx, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
    if len(matched_trk_obj_ids) > 1:
        # 가장 먼저 등장한 masklet 유지
        keep = min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
        to_remove = matched_trk_obj_ids - {keep}

# :1428
if overlap_count >= self.hotstart_dup_thresh:
    remove_obj_ids.add(later_obj_id)
```

**관련 파라미터:**
```python
hotstart_dup_thresh=8   # 중복 판정 프레임 수
```

---

## 전략 4: Masklet Suppression (MDS < 0)

**논문:** Si(t_first, τ) < 0 이면 mask 출력을 0으로 만들지만 tracker 상태는 유지.

**구현:** `sam3_video_base.py` — `trk_keep_alive` 스코어로 구현

```python
# :1354-1365  (run_tracker_update_planning_phase 내)
if matched:
    trk_keep_alive[obj_id] = min(max_trk_keep_alive, trk_keep_alive[obj_id] + 1)
else:
    trk_keep_alive[obj_id] = max(min_trk_keep_alive, trk_keep_alive[obj_id] - 1)

# :1393  (_process_hotstart 내)
if trk_keep_alive[obj_id] <= 0:
    suppressed_obj_ids.add(obj_id)   # mask를 0으로 출력
    # tracker 상태는 제거하지 않음
```

**관련 파라미터:**
```python
init_trk_keep_alive=30    # 초기값 (VOS 세팅: 높게 설정됨)
max_trk_keep_alive=30
min_trk_keep_alive=-1
suppress_unmatched_only_within_hotstart=True
```

---

## 전략 5: Periodic Re-Prompting (N=16)

**논문:** 매 N번째 프레임에서 IoU(det, tracker) ≥ 0.8 AND 두 score 모두 > 0.8 이면 tracker 메모리 재초기화.

**구현:** `sam3_video_base.py` — `run_tracker_update_planning_phase()` :506

```python
# :721-725  트리거 조건
should_recondition_periodic = (
    self.recondition_every_nth_frame > 0
    and frame_idx % self.recondition_every_nth_frame == 0
    and len(trk_id_to_max_iou_high_conf_det) > 0
)

# :1277-1289  high-conf detection 조건
HIGH_CONF_THRESH = 0.8
HIGH_IOU_THRESH  = 0.8
det_is_high_conf = (det_scores_np >= HIGH_CONF_THRESH) & ~is_new_det
det_is_high_iou  = np.max(ious_np, axis=1) >= HIGH_IOU_THRESH
# 둘 다 만족하는 경우만 trk_id_to_max_iou_high_conf_det에 추가
```

실행:
```python
# _recondition_masklets() :454
# tracker score > 0.8 일 때만 실제 recondition
if obj_score > HIGH_CONF_THRESH:
    self.tracker.add_new_mask(inference_state, frame_idx, obj_id, mask=new_mask_binary)
    self.tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
```

**관련 파라미터:**
```python
recondition_every_nth_frame=16
```

---

## 전략 6: Detection-Guided Re-Prompting (bbox IoU < 0.85)

**논문:** tracker 예측과 detection의 bbox IoU < 0.85 이면 (drift 감지) tracker 재초기화.

**구현:** `sam3_video_base.py` — `run_tracker_update_planning_phase()` :669

```python
# :669-719
if self.reconstruction_bbox_iou_thresh > 0:
    for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
        iou = fast_diag_box_iou(det_box, tracker_box)[0]
        if (
            iou < self.reconstruction_bbox_iou_thresh   # 기본 0.85
            and det_score >= self.reconstruction_bbox_det_score
        ):
            should_recondition_iou = True
            reconditioned_obj_ids.add(trk_obj_id)
```

실행은 전략 5와 동일하게 `_recondition_masklets()` 호출.

**관련 파라미터:**
```python
reconstruction_bbox_iou_thresh=0.85  # sam3_video_base.py:80 (기본값)
reconstruction_bbox_det_score=0.8
```

---

## 핵심 제약사항

두 re-prompting 전략(5, 6) 모두:
- **detector가 해당 객체를 탐지해야** 함 → 완전히 사라지면 작동 안 함
- `allow_new_detections=True` 필요 → text/box prompt가 있어야 활성화
- tracker score > 0.8 조건 → tracker가 **이미 실패한 경우에는 recondition 안 됨**

따라서 tracker가 완전히 실패한 경우의 복구는 현재 구현에 없음.
