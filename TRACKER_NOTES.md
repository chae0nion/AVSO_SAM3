# SAM3 Detector-Tracker 상호작용 정리

## 1. SAM3 기본 파이프라인 (_det_track_one_frame)

```
매 프레임:
  backbone → detector → _associate_det_trk → run_tracker_update_planning → tracker propagate
                              (IoU 매칭)           (추가/삭제/교정 계획)
```

핵심 파일: `sam3/model/sam3_video_base.py`

---

## 2. Detector가 tracker에 영향을 주는 3가지 경로

| 기능 | 위치 | 설명 |
|------|------|------|
| 새 객체 추가 | `_tracker_add_new_objects` L:1517 | detector 검출 중 기존 track과 IoU 낮은 것 → 새 obj_id 등록 |
| 기존 객체 삭제 | `_process_hotstart` L:1312 | N프레임 연속 unmatched → 해당 obj_id 제거 |
| 교정 (reconditioning) | `_recondition_masklets` L:454 | det score↑ AND IoU↑ → `tracker.add_new_mask()` 로 메모리 갱신 |

---

## 3. Reconditioning 동작 방식

```
조건: det_score >= 0.8 AND IoU(det, trk) >= 0.8
→ 해당 detection mask를 tracker 메모리에 덮어씀
→ propagate_in_video_preflight 재실행

주의: IoU 기반 매칭이므로 identity switch를 고착시킴
      (switch 발생 시 det가 switch된 위치를 정답으로 학습)
```

---

## 4. Text prompt 동작 방식

- `add_prompt(text_str=...)` → detector의 CLIP 텍스트 쿼리로 사용
- text feature는 `feature_cache`에 캐싱 → 동일 text는 재계산 안 함
- text 변경 시 자동으로 다음 프레임에 새 feature 계산
- **CLIP context limit: 77 tokens** → 복잡한 구문은 일부만 반영됨
- presence head는 **개념 단위** (instance 단위 아님)
  - pig#1 off-screen + pig#2 visible → presence score 여전히 높음

---

## 5. Tracker 단독 사용 방법 (SAM2 방식)

```python
# detector 없이 tracker만 사용
tracker_state = model.tracker.init_state(
    cached_features=inference_state["feature_cache"],
    video_height=H, video_width=W, num_frames=N,
)
model.tracker.add_new_mask(tracker_state, frame_idx=0, obj_id=0, mask=my_mask)
model.tracker.propagate_in_video_preflight(tracker_state)

for frame_idx, obj_ids, _, video_res_masks, obj_scores, ... in \
    model.tracker.propagate_in_video(tracker_state, ...):
    mask = video_res_masks[0, 0] > 0   # (H, W) bool
    score = obj_scores[0].sigmoid()
```

`video_res_masks[i]` ↔ `obj_ids[i]` 1:1 대응

---

## 6. Mask 겹침 처리

| 기능 | 위치 | 동작 |
|------|------|------|
| non-overlap suppression | `sam3_tracker_base.py` | 겹치는 영역 중 score 낮은 쪽 제거 |
| shrink suppression | `sam3_tracking_predictor.py` | 특정 크기 이하로 줄어든 mask 제거 |

→ tracker 단독 사용 시에도 적용됨 (`non_overlap_masks_for_output=True` 기본값)

---

## 7. Identity Switch 한계

- reconditioning은 "현재 위치 기준 IoU"만 사용 → switch 교정 불가
- switch 발생 후 detector가 switch된 위치에 맞춰 memory를 다시 굳힘
- 해결하려면 **외부에서 직접** `tracker.add_new_mask()` 로 올바른 mask 주입 필요

---

## 8. 계획 중인 개선 (Robust Single Object Tracking)

```
안정 구간: tracker score 높음 + 유사 객체 여러 개 검출
  → target crop / distractor crop 의 visual feature 온라인 수집

위험 구간: tracker score 저하 OR distractor 근접
  → 수집된 feature로 cosine similarity 기반 ReID
  → 애매하면 Qwen3-VL 비동기 호출 (reference image 비교)
  → 확인된 mask로 tracker.add_new_mask() 교정

Text 전략:
  - 평상시: 짧은 카테고리 text ("아기 돼지")
  - 위험 시: 더 구체적인 text로 교체 ("왼쪽 귀에 갈색 점 있는 흰색 아기 돼지")
  - Qwen: 복잡한 구문 이해 가능 / CLIP은 77 token 한계 있음

GPU 분리:
  SAM3  → cuda:0
  Qwen  → cuda:3  (Qwen3-VL-7B: ~16GB)
```
