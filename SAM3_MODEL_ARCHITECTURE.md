# SAM3 model/ 파일 구조 및 작동 방식

## 목차
1. [전체 구조 개요](#1-전체-구조-개요)
2. [이미지 모델 흐름](#2-이미지-모델-흐름)
3. [비디오 모델 흐름](#3-비디오-모델-흐름)
4. [파일별 상세 설명](#4-파일별-상세-설명)

---

## 1. 전체 구조 개요

SAM3는 **Detector**(이미지)와 **Tracker**(비디오)가 **ViT backbone을 공유**하는 구조다.

```
                    ┌─────────────────────────────┐
                    │        SAM3 전체 모델         │
                    │                             │
          ┌─────────┴──────────┐                 │
          │     Detector       │   Tracker        │
          │   (Sam3Image)      │ (Sam3TrackerPredictor)
          │                   │                  │
          └──────┬────────────┘                  │
                 │ 공유                           │
          ┌──────▼──────────────────────┐         │
          │       SAM3VLBackbone        │─────────┘
          │  ViT + Neck + TextEncoder   │
          └─────────────────────────────┘
```

**진입점**: `sam3/model_builder.py`
- `build_sam3_image_model()` → 이미지 전용
- `build_sam3_video_model()` → 비디오 (Detector + Tracker)
- `build_sam3_video_predictor()` → 비디오 텍스트 프롬프트 API

---

## 2. 이미지 모델 흐름

### 2-1. 조립 순서 (model_builder.py)

```python
# 1. Vision Backbone
vit = ViT(img_size=1008, patch_size=14, embed_dim=1024, depth=32)  # vitdet.py
neck = Sam3DualViTDetNeck(trunk=vit, scale_factors=[4,2,1,0.5])    # necks.py
# → ViT 출력을 4단계 FPN으로 변환. SAM3용(고해상도)과 SAM2용(tracker용) 두 갈래 생성

# 2. Text Encoder
tokenizer = SimpleTokenizer(bpe_path=...)         # tokenizer_ve.py
text_enc = VETextEncoder(tokenizer, d_model=256)  # text_encoder_ve.py
# → CLIP 스타일. 텍스트를 (seq_len, batch, 256) 토큰 임베딩으로 변환

# 3. Vision-Language Backbone 결합
backbone = SAM3VLBackbone(visual=neck, text=text_enc)  # vl_combiner.py
# → forward_image()와 forward_text()를 각각 분리 호출 가능

# 4. Geometry Encoder (박스/포인트 프롬프트 인코딩)
geo_encoder = SequenceGeometryEncoder(d_model=256, num_layers=3)  # geometry_encoders.py
# → 입력 박스/포인트를 이미지 feature와 cross-attention하여 256d 임베딩으로 변환

# 5. Transformer (Encoder + Decoder)
encoder = TransformerEncoderFusion(num_layers=6)   # encoder.py
decoder = TransformerDecoder(num_layers=6, num_queries=200)  # decoder.py
transformer = TransformerWrapper(encoder, decoder)  # model_misc.py

# 6. Scoring & Segmentation Head
dot_scorer = DotProductScoring(d_model=256)        # model_misc.py
seg_head = UniversalSegmentationHead(              # maskformer_segmentation.py
    pixel_decoder=PixelDecoder(num_upsampling_stages=3)
)

# 7. 최종 모델 조립
model = Sam3Image(
    backbone=backbone,
    transformer=transformer,
    input_geometry_encoder=geo_encoder,
    segmentation_head=seg_head,
    dot_prod_scoring=dot_scorer,
)
```

### 2-2. Forward 흐름 (sam3_image.py)

```python
def forward(self, input: BatchedDatapoint):
    # Step 1: 이미지 인코딩 (vitdet.py → necks.py → vl_combiner.py)
    backbone_out = self.backbone.forward_image(input.img_batch)
    # backbone_out = {
    #   "backbone_fpn": [feat_4x, feat_2x, feat_1x, feat_0.5x],  # SAM3용
    #   "vision_pos_enc": [pos_4x, ...],
    #   "sam2_backbone_out": {...}  # Tracker에 넘겨주는 SAM2 호환 feature
    # }

    # Step 2: 텍스트 인코딩 (text_encoder_ve.py)
    text_out = self.backbone.forward_text(input.find_text_batch)
    # text_out = {
    #   "language_features": (seq_len, num_prompts, 256),
    #   "language_mask": (num_prompts, seq_len)
    # }

    # Step 3: 프롬프트 인코딩 (_encode_prompt)
    #   - 텍스트 feature + 박스/포인트 geometry feature를 합쳐 prompt tensor 생성
    #   - geometry_encoder (geometry_encoders.py): 박스/포인트 → 256d
    prompt, prompt_mask, _ = self._encode_prompt(backbone_out, find_input, geometric_prompt)
    # prompt: (prompt_seq_len, batch, 256)  [txt_feats | geo_feats]

    # Step 4: Encoder (encoder.py)
    #   - 이미지 feature와 prompt가 cross-attention
    memory = self.transformer.encoder(
        src=img_feats,          # FPN feature (HW, batch, 256)
        prompt=prompt,          # 텍스트+박스 프롬프트
        prompt_key_padding_mask=prompt_mask,
    )
    # memory["memory"]: 이미지 feature가 프롬프트에 의해 강화됨 (HW, batch, 256)

    # Step 5: Decoder (decoder.py)
    #   - 200개의 learnable query가 memory에 cross-attention
    #   - 각 query → bbox + presence logit 예측
    hs, reference_boxes, dec_presence_out = self.transformer.decoder(
        tgt=query_embed,        # (200, batch, 256) learnable queries
        memory=memory,          # encoder 출력
        memory_text=prompt,     # 텍스트 feature (text cross-attention용)
    )
    # hs: (num_layers=6, 200, batch, 256) 각 decoder layer 출력
    # reference_boxes: (num_layers, 200, batch, 4) cxcywh 형식

    # Step 6: Scoring (model_misc.py DotProductScoring)
    #   - query 임베딩과 prompt 임베딩의 dot product → 매칭 점수
    scores = self.dot_prod_scoring(hs, prompt, prompt_mask)
    # scores: (batch, 200, 1) 각 query가 프롬프트와 얼마나 매칭되는지

    # Step 7: Segmentation Head (maskformer_segmentation.py)
    #   - FPN feature + query → mask 예측
    #   - PixelDecoder: FPN feature를 3번 upsample (1/14 → 1/4 해상도)
    #   - 각 query와 pixel feature의 dot product → binary mask
    seg_out = self.segmentation_head(
        backbone_feats=backbone_out["backbone_fpn"],
        obj_queries=hs,
        encoder_hidden_states=memory,
        prompt=prompt,
    )
    # seg_out["pred_masks"]: (batch, 200, H/4, W/4)

    # 최종 출력
    # out = {
    #   "pred_logits":    (batch, 200, 1)        # 매칭 점수
    #   "pred_boxes":     (batch, 200, 4)         # cxcywh
    #   "pred_boxes_xyxy":(batch, 200, 4)         # xyxy
    #   "pred_masks":     (batch, 200, H/4, W/4)  # binary mask
    #   "presence_logit_dec": (batch, 200, 1)     # presence token 점수
    # }
```

### 2-3. 후처리 (eval/postprocessors.py)

```python
postprocessor = PostProcessImage(
    detection_threshold=0.5,   # 이 이상인 query만 살림
    iou_type="segm",
    use_original_sizes_mask=True,
    convert_mask_to_rle=False,
)

results = postprocessor.process_results(model_output, batch.find_metadatas)
# results[query_id] = {
#   "masks": (N, H, W) bool,   # 원본 해상도로 resize된 마스크
#   "boxes": (N, 4),           # xyxy
#   "scores": (N,)
# }
```

---

## 3. 비디오 모델 흐름

### 3-1. 두 모듈의 역할 분리

```
비디오 입력 (프레임 시퀀스)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Sam3VideoInferenceWithInstanceInteractivity         │  sam3_video_inference.py
│                                                     │
│  매 프레임마다:                                       │
│                                                     │
│  1. Detector (Sam3ImageOnVideoMultiGPU)              │
│     - 이미지 모델과 동일한 구조                        │
│     - 현재 프레임에서 새 객체 detection                │
│     - 텍스트/박스 프롬프트로 구동                       │
│                                                     │
│  2. Tracker (Sam3TrackerPredictor)                   │
│     - SAM2 스타일 메모리 기반 tracking                 │
│     - Detector의 detection 결과를 초기 마스크로 받음    │
│     - 이전 프레임 메모리(7프레임)로 현재 마스크 예측     │
│                                                     │
│  3. 연결 로직                                        │
│     - Detector output → NMS → Tracker에 할당          │
│     - 새 객체: Tracker에 add_new_mask()               │
│     - 기존 객체: Tracker가 memory로 predict            │
└─────────────────────────────────────────────────────┘
```

### 3-2. Tracker 내부 구조 (sam3_tracker_base.py)

```
Sam3TrackerBase
│
├── backbone: SAM3VLBackbone       # Detector와 공유 (ViT feature 재사용)
│     └── Sam3DualViTDetNeck       # SAM2 호환 feature (3레벨 FPN) 제공
│
├── transformer: TransformerWrapper  # SAM2 스타일 memory attention
│     └── TransformerEncoderCrossAttention (4 layers)
│           - self_attention: RoPEAttention  (sam/transformer.py)
│           - cross_attention: RoPEAttention (kv_in_dim=64, 메모리 읽기)
│
├── maskmem_backbone: SimpleMaskEncoder  # memory.py
│     ├── SimpleMaskDownSampler   # 이전 마스크를 feature map 크기로 downsample
│     └── SimpleFuser (CXBlock x2) # mask + image feature 융합 → 64d memory
│
└── SAM decoder (sam/ 디렉토리)
      ├── PromptEncoder   # point/box → sparse/dense embedding (sam1 호환)
      └── MaskDecoder     # memory attention 출력 → binary mask
```

### 3-3. Tracker Forward 흐름 (한 프레임 처리)

```python
# 매 프레임 처리 시 (sam3_tracker_base.py 내부)

# Step 1: 현재 프레임 image feature 추출
#   - backbone이 있으면: backbone.forward_image(current_frame)
#   - 없으면: Detector가 미리 계산한 feature 재사용 (sam2_backbone_out)
current_vision_feats = backbone(current_frame)
# shape: (HW, batch, 256), SAM2 호환 3레벨 FPN

# Step 2: 이전 프레임 메모리 읽기
#   - num_maskmem=7: 최근 6프레임 + conditioning 프레임 1개
#   - maskmem_backbone으로 인코딩된 이전 마스크들 (64d)
memory_bank = select_closest_cond_frames(inference_state, current_frame_idx)

# Step 3: Memory Transformer (4 layer cross-attention)
#   - 현재 frame feature가 query
#   - 과거 memory가 key/value
#   - RoPEAttention 사용 (회전 위치 인코딩)
current_out = self.transformer.encoder(
    src=[current_vision_feats],
    memory_bank=memory_bank,    # (num_maskmem * HW, batch, 64)
)

# Step 4: SAM Decoder
#   - prompt_encoder: 포인트/박스 → embedding (초기 프레임에만 사용)
#   - mask_decoder: current_out + prompt → mask 예측
#   - multimask_output=True이면 3개 mask 후보 생성 → 가장 좋은 것 선택
masks, iou_predictions = self.sam_mask_decoder(
    image_embeddings=current_out,
    prompt_embeddings=prompt,   # 초기 프레임의 포인트/박스
    multimask_output=(frame_idx == ann_frame_idx),
)

# Step 5: 현재 프레임 결과를 메모리로 저장
#   - maskmem_backbone: mask + image feature → 64d memory feature
memory_feature = self.maskmem_backbone(
    pix_feat=current_vision_feats,
    mask=masks,                 # 현재 예측 마스크
)
# 다음 프레임에서 이 memory_feature를 memory_bank에 추가
```

### 3-4. Detector-Tracker 연결 (sam3_video_inference.py)

```python
# Sam3VideoInferenceWithInstanceInteractivity 내부 로직 (단순화)

for frame_idx in range(num_frames):
    # 1. Detector: 현재 프레임에서 detection
    det_out = detector.forward_video_grounding_multigpu(
        backbone_out=backbone_out,
        find_inputs=find_inputs,
        geometric_prompt=text_or_box_prompt,
        frame_idx=frame_idx,
    )
    # det_out: {"pred_logits", "pred_boxes_xyxy", "pred_masks"}

    # 2. NMS 적용
    keep = nms_masks(det_out["pred_masks"], iou_threshold=0.1)

    # 3. 기존 tracked objects와 매칭
    #    - IoU 기반 bipartite matching
    #    - 매칭된 것: Tracker가 이미 tracking 중 → 점수만 업데이트
    #    - 새로운 것 (new_det_thresh=0.7 이상): Tracker에 새 객체로 등록
    new_dets = det_out[~matched]
    for det in new_dets:
        if det["score"] > new_det_thresh:
            tracker.add_new_mask(
                inference_state=tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                mask=det["mask"],
            )

    # 4. Tracker: 모든 registered objects의 마스크 예측
    for frame_idx, obj_ids, masks in tracker.propagate_in_video(
        tracker_state, start_frame_idx=frame_idx
    ):
        # masks: 이전 메모리 기반으로 예측된 현재 프레임 마스크
        yield frame_idx, obj_ids, masks
```

---

## 4. 파일별 상세 설명

### vitdet.py — ViT Backbone

```python
class ViT(nn.Module):
    # img_size=1008, patch_size=14 → 72x72 패치
    # embed_dim=1024, depth=32, num_heads=16
    # global_att_blocks=(7,15,23,31): 4개 레이어는 전체 attention
    # 나머지는 window_size=24 local attention (메모리 효율)
    # RoPE (Rotary Position Embedding) 사용
    # 출력: (batch, 72, 72, 1024)
```

### necks.py — FPN Neck

```python
class Sam3DualViTDetNeck(nn.Module):
    # ViT 출력(72x72, 1024d)을 두 갈래 FPN으로 변환
    #
    # SAM3 FPN (Detector용):
    #   scale_factors=[4,2,1,0.5]
    #   → (288x288, 256), (144x144, 256), (72x72, 256), (36x36, 256)
    #
    # SAM2 FPN (Tracker용, add_sam2_neck=True일 때):
    #   → (288x288, 32), (144x144, 64), (72x72, 256) — SAM2 호환 크기
    #
    # 두 FPN이 ViT feature를 공유하므로 backbone forward는 1번만 실행
    pass
```

### vl_combiner.py — Vision-Language Backbone

```python
class SAM3VLBackbone(nn.Module):
    def forward_image(self, samples):
        # vitdet + neck 실행
        # 반환: {"backbone_fpn", "vision_pos_enc", "sam2_backbone_out"}
        pass

    def forward_text(self, captions, device="cuda"):
        # text_encoder_ve 실행
        # 반환: {"language_features", "language_mask"}
        # language_features: (seq_len, num_captions, 256)
        pass
```

### geometry_encoders.py — 박스/포인트 인코딩

```python
class SequenceGeometryEncoder(nn.Module):
    # 입력: 박스 (N, 4) 또는 포인트 (N, 2)
    # 1. 박스/포인트를 sine position encoding → 256d
    # 2. 이미지 feature와 3-layer cross-attention
    #    (박스가 이미지의 어느 영역을 참조하는지 학습)
    # 3. 출력: (N, batch, 256) geometry embedding
    # → Decoder의 prompt로 텍스트와 concatenate됨
    pass
```

### encoder.py — Transformer Encoder

```python
class TransformerEncoderFusion(nn.Module):
    # 6 layer cross-attention encoder
    # src: 이미지 feature (HW, batch, 256)
    # prompt: 텍스트 + 박스 feature (prompt_len, batch, 256)
    #
    # 각 layer:
    #   1. self_attention: 이미지 feature끼리 attention
    #   2. cross_attention: 이미지 feature → prompt (텍스트로 이미지 강화)
    # 출력: "memory" = 프롬프트에 의해 강화된 이미지 feature
    pass
```

### decoder.py — Transformer Decoder

```python
class TransformerDecoder(nn.Module):
    # num_queries=200: 200개의 learnable object query
    # 6 layer decoder
    # presence_token=True: 각 query가 "이 프롬프트가 이미지에 있는가" 예측
    #
    # 각 layer:
    #   1. self_attention: 200 query끼리 attention
    #   2. cross_attention: query → memory (이미지에서 정보 읽기)
    #   3. text_cross_attention: query → prompt (텍스트로 query 강화)
    #   4. bbox_embed MLP: query → bbox delta (iterative refinement)
    #
    # dac=True: Decoder-only Attention Consistency 적용 (학습 안정화)
    # box_refine=True: 각 layer마다 bbox를 refinement
    # 출력:
    #   hs: (6, batch, 200, 256)  각 layer의 query 상태
    #   reference_boxes: (6, batch, 200, 4)  각 layer의 bbox 예측
    #   dec_presence_out: (6, batch, 200, 1)  presence logit
    pass
```

### model_misc.py — 보조 모듈

```python
class DotProductScoring(nn.Module):
    # query 임베딩과 prompt(텍스트) 임베딩의 유사도 계산
    # 1. prompt를 MLP로 투영
    # 2. query와 prompt의 dot product 평균
    # → "이 query가 이 텍스트 프롬프트에 해당하는가" 점수
    pass

class TransformerWrapper(nn.Module):
    # encoder + decoder를 하나로 묶는 wrapper
    # d_model=256
    pass

class MLP(nn.Module):
    # 범용 MLP (잔차 연결, LayerNorm 옵션)
    pass
```

### maskformer_segmentation.py — Segmentation Head

```python
class PixelDecoder(nn.Module):
    # FPN feature를 3번 upsample (nearest interpolation)
    # 72x72 → 144x144 → 288x288  (1/14 → 1/4 해상도)
    # 각 level에서 conv + feature 융합
    # 출력: (batch, 256, H/4, W/4) = pixel feature

class UniversalSegmentationHead(nn.Module):
    # 1. PixelDecoder로 pixel feature 생성
    # 2. prompt와 query를 cross-attention (prompt의 정보를 query에 반영)
    # 3. 각 query와 pixel feature의 dot product → mask logit
    #    mask = einsum(query, pixel_feature) → (batch, num_queries, H/4, W/4)
    # 출력: {"pred_masks": (batch, 200, H/4, W/4)}
    pass
```

### memory.py — 비디오 메모리 인코딩

```python
class SimpleMaskEncoder(nn.Module):
    # Tracker용: 이전 프레임 마스크를 memory feature로 변환
    #
    # 1. SimpleMaskDownSampler: 마스크를 72x72로 downsample
    # 2. 이미지 feature와 마스크를 concatenate
    # 3. SimpleFuser (CXBlock x2): ConvNext 스타일 depthwise conv
    #    → (batch, 64, 72, 72) memory feature
    # 64d로 압축 (256d 이미지 feature보다 작아 메모리 효율적)
    pass

class CXBlock(nn.Module):
    # ConvNeXt 스타일 블록
    # depthwise conv (kernel=7) + LayerNorm + FFN
    pass
```

### sam3_tracker_base.py — Tracker 핵심

```python
class Sam3TrackerBase(nn.Module):
    # 주요 컴포넌트:
    # - backbone: ViT feature 추출 (Detector와 공유)
    # - transformer: 4-layer memory cross-attention (RoPEAttention)
    # - maskmem_backbone: 마스크 → 64d memory
    # - sam_prompt_encoder: PromptEncoder (sam/prompt_encoder.py)
    # - sam_mask_decoder: MaskDecoder (sam/mask_decoder.py)
    #
    # 핵심 메서드:
    def _run_single_frame_inference(self, current_vision_feats, memory_bank, prompt):
        # 1. memory cross-attention으로 current feature 강화
        # 2. SAM decoder로 mask 예측
        # 3. 결과 mask를 maskmem_backbone으로 인코딩 → 메모리 저장
        pass
    pass
```

### sam3_tracking_predictor.py — SAM2 호환 API

```python
class Sam3TrackerPredictor(Sam3TrackerBase):
    # SAM2의 SAM2VideoPredictor와 동일한 API 제공

    def init_state(self, video_path):
        # 비디오 로드, inference_state 초기화
        pass

    def add_new_points(self, inference_state, frame_idx, obj_id, points, labels):
        # 특정 프레임에 포인트 프롬프트 추가
        # → SAM decoder로 즉시 마스크 예측 후 반환
        pass

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        # GT 마스크를 초기 프롬프트로 추가 (VOS 평가용)
        pass

    def propagate_in_video(self, inference_state, start_frame_idx, max_frame_num_to_track):
        # 프레임 순서대로 tracking 실행
        # yield (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)
        pass
```

### sam3_video_inference.py — 비디오 전체 추론

```python
class Sam3VideoInference(Sam3VideoBase):
    # 텍스트 프롬프트 기반 비디오 추론
    # Detector + Tracker 연결 로직 포함

    def init_state(self, resource_path):
        # 비디오(mp4 or JPEG 폴더) 로드
        # 모든 프레임을 1008x1008로 resize, normalize
        pass

    def handle_request(self, request):
        # request["type"] == "start_session": 비디오 로드
        # request["type"] == "add_prompt": 텍스트 프롬프트 설정
        # request["type"] == "propagate": 전체 비디오 추론
        pass

class Sam3VideoInferenceWithInstanceInteractivity(Sam3VideoInference):
    # Detector + Tracker 통합 버전
    # detector: Sam3ImageOnVideoMultiGPU (매 프레임 detection)
    # tracker: Sam3TrackerPredictor (메모리 기반 tracking)
    #
    # 주요 하이퍼파라미터:
    # - score_threshold_detection=0.5: detection 점수 threshold
    # - new_det_thresh=0.7: 새 객체 등록 threshold
    # - hotstart_delay=15: 초기 15프레임은 안정화 기간
    # - recondition_every_nth_frame=16: 16프레임마다 detector로 재확인
    # - max_trk_keep_alive=30: 30프레임 동안 미검출 시 객체 제거
    pass
```

### sam1_task_predictor.py — SAM1 호환

```python
class SAM3InteractiveImagePredictor:
    # SAM1/SAM2의 SamPredictor와 동일한 API
    # SAM3의 Tracker(SAM decoder 부분)를 이미지에 적용
    #
    # 사용법:
    # predictor.set_image(image)
    # masks, scores, logits = predictor.predict(
    #     point_coords=[[x, y]],
    #     point_labels=[1],   # 1=positive, 0=negative
    # )
    # → SAM3의 ViT backbone + SAM decoder로 단일 이미지 interactive segmentation
    pass
```

### sam/ 디렉토리 — SAM1/SAM2 호환 컴포넌트

```python
# sam/prompt_encoder.py
class PromptEncoder:
    # 포인트/박스/마스크 → sparse/dense embedding
    # Tracker의 초기 프레임 프롬프트 인코딩에 사용

# sam/mask_decoder.py
class MaskDecoder:
    # Tracker의 최종 mask 예측
    # image_embeddings + prompt → (3개 mask 후보, IoU 점수)
    # dynamic_multimask_via_stability: 안정성 기준으로 최적 mask 자동 선택

# sam/transformer.py
class RoPEAttention:
    # Rotary Position Embedding을 사용하는 attention
    # Tracker transformer에서 사용
    # feat_sizes=[72,72]: 2D 공간 위치 정보를 attention에 인코딩

# sam/common.py
class LayerNorm2d:
    # 2D feature map용 LayerNorm
```

---

## 요약: 데이터 shape 흐름 (이미지 모델 기준)

```
입력 이미지: (batch, 3, 1008, 1008)
    │
    ▼ ViT (vitdet.py)
(batch, 72, 72, 1024)  [72 = 1008/14]
    │
    ▼ FPN Neck (necks.py)
backbone_fpn = [
    (batch, 256, 288, 288),  # x4
    (batch, 256, 144, 144),  # x2
    (batch, 256,  72,  72),  # x1  ← num_feature_levels=1이면 이것만 사용
    (batch, 256,  36,  36),  # x0.5
]
    │ flatten + permute
    ▼ (HW=72*72=5184, batch, 256)

입력 텍스트: ["cat", "dog", ...]  (num_prompts,)
    │
    ▼ Text Encoder (text_encoder_ve.py)
(seq_len=77, num_prompts, 256)

입력 박스: (N_boxes, 4)
    │
    ▼ Geometry Encoder (geometry_encoders.py)
(N_boxes, batch, 256)

    ▼ [txt_feats | geo_feats] concatenate
prompt: (prompt_len, batch, 256)

    ▼ Encoder (encoder.py) — 6 layers
memory: (5184, batch, 256)  # 이미지 feature + 프롬프트 정보 융합

    ▼ Decoder (decoder.py) — 200 queries, 6 layers
hs: (6, batch, 200, 256)
boxes: (6, batch, 200, 4)
presence: (6, batch, 200, 1)

    ▼ DotProductScoring
scores: (batch, 200, 1)

    ▼ PixelDecoder (maskformer_segmentation.py) — 3 upsamples
pixel_feat: (batch, 256, 252, 252)  [≈ 1008/4]

    ▼ Mask Head: einsum(query, pixel_feat)
masks: (batch, 200, 252, 252)

    ▼ PostProcessImage — threshold=0.5, resize to original
결과: N개의 (mask, box, score)  [200개 중 살아남은 것들]
```
