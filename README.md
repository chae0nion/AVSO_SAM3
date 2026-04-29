# AVSO SAM3

SAM3 기반 occlusion-robust video object segmentation. 시각적으로 유사한 객체가 혼재하는 환경에서 detector-tracker 파이프라인을 통해 안정적인 추적을 수행합니다.

## LVOS Evaluation Results

Evaluation on the LVOS validation set using the DAVIS J&F protocol. J = region similarity (Jaccard), F = boundary accuracy (higher is better).

|  | LVOSv1 J&F | LVOSv1 J | LVOSv1 F | LVOSv2 J&F | LVOSv2 J | LVOSv2 F |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| SAM3 | 84.0 | 78.5 | 89.4 | 88.5 | 84.7 | 92.3 |
| **Ours** | **86.8** | **81.2** | **92.4** | **89.2** | **85.4** | **93.1** |
