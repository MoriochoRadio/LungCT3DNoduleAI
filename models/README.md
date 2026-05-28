# Models

본 디렉토리는 학부 종합설계 (T.O.P 팀, 2025-1학기) 의 학습된 AI 모델 가중치 파일들입니다.

## 파일 목록

| 파일 | 크기 | 모델 | 용도 |
|---|---|---|---|
| `dhkstjd.pth` | 52 MB (Git LFS) | `HybridSpiralNetPointNetTransformer` | 메인 결절 탐지 (vertex별 결절 확률) |
| `classification_v.1.0.pth` | 84 KB | `MeshCNN` | 결절 유무 분류 (보조, mesh edge 기반) |
| `spiral_9.npy` | 96 KB | spiral index (`(V, K=9)`) | SpiralConv 입력, 학습 시 동일하게 사용 |

## ⚠ `dhkstjd.pth` 파일명에 대하여

`dhkstjd.pth` 의 `dhkstjd` 는 *"완성"* 을 입력하려다 한글 IME 를 안 켠 채로 영문 키를 눌러 생긴 학부생 학습 흔적입니다. 정식 명명으로 변경하지 않고 그대로 보존했습니다 — 본 레포 README 의 *"본인 톤이 살아있는 코드"* 섹션 참조.

## 학습 정보

- **데이터셋**: TCIA (The Cancer Imaging Archive) 공개 폐 CT-DICOM (LIDC-IDRI)
- **분할 비율**: train : validation : test = 8 : 1 : 1
- **학습 환경**: Google Colab (`notebooks/Untitled0.ipynb`)
- **에폭**: 46+ (학습 진화 흔적은 `../notebooks/training_log.txt` 참조)
- **최고 F1**: 약 0.78
- **테스트 결과**: 234건 중 171건 정확 (73.1%), 양성 재현율 90.9% (160/176), 음성 재현율 19.0% (11/58)
  - ⚠ *"데이터 불균형 개선 필요"* (본인 정직 평가, 최종발표 PPT Slide 15)

## 모델 구조 (`HybridSpiralNetPointNetTransformer`)

3 branch 앙상블:
1. **`ImprovedSpiralNet`** — multi-scale SpiralConv encoder-decoder + SE block + skip connection
2. **PointNet MLP** — local feature → global pooling
3. **`SimplePointTransformerBlock`** — multi-head self-attention (heads=4)

Loss: **`HybridLoss`** = AdaptiveFocalLoss (alpha 동적, gamma=2.0) + Dice Loss + Hard Negative Mining (ratio=3.0).

입력: vertex features `(B, V=2674, C=7)` — coords + dist_center + zscore
출력: vertex별 결절 확률 `(B, V)`

## 모델 가중치 다운로드 (Git LFS)

`dhkstjd.pth` 는 Git LFS 로 트래킹됩니다. 클론 후 자동 다운로드되지 않으면:
```bash
git lfs pull
```

## 데이터셋 다운로드

학습·시험 데이터는 별도 다운로드가 필요합니다 (의료 영상 GB 단위로 본 레포에 포함 안 됨):
- LIDC-IDRI: https://www.cancerimagingarchive.net/collection/lidc-idri/
- 본인 팀의 전처리 파이프라인 (`../src/test2.py` 의 `find_dicom_files`, `load_scan`, `get_pixels_hu`, `resample`, `segment_lung_mask`, `mesh_load_scan` 등) 으로 DICOM → vertex `.npy` 변환
