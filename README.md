# RF-DETR Pose: Real-Time Keypoint Estimation

[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![python-version](https://img.shields.io/pypi/pyversions/rfdetr)](https://badge.fury.io/py/rfdetr)

RF-DETR Pose extends [RF-DETR](https://github.com/roboflow/rf-detr) with human pose estimation (keypoint detection). Built on the same DINOv2 + deformable DETR backbone, it adds a spatial keypoint head that produces per-keypoint heatmaps via cross-attention between backbone features and decoder queries, then extracts coordinates via differentiable soft-argmax.

Designed for surveillance use cases: fixed IP cameras, behavior detection (hands up, weapon holding, fighting).

## Architecture

```
Backbone (DINOv2)                     Decoder (MSDeformAttn)
      |                                       |
  (B, 256, 32, 32)                    (L, B, Q, 256) query features
      |                                       |
 DepthwiseConvBlock x L               MLPBlock (residual)
 (refine at 32x32)                          |
      |                               Linear(256 -> K*64)
 F.interpolate (bilinear)                   |
 32x32 -> 64x64                      Reshape -> (B, Q, K, 64)
      |                                       |
 Conv1x1(256 -> 64)                           |
      |                                       |
  (B, 64, 64, 64)                    (B, Q, K, 64)
       \                                    /
        ---  einsum("bchw,bnkc->bnkhw")  ---
                        |
              (B, Q, K, 64, 64)  per-keypoint heatmaps
                        |
                  soft-argmax 2D (differentiable)
                        |
                  (B, Q, K, 2)  xy coords in [0,1]
                        +
              vis from Linear(256 -> K) on query features
                        |
                  (B, Q, K, 3)  final output (x, y, visibility)
```

**Key design decisions:**

- **Spatial cross-attention** via `einsum` (same pattern as RF-DETR segmentation head)
- **64x64 heatmaps** via bilinear upsampling from native 32x32 backbone resolution (`downsample_ratio=8`)
- **Soft-argmax** with learnable temperature for differentiable coordinate extraction
- **Visibility** predicted from query features only (no spatial features needed)
- **Conv blocks at 32x32**, upsampling only before the projection/einsum (memory efficient)
- **Chunked heatmap computation** to bound peak GPU memory during training
- **Transfer learning**: backbone + detector pretrained on COCO, only keypoint head trains from scratch

## Model Variants

| Variant | Class | Resolution | Decoder Layers | Params | Base Detector |
|:-------:|:-----:|:----------:|:--------------:|:------:|:-------------:|
| Small | `RFDETRPoseSmall` | 512x512 | 3 | ~33M | RFDETRSmall |
| Large | `RFDETRPoseLarge` | 704x704 | 4 | ~35M | RFDETRLarge |

## Install

```bash
# From source
pip install -e ".[dev]"

# Or with uv
uv sync --all-groups
```

## Dataset

Download COCO 2017 with person keypoint annotations:

```
datasets/coco/
  train2017/
  val2017/
  annotations/
    person_keypoints_train2017.json
    person_keypoints_val2017.json
```

## Training

### Single GPU

```bash
# Small model, 100 epochs, cosine LR schedule
python train_pose.py \
    --epochs 100 \
    --batch-size 8 \
    --lr-scheduler cosine \
    --output-dir output_pose_small

# Large model
python train_pose.py --large \
    --epochs 100 \
    --batch-size 4 \
    --lr-scheduler cosine \
    --output-dir output_pose_large
```

### Multi-GPU (DDP)

```bash
# 2x GPUs
torchrun --nproc_per_node=2 train_pose.py \
    --epochs 100 \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --lr-scheduler cosine \
    --output-dir output_pose_ddp
```

### Memory-Constrained GPU (16GB)

For GPUs with 16GB VRAM (RTX 4080, RTX 5070 Ti, etc.), use gradient accumulation:

```bash
python train_pose.py \
    --epochs 100 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --lr-scheduler cosine \
    --output-dir output_pose_small
```

> **Note:** The 64x64 heatmap upsampling requires >16GB VRAM during training. On 16GB GPUs, set `kpt_downsample_ratio=16` in config to use native 32x32 heatmaps. Full 64x64 resolution requires 24GB+ (RTX 3090, A100, H100).

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--large` | off | Use RFDETRPoseLarge instead of Small |
| `--dataset-dir` | `/root/work/datasets/coco` | COCO dataset path |
| `--epochs` | 50 | Training epochs |
| `--batch-size` | 8 | Batch size per GPU |
| `--lr` | 1e-4 | Decoder learning rate |
| `--lr-encoder` | 1.5e-4 | Backbone learning rate |
| `--grad-accum-steps` | 4 | Gradient accumulation steps |
| `--lr-scheduler` | cosine | LR schedule: `cosine` or `step` |
| `--lr-min-factor` | 0.0 | Min LR multiplier for cosine (0 = decay to zero) |
| `--kpt-l1-coef` | 5.0 | Keypoint L1 loss weight |
| `--kpt-vis-coef` | 1.0 | Visibility BCE loss weight |
| `--resume` | none | Checkpoint path to resume from |

## Inference

```python
from rfdetr import RFDETRPoseSmall

model = RFDETRPoseSmall(pretrain_weights="output_pose_small/checkpoint_best_total.pth")

detections = model.predict(image, threshold=0.5)

# detections.xyxy       - bounding boxes (N, 4)
# detections.confidence - detection scores (N,)
# detections.data["keypoints"] - keypoints (N, 17, 3) as x, y, visibility
```

### Drawing Keypoints

```python
import cv2
import numpy as np

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # upper body
    (5, 11), (6, 12), (11, 12),             # torso
    (11, 13), (13, 15), (12, 14), (14, 16), # legs
]

for i in range(len(detections)):
    kp = detections.data["keypoints"][i]  # (17, 3)
    for j, k in SKELETON:
        if kp[j, 2] > 0.5 and kp[k, 2] > 0.5:
            pt1 = tuple(kp[j, :2].astype(int))
            pt2 = tuple(kp[k, :2].astype(int))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
```

## Testing

```bash
# Run all CPU tests
uv run --no-sync pytest src/ tests/ -n 2 -m "not gpu"

# Run keypoint-specific tests
uv run --no-sync pytest tests/models/test_keypoint_head.py -v
uv run --no-sync pytest tests/models/test_keypoint_integration.py -v
uv run --no-sync pytest tests/models/test_pose_config.py -v

# Run GPU tests
uv run --no-sync pytest src/ tests/ -n 2 -m gpu

# Lint
pre-commit run --all-files
```

## Speed Benchmark

```bash
# Compare base detector vs pose model latency
python benchmark_pose_speed.py
python benchmark_pose_speed.py --pose-checkpoint output_pose_small/checkpoint_best_total.pth
```

## Debug Video

```bash
# Generate side-by-side comparison video (base detector vs pose)
python debug_video.py --input video.mp4
python debug_video.py --input video.mp4 --pose-checkpoint output_pose_small/checkpoint_best_total.pth
```

## COCO Keypoints

The model predicts 17 COCO keypoints:

| Index | Keypoint | Index | Keypoint |
|:-----:|:--------:|:-----:|:--------:|
| 0 | nose | 9 | left wrist |
| 1 | left eye | 10 | right wrist |
| 2 | right eye | 11 | left hip |
| 3 | left ear | 12 | right hip |
| 4 | right ear | 13 | left knee |
| 5 | left shoulder | 14 | right knee |
| 6 | right shoulder | 15 | left ankle |
| 7 | left elbow | 16 | right ankle |
| 8 | right elbow | | |

## License

Apache License 2.0. See [LICENSE](LICENSE).

Based on [RF-DETR](https://github.com/roboflow/rf-detr) by [Roboflow](https://roboflow.com).
