#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR Pose Training Script
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Pose models on COCO 2017 person keypoints.

Supports both RFDETRPoseSmall (512px, 32M params) and RFDETRPoseLarge (704px).
Uses transfer learning from pretrained RF-DETR detection weights — only the
keypoint head (3-layer MLP → 17×3 outputs) trains from scratch.

Usage:
    # Small model (default)
    python train_pose.py

    # Large model
    python train_pose.py --large

    # Custom settings
    python train_pose.py --epochs 100 --batch-size 16 --lr 5e-5
"""

import argparse
import sys

from rfdetr import RFDETRPoseLarge, RFDETRPoseSmall


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pose training."""
    parser = argparse.ArgumentParser(
        description="Train RF-DETR Pose on COCO 2017 person keypoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model variant
    parser.add_argument(
        "--large", action="store_true",
        help="Use RFDETRPoseLarge (704px, 4 decoder layers) instead of Small (512px, 3 decoder layers)",
    )

    # Dataset
    parser.add_argument(
        "--dataset-dir", type=str, default="/root/work/datasets/coco",
        help="Path to COCO 2017 dataset (must contain train2017/, val2017/, annotations/)",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder learning rate")
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4, help="Backbone learning rate")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")

    # Learning rate schedule
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=["cosine", "step"],
                        help="LR scheduler type")
    parser.add_argument("--lr-min-factor", type=float, default=0.0,
                        help="Minimum LR multiplier for cosine schedule (0.0 = decay to zero)")

    # Keypoint loss coefficients
    parser.add_argument("--kpt-l1-coef", type=float, default=5.0, help="Keypoint L1 loss weight")
    parser.add_argument("--kpt-vis-coef", type=float, default=1.0, help="Keypoint visibility BCE loss weight")
    parser.add_argument(
        "--kpt-oks-coef", type=float, default=0.0,
        help="Keypoint OKS (Object Keypoint Similarity) loss weight. 0.0 disables. "
             "OKS aligns with COCO eval metric and gives strong fine-localization gradients.",
    )

    # Output
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (auto-set if omitted)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


def main() -> None:
    """Run pose training."""
    args = parse_args()

    # Select model variant
    model_cls = RFDETRPoseLarge if args.large else RFDETRPoseSmall
    variant = "large" if args.large else "small"
    print(f"Training RF-DETR Pose ({variant}) on COCO 2017 person keypoints")

    model = model_cls()

    # Auto-set output directory
    output_dir = args.output_dir or f"output_pose_{variant}"

    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        dataset_file="coco",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        grad_accum_steps=args.grad_accum_steps,
        lr_scheduler=args.lr_scheduler,
        lr_min_factor=args.lr_min_factor,
        keypoint_l1_loss_coef=args.kpt_l1_coef,
        keypoint_vis_loss_coef=args.kpt_vis_coef,
        keypoint_oks_loss_coef=args.kpt_oks_coef,
        output_dir=output_dir,
        progress_bar=True,
        num_workers=args.num_workers,
    )

    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)


if __name__ == "__main__":
    sys.exit(main() or 0)
