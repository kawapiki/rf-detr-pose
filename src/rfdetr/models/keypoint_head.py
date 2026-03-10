# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Keypoint estimation heads for RF-DETR pose models."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.models.segmentation_head import DepthwiseConvBlock, MLPBlock


class KeypointHead(nn.Module):
    """Keypoint prediction head that regresses box-relative offsets and visibility from decoder queries.

    Predicts (x, y, visibility) for each keypoint per query using an MLP. The xy outputs are
    box-relative offsets: ``tanh(raw)`` gives offsets in [-1, 1] that are scaled by the predicted
    box size and shifted by the box center to produce absolute [0, 1] normalized coordinates.

    .. deprecated::
        Use :class:`SpatialKeypointHead` instead for better keypoint AP.

    Args:
        hidden_dim: Dimension of the input query features.
        num_keypoints: Number of keypoints to predict per detection.
        num_layers: Number of MLP layers.
    """

    def __init__(self, hidden_dim: int, num_keypoints: int = 17, num_layers: int = 3) -> None:
        super().__init__()
        from rfdetr.models.lwdetr import MLP

        self.num_keypoints = num_keypoints
        self.mlp = MLP(hidden_dim, hidden_dim, num_keypoints * 3, num_layers)

        # Zero-initialize the output layer (matching bbox_embed init pattern)
        nn.init.constant_(self.mlp.layers[-1].weight.data, 0)
        nn.init.constant_(self.mlp.layers[-1].bias.data, 0)

        self._export = False

    def export(self) -> None:
        """Switch to export mode for ONNX compatibility."""
        self._export = True

    def forward(self, query_features: torch.Tensor, pred_boxes: torch.Tensor) -> torch.Tensor:
        """Predict keypoints from decoder query features and predicted boxes.

        Keypoint xy coordinates are predicted as box-relative offsets: the MLP outputs raw
        values that are passed through ``tanh`` to bound them to [-1, 1], then scaled by the
        predicted box dimensions and offset from the box center.

        Args:
            query_features: Query features of shape ``(..., hidden_dim)``.
            pred_boxes: Predicted boxes in cxcywh format, shape ``(..., 4)``.

        Returns:
            Keypoint predictions of shape ``(..., num_keypoints, 3)`` where the last
            dimension contains ``(x, y, visibility)``. Coordinates are in absolute [0, 1]
            normalized image coordinates. During training, visibility contains raw logits;
            in export mode, visibility is sigmoid-activated.
        """
        raw = self.mlp(query_features)
        # Reshape to (..., num_keypoints, 3)
        shape = raw.shape[:-1] + (self.num_keypoints, 3)
        raw = raw.view(shape)

        # Box-relative offset prediction
        # pred_boxes is (..., 4) in cxcywh format
        box_cx = pred_boxes[..., 0:1]  # (..., 1)
        box_cy = pred_boxes[..., 1:2]  # (..., 1)
        box_w = pred_boxes[..., 2:3]  # (..., 1)
        box_h = pred_boxes[..., 3:4]  # (..., 1)

        # Expand to (..., K, 1) for broadcasting with (..., K, 2)
        box_cx = box_cx.unsqueeze(-2).expand_as(raw[..., 0:1])
        box_cy = box_cy.unsqueeze(-2).expand_as(raw[..., 0:1])
        box_w = box_w.unsqueeze(-2).expand_as(raw[..., 0:1])
        box_h = box_h.unsqueeze(-2).expand_as(raw[..., 0:1])

        # tanh offsets in [-1, 1], scaled by box size, shifted by box center
        offset_x = raw[..., 0:1].tanh() * box_w + box_cx
        offset_y = raw[..., 1:2].tanh() * box_h + box_cy
        xy = torch.cat([offset_x, offset_y], dim=-1).clamp(0.0, 1.0)

        vis = raw[..., 2:3]
        if self._export:
            vis = vis.sigmoid()

        return torch.cat([xy, vis], dim=-1)


# Backward-compatible alias
KeypointHeadMLP = KeypointHead


class SpatialKeypointHead(nn.Module):
    """Spatial keypoint head that uses cross-attention between backbone features and decoder queries.

    Produces per-keypoint heatmaps via einsum between spatial feature projections and
    query projections, then extracts coordinates via differentiable soft-argmax. Follows
    the same architectural pattern as :class:`~rfdetr.models.segmentation_head.SegmentationHead`.

    Args:
        hidden_dim: Dimension of backbone spatial features and decoder query features.
        num_keypoints: Number of keypoints to predict per detection.
        num_blocks: Number of DepthwiseConvBlocks (one per decoder layer).
        kpt_embed_dim: Bottleneck dimension for spatial/query projections.
        downsample_ratio: Factor by which to downsample the image for heatmap resolution.
            Spatial features are upsampled to ``image_size // downsample_ratio``.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_keypoints: int = 17,
        num_blocks: int = 3,
        kpt_embed_dim: int = 64,
        downsample_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kpt_embed_dim = kpt_embed_dim
        self.downsample_ratio = downsample_ratio

        # Spatial feature refinement (one block per decoder layer)
        self.blocks = nn.ModuleList([DepthwiseConvBlock(hidden_dim) for _ in range(num_blocks)])

        # Spatial projection: (B, hidden_dim, H, W) -> (B, kpt_embed_dim, H, W)
        self.spatial_proj = nn.Conv2d(hidden_dim, kpt_embed_dim, kernel_size=1)

        # Query processing
        self.query_block = MLPBlock(hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, num_keypoints * kpt_embed_dim)

        # Visibility head (from query features, no spatial features needed)
        self.vis_head = nn.Linear(hidden_dim, num_keypoints)

        # Learnable temperature for soft-argmax
        self.temperature = nn.Parameter(torch.ones(1))

        self._export = False

    def export(self) -> None:
        """Switch to export mode for ONNX compatibility."""
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if hasattr(m, "export") and isinstance(m.export, Callable) and hasattr(m, "_export") and not m._export:
                m.export()

    def _ensure_grid(self, H: int, W: int, device: torch.device) -> None:
        """Lazily create coordinate grid buffers matching spatial resolution."""
        if hasattr(self, "_grid_h") and self._grid_h == H and self._grid_w == W and self._grid_x.device == device:
            return

        self._grid_h = H
        self._grid_w = W

        # Grid coordinates normalized to [0, 1], centered on each cell
        gy = torch.arange(H, device=device, dtype=torch.float32)
        gx = torch.arange(W, device=device, dtype=torch.float32)
        gy = (gy + 0.5) / H
        gx = (gx + 0.5) / W
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")

        # Flatten to (H*W,) for einsum with softmax weights
        self._grid_x = grid_x.reshape(-1)
        self._grid_y = grid_y.reshape(-1)

    def _upsample_spatial(self, spatial: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        """Upsample spatial features to target resolution.

        Args:
            spatial: Backbone features of shape ``(B, C, H, W)``.
            image_size: Original image size ``(H, W)``.

        Returns:
            Upsampled features of shape ``(B, C, H', W')`` where
            ``H' = image_size[0] // downsample_ratio``.
        """
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        if spatial.shape[-2:] == target_size:
            return spatial
        return F.interpolate(spatial, size=target_size, mode="bilinear", align_corners=False)

    def _soft_argmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Extract coordinates from heatmaps via differentiable soft-argmax.

        Args:
            heatmaps: Per-keypoint heatmaps of shape ``(B, Q, K, H, W)``.

        Returns:
            Coordinates of shape ``(B, Q, K, 2)`` in [0, 1] normalized space.
        """
        B, Q, K, H, W = heatmaps.shape
        flat = heatmaps.reshape(B, Q, K, -1)  # (B, Q, K, H*W)
        weights = F.softmax(flat * self.temperature.abs().clamp(min=0.01), dim=-1)

        self._ensure_grid(H, W, heatmaps.device)

        x = (weights * self._grid_x).sum(-1)  # (B, Q, K)
        y = (weights * self._grid_y).sum(-1)  # (B, Q, K)
        return torch.stack([x, y], dim=-1)  # (B, Q, K, 2)

    def _chunked_heatmap_to_coords(self, sp: torch.Tensor, qf_proj: torch.Tensor, chunk_size: int = 50) -> torch.Tensor:
        """Compute heatmaps and extract coordinates in chunks to bound peak memory.

        Instead of materializing the full ``(B, Q, K, H, W)`` heatmap tensor at once,
        processes queries in chunks of ``chunk_size`` and immediately applies soft-argmax.

        Args:
            sp: Projected spatial features of shape ``(B, C, H, W)``.
            qf_proj: Projected query features of shape ``(B, Q, K, C)``.
            chunk_size: Number of queries to process at once.

        Returns:
            Coordinates of shape ``(B, Q, K, 2)`` in [0, 1] normalized space.
        """
        B, Q, K, C = qf_proj.shape
        if Q <= chunk_size:
            heatmaps = torch.einsum("bchw,bnkc->bnkhw", sp, qf_proj)
            return self._soft_argmax(heatmaps)

        xy_chunks = []
        for start in range(0, Q, chunk_size):
            end = min(start + chunk_size, Q)
            qf_chunk = qf_proj[:, start:end]  # (B, chunk, K, C)
            heatmaps = torch.einsum("bchw,bnkc->bnkhw", sp, qf_chunk)
            xy_chunks.append(self._soft_argmax(heatmaps))
        return torch.cat(xy_chunks, dim=1)  # (B, Q, K, 2)

    def forward(
        self,
        spatial_features: torch.Tensor,
        query_features: torch.Tensor,
        image_size: tuple[int, int],
        skip_blocks: bool = False,
    ) -> list[torch.Tensor]:
        """Predict keypoints from spatial features and decoder query features.

        Args:
            spatial_features: Backbone spatial features of shape ``(B, C, H, W)``.
            query_features: Stacked decoder query features of shape ``(L, B, Q, C)``
                where L is the number of decoder layers.
            image_size: Original image size ``(H, W)`` (unused, kept for API consistency
                with segmentation head).
            skip_blocks: If True, skip spatial refinement blocks (for encoder path).

        Returns:
            List of keypoint tensors, one per decoder layer. Each tensor has shape
            ``(B, Q, K, 3)`` where the last dimension is ``(x, y, visibility)``.
            Coordinates are in [0, 1] normalized space. Visibility contains raw logits
            during training; sigmoid-activated in export mode.
        """
        results = []

        if not skip_blocks:
            for block, qf in zip(self.blocks, query_features):
                spatial_features = block(spatial_features)
                # Upsample AFTER conv blocks (blocks stay at native 32x32, saving memory)
                sp = self._upsample_spatial(spatial_features, image_size)
                sp = self.spatial_proj(sp)  # (B, kpt_embed_dim, H', W')

                # Query projection: (B, Q, C) -> (B, Q, K, kpt_embed_dim)
                qf_processed = self.query_block(qf)
                qf_proj = self.query_proj(qf_processed)
                qf_proj = qf_proj.reshape(qf.shape[0], qf.shape[1], self.num_keypoints, self.kpt_embed_dim)

                # Chunked heatmap -> soft-argmax -> (B, Q, K, 2)
                xy = self._chunked_heatmap_to_coords(sp, qf_proj)

                # Visibility from query features
                vis = self.vis_head(qf_processed)  # (B, Q, K)
                if self._export:
                    vis = vis.sigmoid()
                vis = vis.unsqueeze(-1)  # (B, Q, K, 1)

                results.append(torch.cat([xy, vis], dim=-1))  # (B, Q, K, 3)
        else:
            assert len(query_features) == 1, "skip_blocks only supported for single query set"
            qf = query_features[0]
            qf_processed = self.query_block(qf)
            qf_proj = self.query_proj(qf_processed)
            qf_proj = qf_proj.reshape(qf.shape[0], qf.shape[1], self.num_keypoints, self.kpt_embed_dim)

            # No upsampling for encoder path (auxiliary loss, saves memory)
            sp = self.spatial_proj(spatial_features)  # (B, kpt_embed_dim, H, W)
            xy = self._chunked_heatmap_to_coords(sp, qf_proj)

            vis = self.vis_head(qf_processed)
            if self._export:
                vis = vis.sigmoid()
            vis = vis.unsqueeze(-1)

            results.append(torch.cat([xy, vis], dim=-1))

        return results

    def forward_export(
        self,
        spatial_features: torch.Tensor,
        query_features: torch.Tensor,
        image_size: tuple[int, int],
        skip_blocks: bool = False,
    ) -> list[torch.Tensor]:
        """Export-time forward: all blocks applied sequentially, single query set.

        Args:
            spatial_features: Backbone spatial features of shape ``(B, C, H, W)``.
            query_features: Single query features of shape ``(1, B, Q, C)`` or ``(B, Q, C)``.
            image_size: Original image size ``(H, W)``.
            skip_blocks: If True, skip spatial refinement blocks.

        Returns:
            List with a single keypoint tensor of shape ``(B, Q, K, 3)``.
        """
        if query_features.dim() == 3:
            qf = query_features
        else:
            assert query_features.shape[0] == 1, "export expects single query set"
            qf = query_features[0]

        if not skip_blocks:
            for block in self.blocks:
                spatial_features = block(spatial_features)

        # Upsample AFTER conv blocks (blocks stay at native resolution, saving memory)
        sp = self._upsample_spatial(spatial_features, image_size)
        sp = self.spatial_proj(sp)

        qf_processed = self.query_block(qf)
        qf_proj = self.query_proj(qf_processed)
        qf_proj = qf_proj.reshape(qf.shape[0], qf.shape[1], self.num_keypoints, self.kpt_embed_dim)

        xy = self._chunked_heatmap_to_coords(sp, qf_proj)

        vis = self.vis_head(qf_processed)
        if self._export:
            vis = vis.sigmoid()
        vis = vis.unsqueeze(-1)

        return [torch.cat([xy, vis], dim=-1)]
