# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Keypoint estimation head for RF-DETR pose models."""

import torch
import torch.nn as nn

from rfdetr.models.lwdetr import MLP


class KeypointHead(nn.Module):
    """Keypoint prediction head that regresses box-relative offsets and visibility from decoder queries.

    Predicts (x, y, visibility) for each keypoint per query using an MLP. The xy outputs are
    box-relative offsets: ``tanh(raw)`` gives offsets in [-1, 1] that are scaled by the predicted
    box size and shifted by the box center to produce absolute [0, 1] normalized coordinates.

    Args:
        hidden_dim: Dimension of the input query features.
        num_keypoints: Number of keypoints to predict per detection.
        num_layers: Number of MLP layers.
    """

    def __init__(self, hidden_dim: int, num_keypoints: int = 17, num_layers: int = 3) -> None:
        super().__init__()
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
