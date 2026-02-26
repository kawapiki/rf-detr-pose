# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Keypoint estimation head for RF-DETR pose models."""

from typing import Callable

import torch
import torch.nn as nn

from rfdetr.models.lwdetr import MLP


class KeypointHead(nn.Module):
    """Keypoint prediction head that regresses keypoint coordinates and visibility from decoder queries.

    Predicts (x, y, visibility) for each keypoint per query using an MLP.
    Coordinates are sigmoid-activated to [0, 1] range (normalized image coordinates).
    Visibility values are raw logits during training and sigmoid-activated during export.

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

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """Predict keypoints from decoder query features.

        Args:
            query_features: Query features of shape ``(..., hidden_dim)``.

        Returns:
            Keypoint predictions of shape ``(..., num_keypoints, 3)`` where the last
            dimension contains ``(x, y, visibility)``. During training, ``x`` and ``y``
            are sigmoid-activated while visibility contains raw logits. In export mode,
            visibility is also sigmoid-activated.
        """
        raw = self.mlp(query_features)
        # Reshape to (..., num_keypoints, 3)
        shape = raw.shape[:-1] + (self.num_keypoints, 3)
        raw = raw.view(shape)

        xy = raw[..., :2].sigmoid()
        vis = raw[..., 2:3]
        if self._export:
            vis = vis.sigmoid()

        return torch.cat([xy, vis], dim=-1)
