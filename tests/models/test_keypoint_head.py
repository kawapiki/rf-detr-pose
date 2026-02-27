# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the KeypointHead module."""

import pytest
import torch

from rfdetr.models.keypoint_head import KeypointHead


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="gpu",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(),
                    reason="CUDA is not available",
                ),
            ],
        ),
    ],
)
class TestKeypointHeadForward:
    """Test KeypointHead forward pass on CPU and GPU."""

    def test_output_shape_single_layer(self, device: str) -> None:
        """Forward pass produces correct output shape for a single decoder layer."""
        head = KeypointHead(hidden_dim=256, num_keypoints=17).to(device)
        query_features = torch.randn(2, 300, 256, device=device)
        pred_boxes = torch.rand(2, 300, 4, device=device)

        output = head(query_features, pred_boxes)

        assert output.shape == (2, 300, 17, 3)

    def test_output_shape_multi_layer(self, device: str) -> None:
        """Forward pass works with stacked decoder layer features."""
        head = KeypointHead(hidden_dim=128, num_keypoints=5).to(device)
        query_features = torch.randn(4, 3, 100, 128, device=device)
        pred_boxes = torch.rand(4, 3, 100, 4, device=device)

        output = head(query_features, pred_boxes)

        assert output.shape == (4, 3, 100, 5, 3)

    def test_xy_coords_in_zero_one_range(self, device: str) -> None:
        """XY coordinates should be in [0, 1] (box-relative offsets clamped)."""
        head = KeypointHead(hidden_dim=64, num_keypoints=3).to(device)
        query_features = torch.randn(1, 10, 64, device=device)
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4]]] * 10, device=device).reshape(1, 10, 4)

        output = head(query_features, pred_boxes)

        xy = output[..., :2]
        assert (xy >= 0).all()
        assert (xy <= 1).all()


class TestKeypointHeadInit:
    """Test KeypointHead initialization."""

    def test_output_layer_zero_initialized(self) -> None:
        """Output layer weights and biases should be zero-initialized."""
        head = KeypointHead(hidden_dim=64, num_keypoints=17)

        last_layer = head.mlp.layers[-1]
        assert torch.allclose(last_layer.weight.data, torch.zeros_like(last_layer.weight.data))
        assert torch.allclose(last_layer.bias.data, torch.zeros_like(last_layer.bias.data))

    def test_custom_num_keypoints(self) -> None:
        """Head respects custom number of keypoints."""
        head = KeypointHead(hidden_dim=32, num_keypoints=5, num_layers=2)

        assert head.num_keypoints == 5
        output = head(torch.randn(1, 4, 32), torch.rand(1, 4, 4))
        assert output.shape == (1, 4, 5, 3)


class TestKeypointHeadExport:
    """Test KeypointHead export mode behavior."""

    def test_visibility_raw_logits_in_training_mode(self) -> None:
        """In training mode, visibility values are raw logits (not sigmoid-activated)."""
        head = KeypointHead(hidden_dim=64, num_keypoints=3)

        assert not head._export

    def test_visibility_sigmoid_in_export_mode(self) -> None:
        """In export mode, visibility values should be sigmoid-activated to [0, 1]."""
        head = KeypointHead(hidden_dim=64, num_keypoints=3)
        head.export()

        output = head(torch.randn(1, 5, 64), torch.rand(1, 5, 4))
        vis = output[..., 2]

        assert head._export
        assert (vis >= 0).all()
        assert (vis <= 1).all()

    def test_export_flag_set(self) -> None:
        """export() method should set _export flag to True."""
        head = KeypointHead(hidden_dim=64)

        assert not head._export
        head.export()
        assert head._export
