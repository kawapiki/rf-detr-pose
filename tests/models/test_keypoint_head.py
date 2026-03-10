# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the SpatialKeypointHead and KeypointHead modules."""

import pytest
import torch

from rfdetr.models.keypoint_head import KeypointHead, KeypointHeadMLP, SpatialKeypointHead


class TestKeypointHeadMLPBackwardCompat:
    """Verify KeypointHeadMLP alias still works."""

    def test_alias_is_same_class(self) -> None:
        """KeypointHeadMLP should be the same class as KeypointHead."""
        assert KeypointHeadMLP is KeypointHead


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
class TestSpatialKeypointHeadForward:
    """Test SpatialKeypointHead forward pass on CPU and GPU."""

    def test_output_shape_single_layer(self, device: str) -> None:
        """Forward produces one (B, Q, K, 3) tensor per decoder layer."""
        head = SpatialKeypointHead(hidden_dim=256, num_keypoints=17, num_blocks=3, kpt_embed_dim=64).to(device)
        spatial = torch.randn(2, 256, 32, 32, device=device)
        # hs shape: (L, B, Q, C) — 3 decoder layers
        queries = torch.randn(3, 2, 300, 256, device=device)

        results = head(spatial, queries, (512, 512))

        assert len(results) == 3
        for r in results:
            assert r.shape == (2, 300, 17, 3)

    def test_output_shape_custom_keypoints(self, device: str) -> None:
        """Forward works with custom number of keypoints."""
        head = SpatialKeypointHead(hidden_dim=128, num_keypoints=5, num_blocks=2, kpt_embed_dim=32).to(device)
        spatial = torch.randn(1, 128, 16, 16, device=device)
        queries = torch.randn(2, 1, 50, 128, device=device)

        results = head(spatial, queries, (256, 256))

        assert len(results) == 2
        for r in results:
            assert r.shape == (1, 50, 5, 3)

    def test_xy_coords_in_zero_one_range(self, device: str) -> None:
        """XY coordinates from soft-argmax should be in (0, 1)."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=3, num_blocks=1, kpt_embed_dim=16).to(device)
        spatial = torch.randn(1, 64, 8, 8, device=device)
        queries = torch.randn(1, 1, 10, 64, device=device)

        results = head(spatial, queries, (128, 128))

        xy = results[0][..., :2]
        assert (xy > 0).all()
        assert (xy < 1).all()

    def test_skip_blocks_encoder_path(self, device: str) -> None:
        """skip_blocks=True works for encoder path (single query set)."""
        head = SpatialKeypointHead(hidden_dim=256, num_keypoints=17, num_blocks=3, kpt_embed_dim=64).to(device)
        spatial = torch.randn(2, 256, 32, 32, device=device)
        queries_enc = [torch.randn(2, 300, 256, device=device)]

        results = head(spatial, queries_enc, (512, 512), skip_blocks=True)

        assert len(results) == 1
        assert results[0].shape == (2, 300, 17, 3)


class TestSpatialKeypointHeadInit:
    """Test SpatialKeypointHead initialization."""

    def test_temperature_initialized_to_one(self) -> None:
        """Learnable temperature should start at 1.0."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=5)

        assert head.temperature.item() == pytest.approx(1.0)

    def test_num_blocks_matches_config(self) -> None:
        """Number of DepthwiseConvBlocks should match num_blocks."""
        head = SpatialKeypointHead(hidden_dim=64, num_blocks=4)

        assert len(head.blocks) == 4


class TestSpatialKeypointHeadExport:
    """Test SpatialKeypointHead export mode behavior."""

    def test_export_sets_flag(self) -> None:
        """export() should set _export flag."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=3)

        assert not head._export
        head.export()
        assert head._export

    def test_export_forward_produces_single_result(self) -> None:
        """Export forward should return list with single tensor."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=3, num_blocks=2, kpt_embed_dim=16)
        head.export()

        spatial = torch.randn(1, 64, 8, 8)
        queries = torch.randn(1, 1, 10, 64)  # (1, B, Q, C) — single layer

        results = head(spatial, queries, (128, 128))

        assert len(results) == 1
        assert results[0].shape == (1, 10, 3, 3)

    def test_export_visibility_sigmoid(self) -> None:
        """In export mode, visibility should be sigmoid-activated to [0, 1]."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=3, num_blocks=1, kpt_embed_dim=16)
        head.export()

        spatial = torch.randn(1, 64, 8, 8)
        queries = torch.randn(1, 1, 5, 64)

        results = head(spatial, queries, (128, 128))
        vis = results[0][..., 2]

        assert (vis >= 0).all()
        assert (vis <= 1).all()


class TestSpatialKeypointHeadUpsample:
    """Test spatial upsampling behavior."""

    def test_upsample_doubles_resolution(self) -> None:
        """With downsample_ratio=8 and image_size=512, features should be 64x64."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=3, num_blocks=1, kpt_embed_dim=16, downsample_ratio=8)
        spatial = torch.randn(1, 64, 32, 32)

        upsampled = head._upsample_spatial(spatial, (512, 512))

        assert upsampled.shape == (1, 64, 64, 64)

    def test_upsample_ratio_4_gives_128(self) -> None:
        """With downsample_ratio=4, features should be 128x128 for 512px input."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=3, num_blocks=1, kpt_embed_dim=16, downsample_ratio=4)
        spatial = torch.randn(1, 64, 32, 32)

        upsampled = head._upsample_spatial(spatial, (512, 512))

        assert upsampled.shape == (1, 64, 128, 128)

    def test_no_upsample_when_already_target_size(self) -> None:
        """If features already match target size, no interpolation needed."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=3, num_blocks=1, kpt_embed_dim=16, downsample_ratio=16)
        spatial = torch.randn(1, 64, 32, 32)

        upsampled = head._upsample_spatial(spatial, (512, 512))

        assert upsampled.shape == (1, 64, 32, 32)
        assert upsampled is spatial

    def test_forward_output_shape_unchanged_with_upsample(self) -> None:
        """Output shape should be the same regardless of downsample_ratio."""
        for ratio in [4, 8, 16]:
            head = SpatialKeypointHead(
                hidden_dim=64, num_keypoints=3, num_blocks=1, kpt_embed_dim=16, downsample_ratio=ratio
            )
            spatial = torch.randn(1, 64, 32, 32)
            queries = torch.randn(1, 1, 10, 64)

            results = head(spatial, queries, (512, 512))

            assert results[0].shape == (1, 10, 3, 3), f"Failed for ratio={ratio}"

    def test_higher_resolution_improves_precision(self) -> None:
        """Higher resolution heatmaps should give more precise coordinates for sharp peaks."""
        head_coarse = SpatialKeypointHead(
            hidden_dim=64, num_keypoints=1, num_blocks=1, kpt_embed_dim=16, downsample_ratio=16
        )
        head_fine = SpatialKeypointHead(
            hidden_dim=64, num_keypoints=1, num_blocks=1, kpt_embed_dim=16, downsample_ratio=4
        )
        head_fine.load_state_dict(head_coarse.state_dict())

        spatial = torch.randn(1, 64, 32, 32)
        queries = torch.randn(1, 1, 5, 64)

        result_coarse = head_coarse(spatial, queries, (512, 512))
        result_fine = head_fine(spatial, queries, (512, 512))

        assert (result_coarse[0][..., :2] > 0).all()
        assert (result_fine[0][..., :2] > 0).all()
        assert (result_coarse[0][..., :2] < 1).all()
        assert (result_fine[0][..., :2] < 1).all()


class TestSoftArgmax:
    """Test soft-argmax correctness with known heatmaps."""

    def test_peak_at_center_returns_center_coords(self) -> None:
        """A strong peak at the center of a heatmap should give ~(0.5, 0.5) coordinates."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=1, num_blocks=1, kpt_embed_dim=16)

        # Create a heatmap with a strong peak at center
        H, W = 16, 16
        heatmap = torch.zeros(1, 1, 1, H, W)
        heatmap[0, 0, 0, H // 2, W // 2] = 100.0  # Strong peak at center

        xy = head._soft_argmax(heatmap)

        assert xy.shape == (1, 1, 1, 2)
        # Center of a 16x16 grid: cell (8, 8) -> (8.5/16, 8.5/16) = (0.53125, 0.53125)
        assert xy[0, 0, 0, 0].item() == pytest.approx(0.53125, abs=0.05)
        assert xy[0, 0, 0, 1].item() == pytest.approx(0.53125, abs=0.05)

    def test_peak_at_corner_returns_corner_coords(self) -> None:
        """A strong peak at top-left corner should give coordinates near (0, 0)."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=1, num_blocks=1, kpt_embed_dim=16)

        H, W = 16, 16
        heatmap = torch.zeros(1, 1, 1, H, W)
        heatmap[0, 0, 0, 0, 0] = 100.0  # Peak at top-left

        xy = head._soft_argmax(heatmap)

        # Top-left cell: (0.5/16, 0.5/16) = (0.03125, 0.03125)
        assert xy[0, 0, 0, 0].item() == pytest.approx(0.03125, abs=0.05)
        assert xy[0, 0, 0, 1].item() == pytest.approx(0.03125, abs=0.05)

    def test_uniform_heatmap_returns_center(self) -> None:
        """A uniform heatmap should give coordinates at ~(0.5, 0.5)."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=1, num_blocks=1, kpt_embed_dim=16)

        H, W = 8, 8
        heatmap = torch.ones(1, 1, 1, H, W)

        xy = head._soft_argmax(heatmap)

        assert xy[0, 0, 0, 0].item() == pytest.approx(0.5, abs=0.01)
        assert xy[0, 0, 0, 1].item() == pytest.approx(0.5, abs=0.01)

    def test_soft_argmax_is_differentiable(self) -> None:
        """Soft-argmax should be differentiable (gradients flow back)."""
        head = SpatialKeypointHead(hidden_dim=64, num_keypoints=1, num_blocks=1, kpt_embed_dim=16)

        heatmap = torch.randn(1, 1, 1, 8, 8, requires_grad=True)
        xy = head._soft_argmax(heatmap)
        loss = xy.sum()
        loss.backward()

        assert heatmap.grad is not None
        assert heatmap.grad.shape == heatmap.shape


# Legacy tests for KeypointHead (MLP-based) to ensure backward compatibility
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
    """Test KeypointHead (MLP) forward pass on CPU and GPU."""

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
