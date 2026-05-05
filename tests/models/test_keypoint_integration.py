# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for keypoint integration in LWDETR: loss computation, PostProcess, and build functions."""

import pytest
import torch


class TestLossKeypoints:
    """Test the SetCriterion.loss_keypoints method."""

    @pytest.fixture
    def criterion(self):
        """Build a minimal SetCriterion with keypoints loss registered."""
        from rfdetr.models.lwdetr import SetCriterion

        class _DummyMatcher:
            def __call__(self, outputs, targets):
                return [(torch.tensor([0]), torch.tensor([0]))] * len(targets)

        criterion = SetCriterion(
            num_classes=1,
            matcher=_DummyMatcher(),
            weight_dict={"loss_keypoint_l1": 5.0, "loss_keypoint_vis": 1.0},
            focal_alpha=0.25,
            losses=["keypoints"],
        )
        return criterion

    def test_loss_keypoints_basic(self, criterion) -> None:
        """loss_keypoints returns both L1 and visibility losses."""
        B, Q, K = 2, 5, 17
        outputs = {
            "pred_keypoints": torch.randn(B, Q, K, 3),
        }
        targets = [
            {"keypoints": torch.rand(1, K, 3) * torch.tensor([1.0, 1.0, 2.0])},
            {"keypoints": torch.rand(1, K, 3) * torch.tensor([1.0, 1.0, 2.0])},
        ]
        for t in targets:
            t["keypoints"][..., 2] = 2.0

        indices = [(torch.tensor([0]), torch.tensor([0]))] * B
        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=2)

        assert "loss_keypoint_l1" in losses
        assert "loss_keypoint_vis" in losses
        assert losses["loss_keypoint_l1"].ndim == 0
        assert losses["loss_keypoint_vis"].ndim == 0

    def test_loss_keypoints_zero_visibility_masks_l1(self, criterion) -> None:
        """L1 loss should be zero when all keypoints have zero visibility."""
        B, Q, K = 1, 3, 5
        kpts = torch.tensor([[[0.5, 0.5, 0.0]] * K])  # visibility = 0
        outputs = {
            "pred_keypoints": torch.zeros(B, Q, K, 3),
        }
        outputs["pred_keypoints"][0, 0] = torch.tensor([[0.1, 0.9, -1.0]] * K)
        targets = [{"keypoints": kpts}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1)

        assert losses["loss_keypoint_l1"].item() == pytest.approx(0.0)

    def test_loss_keypoints_empty_matches(self, criterion) -> None:
        """Loss should handle empty matches gracefully."""
        B, Q, K = 1, 3, 17
        outputs = {"pred_keypoints": torch.randn(B, Q, K, 3)}
        targets = [{"keypoints": torch.rand(0, K, 3)}]
        indices = [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1)

        assert losses["loss_keypoint_l1"].item() == 0.0
        assert losses["loss_keypoint_vis"].item() == 0.0
        assert losses["loss_keypoint_oks"].item() == 0.0


class TestLossKeypointsOKS:
    """Test the OKS (Object Keypoint Similarity) loss term."""

    @pytest.fixture
    def criterion(self):
        from rfdetr.models.lwdetr import SetCriterion

        class _DummyMatcher:
            def __call__(self, outputs, targets):
                return [(torch.tensor([0]), torch.tensor([0]))] * len(targets)

        return SetCriterion(
            num_classes=1,
            matcher=_DummyMatcher(),
            weight_dict={"loss_keypoint_oks": 5.0},
            focal_alpha=0.25,
            losses=["keypoints"],
        )

    def test_oks_zero_when_predictions_are_perfect(self, criterion) -> None:
        """OKS loss = 0 when predicted keypoints match ground truth exactly."""
        B, Q, K = 1, 1, 17
        gt_xy = torch.full((K, 2), 0.5)
        outputs = {"pred_keypoints": torch.zeros(B, Q, K, 3)}
        outputs["pred_keypoints"][0, 0, :, :2] = gt_xy

        kpts = torch.zeros(1, K, 3)
        kpts[0, :, :2] = gt_xy
        kpts[0, :, 2] = 2.0  # all visible
        targets = [{
            "keypoints": kpts,
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),  # cxcywh, area=0.16
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1)

        assert losses["loss_keypoint_oks"].item() == pytest.approx(0.0, abs=1e-6)

    def test_oks_close_to_one_when_predictions_are_far(self, criterion) -> None:
        """OKS loss → 1 when predictions are far (relative to area & sigma)."""
        B, Q, K = 1, 1, 17
        outputs = {"pred_keypoints": torch.zeros(B, Q, K, 3)}
        outputs["pred_keypoints"][0, 0, :, :2] = torch.tensor([0.0, 0.0])

        kpts = torch.zeros(1, K, 3)
        kpts[0, :, :2] = torch.tensor([1.0, 1.0])  # opposite corner
        kpts[0, :, 2] = 2.0
        targets = [{
            "keypoints": kpts,
            "boxes": torch.tensor([[0.5, 0.5, 0.05, 0.05]]),  # very small area, big d/s ratio
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1)

        # exp(-large) is ~0, so 1-OKS ~ 1
        assert losses["loss_keypoint_oks"].item() == pytest.approx(1.0, abs=1e-3)

    def test_oks_ignores_invisible_keypoints(self, criterion) -> None:
        """Invisible keypoints (v=0) must not contribute to OKS."""
        B, Q, K = 1, 1, 17
        outputs = {"pred_keypoints": torch.zeros(B, Q, K, 3)}
        # First keypoint perfect; rest are wildly wrong
        outputs["pred_keypoints"][0, 0, 0, :2] = torch.tensor([0.5, 0.5])
        outputs["pred_keypoints"][0, 0, 1:, :2] = torch.tensor([0.0, 0.0])

        kpts = torch.zeros(1, K, 3)
        kpts[0, 0, :2] = torch.tensor([0.5, 0.5])
        kpts[0, 0, 2] = 2.0  # only first keypoint is visible
        # Other keypoints labeled at (1,1) but invisible (v=0)
        kpts[0, 1:, :2] = torch.tensor([1.0, 1.0])
        kpts[0, 1:, 2] = 0.0
        targets = [{
            "keypoints": kpts,
            "boxes": torch.tensor([[0.5, 0.5, 0.05, 0.05]]),
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1)

        # Only the perfect (visible) keypoint contributes → OKS ≈ 1, loss ≈ 0
        assert losses["loss_keypoint_oks"].item() == pytest.approx(0.0, abs=1e-3)

    def test_oks_is_zero_when_no_visible_keypoints(self, criterion) -> None:
        """No visible keypoints → OKS contribution is zero."""
        B, Q, K = 1, 1, 17
        outputs = {"pred_keypoints": torch.zeros(B, Q, K, 3)}

        kpts = torch.zeros(1, K, 3)  # all visibility=0
        targets = [{
            "keypoints": kpts,
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1)
        assert losses["loss_keypoint_oks"].item() == pytest.approx(0.0)

    def test_oks_gracefully_skipped_without_boxes(self, criterion) -> None:
        """When targets lack 'boxes', OKS returns 0 (not an error)."""
        B, Q, K = 1, 1, 17
        outputs = {"pred_keypoints": torch.zeros(B, Q, K, 3)}
        kpts = torch.zeros(1, K, 3)
        kpts[0, :, 2] = 2.0
        targets = [{"keypoints": kpts}]  # no "boxes"
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1)
        assert losses["loss_keypoint_oks"].item() == pytest.approx(0.0)

    def test_oks_loss_is_differentiable(self, criterion) -> None:
        """OKS loss must produce gradients on predicted coordinates."""
        B, Q, K = 1, 1, 17
        pred = torch.zeros(B, Q, K, 3, requires_grad=True)
        kpts = torch.zeros(1, K, 3)
        kpts[0, :, :2] = 0.3
        kpts[0, :, 2] = 2.0
        targets = [{
            "keypoints": kpts,
            "boxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints({"pred_keypoints": pred}, targets, indices, num_boxes=1)
        losses["loss_keypoint_oks"].backward()

        assert pred.grad is not None
        # Coords should have non-zero grad (visibility column may be zero)
        assert pred.grad[..., :2].abs().sum() > 0


class TestPostProcessKeypoints:
    """Test PostProcess handling of keypoint outputs."""

    def test_postprocess_includes_keypoints_in_results(self) -> None:
        """PostProcess should include scaled keypoints in results when pred_keypoints is present."""
        from rfdetr.models.lwdetr import PostProcess

        num_select = 10
        postprocess = PostProcess(num_select=num_select)

        B, Q, K = 1, 50, 17
        outputs = {
            "pred_logits": torch.randn(B, Q, 1),
            "pred_boxes": torch.rand(B, Q, 4),
            "pred_keypoints": torch.rand(B, Q, K, 3),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = postprocess(outputs, target_sizes)

        assert len(results) == 1
        assert "keypoints" in results[0]
        kpts = results[0]["keypoints"]
        assert kpts.shape == (num_select, K, 3)

    def test_postprocess_scales_keypoints_to_absolute(self) -> None:
        """PostProcess should scale keypoint xy from [0,1] to absolute pixel coordinates."""
        from rfdetr.models.lwdetr import PostProcess

        num_select = 1
        postprocess = PostProcess(num_select=num_select)

        B, Q, K = 1, 1, 2
        kpts = torch.zeros(B, Q, K, 3)
        kpts[..., 0] = 0.5  # x
        kpts[..., 1] = 0.5  # y
        kpts[..., 2] = 0.0  # vis logit

        outputs = {
            "pred_logits": torch.ones(B, Q, 1),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
            "pred_keypoints": kpts,
        }
        target_sizes = torch.tensor([[480, 640]])

        results = postprocess(outputs, target_sizes)

        result_kpts = results[0]["keypoints"]
        # x=0.5 * 640 = 320, y=0.5 * 480 = 240
        assert result_kpts[0, 0, 0].item() == pytest.approx(320.0, abs=1.0)
        assert result_kpts[0, 0, 1].item() == pytest.approx(240.0, abs=1.0)
        # Visibility is converted to COCO format: 0 (not visible) or 2 (visible)
        assert result_kpts[0, 0, 2].item() in (0.0, 2.0)

    def test_postprocess_without_keypoints(self) -> None:
        """PostProcess should work normally when no keypoints are present."""
        from rfdetr.models.lwdetr import PostProcess

        postprocess = PostProcess(num_select=5)

        outputs = {
            "pred_logits": torch.randn(1, 20, 1),
            "pred_boxes": torch.rand(1, 20, 4),
        }
        target_sizes = torch.tensor([[480, 640]])

        results = postprocess(outputs, target_sizes)

        assert "keypoints" not in results[0]
        assert "scores" in results[0]
        assert "boxes" in results[0]


def _make_build_args(**overrides):
    """Create a minimal argparse.Namespace for build_criterion_and_postprocessors."""
    import argparse

    defaults = dict(
        num_classes=1,
        hidden_dim=256,
        set_cost_class=2.0,
        set_cost_bbox=5.0,
        set_cost_giou=2.0,
        cls_loss_coef=2.0,
        bbox_loss_coef=5.0,
        giou_loss_coef=2.0,
        focal_alpha=0.25,
        dec_layers=3,
        segmentation_head=False,
        keypoint_head=False,
        num_select=300,
        mask_loss_coef=1.0,
        dice_loss_coef=1.0,
        device="cpu",
        aux_loss=True,
        two_stage=False,
        group_detr=1,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestBuildCriterionWithKeypoints:
    """Test build_criterion_and_postprocessors with keypoint config."""

    def test_build_criterion_registers_keypoint_losses(self) -> None:
        """build_criterion_and_postprocessors should include keypoint losses when configured."""
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        args = _make_build_args(
            keypoint_head=True,
            num_keypoints=17,
            keypoint_l1_loss_coef=5.0,
            keypoint_vis_loss_coef=1.0,
        )

        criterion, postprocessors = build_criterion_and_postprocessors(args)

        assert "loss_keypoint_l1" in criterion.weight_dict
        assert "loss_keypoint_vis" in criterion.weight_dict
        assert criterion.weight_dict["loss_keypoint_l1"] == 5.0
        assert criterion.weight_dict["loss_keypoint_vis"] == 1.0

        # Check aux weight dicts too (aux_loss=True, dec_layers=3 -> 2 aux layers)
        for i in range(args.dec_layers - 1):
            assert f"loss_keypoint_l1_{i}" in criterion.weight_dict
            assert f"loss_keypoint_vis_{i}" in criterion.weight_dict

        assert "keypoints" in criterion.losses
        # OKS coef defaults to 0 → not in weight_dict
        assert "loss_keypoint_oks" not in criterion.weight_dict

    def test_build_criterion_registers_oks_when_coef_positive(self) -> None:
        """OKS loss appears in weight_dict only when keypoint_oks_loss_coef > 0."""
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        args = _make_build_args(
            keypoint_head=True,
            num_keypoints=17,
            keypoint_l1_loss_coef=2.0,
            keypoint_vis_loss_coef=0.3,
            keypoint_oks_loss_coef=10.0,
        )
        criterion, _ = build_criterion_and_postprocessors(args)

        assert "loss_keypoint_oks" in criterion.weight_dict
        assert criterion.weight_dict["loss_keypoint_oks"] == 10.0
        # Aux weights for OKS too
        for i in range(args.dec_layers - 1):
            assert f"loss_keypoint_oks_{i}" in criterion.weight_dict

    def test_build_criterion_without_keypoints_has_no_keypoint_losses(self) -> None:
        """Without keypoint_head, criterion should not have keypoint losses."""
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        args = _make_build_args(keypoint_head=False)

        criterion, _ = build_criterion_and_postprocessors(args)

        assert "loss_keypoint_l1" not in criterion.weight_dict
        assert "keypoints" not in criterion.losses
