# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for RF-DETR Pose configuration classes."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from rfdetr.config import (
    PoseTrainConfig,
    RFDETRPoseBaseConfig,
    RFDETRPoseLargeConfig,
    RFDETRPoseSmallConfig,
)
from rfdetr.detr import RFDETR, RFDETRPoseLarge, RFDETRPoseSmall


class TestPoseModelConfigs:
    """Test pose model configuration classes."""

    def test_pose_base_config_defaults(self) -> None:
        """RFDETRPoseBaseConfig should have keypoint_head=True, num_keypoints=17, num_classes=1."""
        config = RFDETRPoseBaseConfig()

        assert config.keypoint_head is True
        assert config.num_keypoints == 17
        assert config.num_classes == 1

    @pytest.mark.parametrize(
        "config_class,expected_resolution,expected_dec_layers",
        [
            pytest.param(RFDETRPoseSmallConfig, 512, 3, id="small"),
            pytest.param(RFDETRPoseLargeConfig, 704, 4, id="large"),
        ],
    )
    def test_pose_variant_configs(self, config_class, expected_resolution, expected_dec_layers) -> None:
        """Pose variant configs should have correct architecture parameters."""
        config = config_class()

        assert config.keypoint_head is True
        assert config.num_keypoints == 17
        assert config.num_classes == 1
        assert config.resolution == expected_resolution
        assert config.dec_layers == expected_dec_layers
        assert config.patch_size == 16

    def test_pose_config_rejects_unknown_fields(self) -> None:
        """Pose configs should reject unknown fields (extra='forbid')."""
        with pytest.raises(ValidationError, match="Unknown"):
            RFDETRPoseSmallConfig(unknown_field=True)


class TestPoseTrainConfig:
    """Test pose training configuration."""

    def test_defaults(self) -> None:
        """PoseTrainConfig should have correct defaults."""
        config = PoseTrainConfig(dataset_dir="/tmp")

        assert config.keypoint_head is True
        assert config.num_keypoints == 17
        assert config.keypoint_l1_loss_coef == 5.0
        assert config.keypoint_vis_loss_coef == 1.0
        assert config.cls_loss_coef == 2.0
        assert config.square_resize_div_64 is True

    def test_dataset_file_defaults_to_coco(self) -> None:
        """PoseTrainConfig should default dataset_file to 'coco'."""
        config = PoseTrainConfig(dataset_dir="/tmp")

        assert config.dataset_file == "coco"

    def test_accepts_extra_fields(self) -> None:
        """PoseTrainConfig inherits TrainConfig which allows extra fields."""
        config = PoseTrainConfig(dataset_dir="/tmp", extra_param=True)

        assert config.keypoint_head is True

    def test_custom_loss_coefficients(self) -> None:
        """PoseTrainConfig should accept custom loss coefficients."""
        config = PoseTrainConfig(
            dataset_dir="/tmp",
            keypoint_l1_loss_coef=10.0,
            keypoint_vis_loss_coef=2.0,
        )

        assert config.keypoint_l1_loss_coef == 10.0
        assert config.keypoint_vis_loss_coef == 2.0


class TestPoseUserClasses:
    """Test user-facing pose classes."""

    def test_rfdetr_pose_small_model_config(self) -> None:
        """RFDETRPoseSmall should produce RFDETRPoseSmallConfig."""
        model = RFDETRPoseSmall.__new__(RFDETRPoseSmall)
        config = model.get_model_config()

        assert isinstance(config, RFDETRPoseSmallConfig)
        assert config.keypoint_head is True

    def test_rfdetr_pose_large_model_config(self) -> None:
        """RFDETRPoseLarge should produce RFDETRPoseLargeConfig."""
        model = RFDETRPoseLarge.__new__(RFDETRPoseLarge)
        config = model.get_model_config()

        assert isinstance(config, RFDETRPoseLargeConfig)
        assert config.keypoint_head is True

    @pytest.mark.parametrize(
        "cls",
        [
            pytest.param(RFDETRPoseSmall, id="small"),
            pytest.param(RFDETRPoseLarge, id="large"),
        ],
    )
    def test_train_config_is_pose(self, cls) -> None:
        """Pose classes should produce PoseTrainConfig."""
        model = cls.__new__(cls)
        config = model.get_train_config(dataset_dir="/tmp")

        assert isinstance(config, PoseTrainConfig)
        assert config.keypoint_head is True


def _make_rfdetr_stub(model_config):
    """Build a minimal RFDETR-like object for testing train_from_config()."""
    stub = RFDETR.__new__(RFDETR)
    stub.model_config = model_config
    stub.model = MagicMock()
    stub.model.class_names = []
    stub.callbacks = {
        "on_fit_epoch_end": [],
        "on_train_end": [],
    }
    return stub


class TestPoseTrainFromConfig:
    """Test train_from_config behavior for pose models."""

    def test_pose_raises_when_square_resize_disabled(self, tmp_path) -> None:
        """Pose training must raise ValueError when square_resize_div_64=False."""
        stub = _make_rfdetr_stub(RFDETRPoseSmallConfig())
        train_config = PoseTrainConfig(
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path),
            dataset_file="coco",
            tensorboard=False,
            square_resize_div_64=False,
        )

        with pytest.raises(ValueError, match="square_resize_div_64"):
            stub.train_from_config(train_config)

    def test_pose_config_passes_keypoint_params(self, tmp_path) -> None:
        """train_from_config should forward keypoint parameters to model.train()."""
        stub = _make_rfdetr_stub(RFDETRPoseSmallConfig())
        train_config = PoseTrainConfig(
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path),
            dataset_file="coco",
            tensorboard=False,
        )

        stub.train_from_config(train_config)

        call_kwargs = stub.model.train.call_args.kwargs
        assert call_kwargs["keypoint_head"] is True
        assert call_kwargs["num_keypoints"] == 17
        assert call_kwargs["keypoint_l1_loss_coef"] == 5.0
        assert call_kwargs["keypoint_vis_loss_coef"] == 1.0
