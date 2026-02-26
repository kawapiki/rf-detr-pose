# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for pose dataset support: synthetic generation, ConvertCoco keypoints, and transforms."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets.synthetic import (
    COCO_PERSON_KEYPOINTS,
    COCO_PERSON_SKELETON,
    DatasetSplitRatios,
    generate_coco_pose_dataset,
)
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize


class TestGenerateCocoPostDataset:
    """Tests for synthetic pose dataset generation."""

    def test_generates_valid_coco_pose_format(self, tmp_path) -> None:
        """Generated dataset should have valid COCO keypoint annotation format."""
        output_dir = tmp_path / "pose_dataset"
        generate_coco_pose_dataset(
            output_dir=str(output_dir),
            num_images=5,
            img_size=128,
            split_ratios=DatasetSplitRatios(train=0.8, val=0.2, test=0.0),
        )

        train_dir = output_dir / "train"
        assert train_dir.exists()
        ann_path = train_dir / "_annotations.coco.json"
        assert ann_path.exists()

        with open(ann_path) as f:
            data = json.load(f)

        assert "images" in data
        assert "annotations" in data
        assert "categories" in data

    def test_annotations_have_keypoints(self, tmp_path) -> None:
        """Each annotation should have keypoints and num_keypoints fields."""
        output_dir = tmp_path / "pose_dataset"
        generate_coco_pose_dataset(
            output_dir=str(output_dir),
            num_images=5,
            img_size=128,
            split_ratios=DatasetSplitRatios(train=1.0, val=0.0, test=0.0),
        )

        with open(output_dir / "train" / "_annotations.coco.json") as f:
            data = json.load(f)

        assert len(data["annotations"]) > 0
        for ann in data["annotations"]:
            assert "keypoints" in ann
            assert "num_keypoints" in ann
            assert len(ann["keypoints"]) == 17 * 3
            assert ann["num_keypoints"] > 0

    def test_categories_have_keypoint_metadata(self, tmp_path) -> None:
        """Categories should include keypoint names and skeleton."""
        output_dir = tmp_path / "pose_dataset"
        generate_coco_pose_dataset(
            output_dir=str(output_dir),
            num_images=3,
            img_size=64,
            split_ratios=DatasetSplitRatios(train=1.0, val=0.0, test=0.0),
        )

        with open(output_dir / "train" / "_annotations.coco.json") as f:
            data = json.load(f)

        cat = data["categories"][0]
        assert cat["name"] == "person"
        assert "keypoints" in cat
        assert len(cat["keypoints"]) == 17
        assert "skeleton" in cat

    def test_images_exist_on_disk(self, tmp_path) -> None:
        """Generated images should exist on disk."""
        output_dir = tmp_path / "pose_dataset"
        generate_coco_pose_dataset(
            output_dir=str(output_dir),
            num_images=3,
            img_size=64,
            split_ratios=DatasetSplitRatios(train=1.0, val=0.0, test=0.0),
        )

        with open(output_dir / "train" / "_annotations.coco.json") as f:
            data = json.load(f)

        for img_info in data["images"]:
            assert (output_dir / "train" / img_info["file_name"]).exists()


class TestCocoPoseConstants:
    """Tests for COCO pose keypoint constants."""

    def test_keypoint_names_length(self) -> None:
        """COCO person has 17 keypoints."""
        assert len(COCO_PERSON_KEYPOINTS) == 17

    def test_skeleton_connections(self) -> None:
        """Skeleton should have valid connections (1-indexed into keypoints)."""
        assert len(COCO_PERSON_SKELETON) > 0
        for connection in COCO_PERSON_SKELETON:
            assert len(connection) == 2
            assert 1 <= connection[0] <= 17
            assert 1 <= connection[1] <= 17


class TestNormalizeKeypoints:
    """Test Normalize transform handles keypoints."""

    def test_normalize_keypoints(self) -> None:
        """Normalize should convert keypoint xy to [0,1] by dividing by image dimensions."""
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        image = torch.rand(3, 100, 200)  # C, H, W -> h=100, w=200
        target = {
            "keypoints": torch.tensor(
                [
                    [[100.0, 50.0, 2.0], [150.0, 75.0, 1.0], [0.0, 0.0, 0.0]],
                ]
            ),  # [1, 3, 3] - one instance, 3 keypoints
        }

        _, new_target = normalize(image, target)

        kpts = new_target["keypoints"]
        # x=100 / w=200 = 0.5, y=50 / h=100 = 0.5
        assert kpts[0, 0, 0] == pytest.approx(0.5)
        assert kpts[0, 0, 1] == pytest.approx(0.5)
        # x=150 / 200 = 0.75, y=75 / 100 = 0.75
        assert kpts[0, 1, 0] == pytest.approx(0.75)
        assert kpts[0, 1, 1] == pytest.approx(0.75)
        # Visibility unchanged
        assert kpts[0, 0, 2] == pytest.approx(2.0)
        assert kpts[0, 1, 2] == pytest.approx(1.0)
        assert kpts[0, 2, 2] == pytest.approx(0.0)


class TestAlbumentationsWrapperKeypoints:
    """Test AlbumentationsWrapper handles keypoints in geometric transforms."""

    def test_horizontal_flip_transforms_keypoints(self) -> None:
        """Horizontal flip should mirror keypoint x-coordinates."""
        import albumentations as A

        wrapper = AlbumentationsWrapper(A.HorizontalFlip(p=1.0))

        image = Image.new("RGB", (100, 100))
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            "labels": torch.tensor([1]),
            "keypoints": torch.tensor(
                [
                    [[20.0, 30.0, 2.0], [80.0, 50.0, 1.0]],
                ]
            ),  # [1, 2, 3]
        }

        _, aug_target = wrapper(image, target)

        # After horizontal flip in 100px image:
        # x=20 -> 100-20=80, x=80 -> 100-80=20
        kpts = aug_target["keypoints"]
        assert kpts.shape[-1] == 3
        assert kpts[0, 0, 0] == pytest.approx(80.0, abs=1.0)
        assert kpts[0, 1, 0] == pytest.approx(20.0, abs=1.0)
        # y unchanged
        assert kpts[0, 0, 1] == pytest.approx(30.0, abs=1.0)
        assert kpts[0, 1, 1] == pytest.approx(50.0, abs=1.0)

    def test_keypoints_preserved_with_non_geometric_transform(self) -> None:
        """Non-geometric transforms should pass through keypoints unchanged."""
        import albumentations as A

        wrapper = AlbumentationsWrapper(A.RandomBrightnessContrast(p=1.0))

        image = Image.new("RGB", (100, 100))
        keypoints_in = torch.tensor(
            [
                [[25.0, 35.0, 2.0], [75.0, 65.0, 1.0]],
            ]
        )
        target = {
            "boxes": torch.tensor([[10.0, 20.0, 50.0, 60.0]]),
            "labels": torch.tensor([1]),
            "keypoints": keypoints_in.clone(),
        }

        _, aug_target = wrapper(image, target)

        torch.testing.assert_close(aug_target["keypoints"], keypoints_in)


class TestSyntheticPoseDatasetFixture:
    """Test the session-scoped synthetic_pose_dataset_dir fixture."""

    def test_fixture_creates_splits(self, synthetic_pose_dataset_dir) -> None:
        """Fixture should create train, valid, and test splits."""
        assert (synthetic_pose_dataset_dir / "train").exists()
        assert (synthetic_pose_dataset_dir / "valid").exists()
        assert (synthetic_pose_dataset_dir / "test").exists()

    def test_fixture_annotations_have_keypoints(self, synthetic_pose_dataset_dir) -> None:
        """Annotations in the fixture dataset should have keypoints."""
        with open(synthetic_pose_dataset_dir / "train" / "_annotations.coco.json") as f:
            data = json.load(f)

        assert len(data["annotations"]) > 0
        for ann in data["annotations"]:
            assert "keypoints" in ann
            assert len(ann["keypoints"]) == 17 * 3
