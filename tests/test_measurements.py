"""Tests for utils.measurements module."""

import numpy as np
import pytest

from utils.measurements import (
    calculate_dice_score,
    calculate_volumes,
    format_measurements_for_report,
    get_bounding_boxes,
    measure_lesion_characteristics,
)


# ------------------------------------------------------------------
# calculate_volumes
# ------------------------------------------------------------------


class TestCalculateVolumes:
    """Tests for calculate_volumes."""

    def test_single_label(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1  # 27 voxels
        volumes = calculate_volumes(mask, spacing=(1.0, 1.0, 1.0))
        assert "label_1" in volumes
        assert volumes["label_1"] == pytest.approx(27.0)

    def test_multiple_labels(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:2, 0:2, 0:2] = 1  # 8 voxels
        mask[5:7, 5:7, 5:7] = 2  # 8 voxels
        volumes = calculate_volumes(mask)
        assert len(volumes) == 2
        assert volumes["label_1"] == pytest.approx(8.0)
        assert volumes["label_2"] == pytest.approx(8.0)

    def test_with_organ_labels(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:2, 0:2, 0:2] = 1
        mask[5:7, 5:7, 5:7] = 2
        labels = {1: "liver", 2: "kidney"}
        volumes = calculate_volumes(mask, organ_labels=labels)
        assert "liver" in volumes
        assert "kidney" in volumes

    def test_with_spacing(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:2, 0:2, 0:2] = 1  # 8 voxels
        volumes = calculate_volumes(mask, spacing=(2.0, 2.0, 2.0))
        assert volumes["label_1"] == pytest.approx(8 * 8.0)  # 8 * 2^3

    def test_empty_mask(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        volumes = calculate_volumes(mask)
        assert len(volumes) == 0

    def test_background_excluded(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        volumes = calculate_volumes(mask)
        assert 0 not in volumes
        assert "label_0" not in volumes


# ------------------------------------------------------------------
# get_bounding_boxes
# ------------------------------------------------------------------


class TestGetBoundingBoxes:
    """Tests for get_bounding_boxes."""

    def test_basic_bbox(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[3:6, 3:6, 3:6] = 1
        bboxes = get_bounding_boxes(mask)
        assert bboxes["label_1"]["min"] == (3, 3, 3)
        assert bboxes["label_1"]["max"] == (5, 5, 5)

    def test_dimensions(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:10, 2:8, 0:4] = 1
        bboxes = get_bounding_boxes(mask)
        dims = bboxes["label_1"]["dimensions_mm"]
        assert dims == (5, 6, 4)

    def test_with_organ_labels(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[0:3, 0:3, 0:3] = 1
        bboxes = get_bounding_boxes(mask, organ_labels={1: "spleen"})
        assert "spleen" in bboxes


# ------------------------------------------------------------------
# calculate_dice_score
# ------------------------------------------------------------------


class TestDiceScore:
    """Tests for calculate_dice_score."""

    def test_perfect_overlap(self):
        mask = np.ones((5, 5, 5))
        assert calculate_dice_score(mask, mask) > 0.99

    def test_no_overlap(self):
        pred = np.zeros((5, 5, 5))
        gt = np.ones((5, 5, 5))
        assert calculate_dice_score(pred, gt) < 0.01

    def test_partial_overlap(self):
        pred = np.zeros((10, 10, 10))
        gt = np.zeros((10, 10, 10))
        pred[0:5, 0:5, 0:5] = 1
        gt[3:8, 3:8, 3:8] = 1
        score = calculate_dice_score(pred, gt)
        assert 0.0 < score < 1.0


# ------------------------------------------------------------------
# measure_lesion_characteristics
# ------------------------------------------------------------------


class TestLesionCharacteristics:
    """Tests for measure_lesion_characteristics."""

    def test_basic_lesion(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:10, 5:10, 5:10] = 1
        result = measure_lesion_characteristics(mask)
        assert result["volume_mm3"] > 0
        assert result["volume_cm3"] == pytest.approx(
            result["volume_mm3"] / 1000.0
        )
        assert "max_diameter_mm" in result

    def test_sphericity_exists(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        result = measure_lesion_characteristics(mask)
        assert "sphericity" in result
        assert result["sphericity"] >= 0


# ------------------------------------------------------------------
# format_measurements_for_report
# ------------------------------------------------------------------


class TestFormatMeasurements:
    """Tests for format_measurements_for_report."""

    def test_volumes_format(self):
        measurements = {
            "volumes_mm3": {"liver": 150000.0, "spleen": 20000.0}
        }
        text = format_measurements_for_report(measurements)
        assert "liver" in text
        assert "spleen" in text
        assert "cm3" in text

    def test_empty_measurements(self):
        text = format_measurements_for_report({})
        assert text == ""
