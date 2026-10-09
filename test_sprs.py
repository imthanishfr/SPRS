"""
Comprehensive Test Suite for Smart Packaging Recommendation System (SPRS)
Tests:
1. Bin packing 3D rotation logic & stock box matching
2. Optimal custom box calculation
3. Rules matrix category lookups & fallback
4. Synthetic image detection & dimension calculation
"""

import sys
import unittest
import numpy as np

from sprs.bin_packer import PackingEngine, DEFAULT_STOCK_BOXES
from sprs.rules import get_packaging_rules, RULES_TABLE, CATEGORIES
from sprs.dimensioner import ObjectDimensioner
from sprs.calibration import CalibrationManager


class TestSPRSBinPacker(unittest.TestCase):
    def setUp(self):
        self.packer = PackingEngine(stock_inventory=DEFAULT_STOCK_BOXES)

    def test_optimal_custom_box_calculation(self):
        """Test calculation of exact custom box with buffer clearance."""
        res = self.packer.pack(10.0, 8.0, 5.0, padding_buffer_cm=2.0)
        opt = res["optimal_custom_box"]
        self.assertEqual(opt["length_cm"], 14.0)
        self.assertEqual(opt["width_cm"], 12.0)
        self.assertEqual(opt["height_cm"], 9.0)
        self.assertEqual(opt["volume_cm3"], 1512.0)

    def test_stock_box_rotation_fit(self):
        """Test that objects fit into stock boxes regardless of spatial rotation."""
        res = self.packer.pack(10.0, 2.0, 3.0, padding_buffer_cm=2.0)
        self.assertEqual(res["status"], "FITS_STOCK_BOX")
        self.assertIsNotNone(res["best_stock_box"])
        self.assertEqual(res["best_stock_box"]["box_id"], "BOX-S1")

    def test_oversized_object(self):
        """Test that objects exceeding all stock boxes are flagged as OVERSIZED."""
        res = self.packer.pack(100.0, 80.0, 70.0, padding_buffer_cm=3.0)
        self.assertEqual(res["status"], "OVERSIZED_MANUAL_PACKING_REQUIRED")
        self.assertIsNone(res["best_stock_box"])


class TestSPRSRulesEngine(unittest.TestCase):
    def test_all_categories_exist(self):
        """Verify rules matrix covers all target categories."""
        for cat in CATEGORIES:
            rules = get_packaging_rules(cat)
            self.assertIn("cardboard_type", rules)
            self.assertIn("padding_buffer_cm", rules)
            self.assertIn("filling_material", rules)
            self.assertIn("handling_instructions", rules)
            self.assertGreater(rules["padding_buffer_cm"], 0)

    def test_unmapped_category_fallback(self):
        """Verify unmapped strings fallback gracefully to 'Other/Miscellaneous'."""
        rules = get_packaging_rules("Unknown Weird Category")
        self.assertEqual(rules, RULES_TABLE["Other/Miscellaneous"])


class TestSPRSDimensioner(unittest.TestCase):
    def setUp(self):
        self.dim = ObjectDimensioner(pixels_per_cm=10.0)

    def test_contour_fallback_measurement(self):
        """Test dimensioning on a synthetic test image with a known white rectangle on black background."""
        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        frame[250:350, 200:400] = [255, 255, 255]

        res = self.dim._detect_contour_fallback(frame, 600, 600)
        self.assertTrue(res["detected"])
        dims = res["dimensions"]
        self.assertAlmostEqual(dims["length_cm"], 20.0, delta=1.0)
        self.assertAlmostEqual(dims["width_cm"], 10.0, delta=1.0)


if __name__ == "__main__":
    print("\nExecuting SPRS Automated Test Suite...\n")
    unittest.main()
