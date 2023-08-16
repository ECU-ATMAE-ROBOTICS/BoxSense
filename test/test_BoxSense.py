import unittest
import numpy as np
from PIL import Image
from BoxSense.BoxSensor import BoxSensor
from BoxSense.src.BoundingBoxSorter import BoundingBoxSorter


class TestBoxSensor(unittest.TestCase):
    def setUp(self):
        self.detector = BoxSensor("src/model/box-model.pt")

    def test_detect_objects(self):
        image_path = "src/data/boxes-on-pallets.jpg"
        image = np.array(Image.open(image_path))
        detected_boxes = self.detector.detectObjects(image)
        self.assertTrue(isinstance(detected_boxes, list))
        self.assertTrue(len(detected_boxes) > 0)

    # You can add more test cases here


class TestBoundingBoxSorter(unittest.TestCase):
    def test_calculate_center(self):
        box = (100, 200, 300, 400)
        center = BoundingBoxSorter.calculateCenter(box)
        self.assertEqual(center, (200, 300))

    def test_calculate_distance(self):
        point1 = (0, 0)
        point2 = (3, 4)
        distance = BoundingBoxSorter.calculateDistance(point1, point2)
        self.assertAlmostEqual(distance, 5.0)

    def test_sort_bounded_items(self):
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        bounding_boxes = [
            (100, 100, 200, 200),
            (300, 300, 400, 400),
            (600, 600, 700, 700),
        ]
        sorted_items = BoundingBoxSorter.sortBoundedItems(image, bounding_boxes)
        self.assertEqual(sorted_items, bounding_boxes)

    # You can add more test cases here


if __name__ == "__main__":
    unittest.main()
