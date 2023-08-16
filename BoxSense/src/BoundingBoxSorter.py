# Built-in
import math
import logging

# Third Party
import numpy as np


class BoundingBoxSorter:
    @staticmethod
    def calculateCenter(box: tuple) -> tuple:
        """Calculates the center coordinates of a bounding box.

        Args:
            box (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).

        Returns:
            tuple: Center coordinates (center_x, center_y).
        """
        xMin, yMin, xMax, yMax = box
        centerX = (xMin + xMax) / 2
        centerY = (yMin + yMax) / 2
        return centerX, centerY

    @staticmethod
    def calculateDistance(point1: tuple, point2: tuple) -> float:
        """Calculates the Euclidean distance between two points.

        Args:
            point1 (tuple): First point coordinates (x1, y1).
            point2 (tuple): Second point coordinates (x2, y2).

        Returns:
            float: Euclidean distance.
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def sortBoundedItems(image: np.ndarray, boundingBoxes: list) -> list:
        """Sorts bounded items based on their distance from the image center.

        Args:
            image (np.ndarray): Image array (HxWxC).
            boundingBoxes (list): List of bounding boxes [(x_min, y_min, x_max, y_max)].

        Returns:
            list: Sorted list of bounding boxes.
        """
        imageCenter = (image.shape[1] / 2, image.shape[0] / 2)
        sortedItems = sorted(
            boundingBoxes,
            key=lambda box: BoundingBoxSorter.calculateDistance(
                imageCenter, BoundingBoxSorter.calculateCenter(box)
            ),
        )
        return sortedItems


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image = np.zeros((800, 800, 3), dtype=np.uint8)  # Example image
    boundingBoxes = [
        (100, 100, 200, 200),
        (300, 300, 400, 400),
        (600, 600, 700, 700),
    ]  # Example bounding boxes

    sortedBoundedItems = BoundingBoxSorter.sortBoundedItems(image, boundingBoxes)
    logging.info("Sorted Bounded Items: %s", sortedBoundedItems)
