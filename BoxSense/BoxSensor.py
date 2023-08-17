# Built-in
import logging

# Third Party
import torch
import cv2
import numpy as np

# Internal
from src.BoundingBoxSorter import BoundingBoxSorter


class BoxSensor:
    def __init__(self, modelPath: str):
        """Initializes the BoxSensor class.

        Args:
            modelPath (str): Path to the model checkpoint file.
        """
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=modelPath,
            verbose=False,
        )

        self.model.eval()

        self.model.conf = 0.75  # NMS confidence threshold

        self.logger = logging.getLogger(__name__)

    def detectBoxes(self, image: np.ndarray) -> list:
        """Detects objects in the provided image.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            list: List of bounding box coordinates.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        with torch.no_grad():
            predictions = self.model(image)

        return predictions.xyxy[0].cpu().numpy()

    def visualizeBoxes(self, image: np.ndarray, boxes: list) -> None:
        """Visualizes detected boxes on the input image.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            boxes (list): List of bounding box coordinates.
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        imageCV2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for box in boxes:
            xMin, yMin, xMax, yMax, conf, _ = box
            cv2.rectangle(
                imageCV2,
                (int(xMin), int(yMin)),
                (int(xMax), int(yMax)),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Detected Boxes", imageCV2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = BoxSensor("BoxSense/src/model/box-model.pt")
    sorter = BoundingBoxSorter()

    imagePath = "BoxSense/src/data/boxes-on-pallets.jpeg"
    imageArray = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)

    detectedBoxes = detector.detectBoxes(imageArray)
    logging.info("Detected Boxes: %s", detectedBoxes)

    detector.visualizeBoxes(imageArray, detectedBoxes)

    sortedBoundedItems = BoundingBoxSorter.sortBoundedItemsByDepth(
        imageArray, detectedBoxes
    )
    logging.info("Sorted Bounded Items: %s", sortedBoundedItems)

    BoundingBoxSorter.displaySortedItems(imageArray, sortedBoundedItems)
