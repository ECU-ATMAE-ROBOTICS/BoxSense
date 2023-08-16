# Built-in
import logging

# Third Party
import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

# Internal
from src.BoundingBoxSorter import BoundingBoxSorter


class BoxSensor:
    def __init__(self, modelPath: str):
        """Initializes the BoxSensor class.

        Args:
            modelPath (str): Path to the model checkpoint file.
        """
        self.model = torch.load(modelPath)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
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

        imagePIL = Image.fromarray(image)
        inputImage = self.transform(imagePIL).unsqueeze(0)

        with torch.no_grad():
            predictions = self.model(inputImage)

        boxes = predictions[0]["boxes"].tolist()
        return boxes

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
            xMin, yMin, xMax, yMax = box
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
    detector = BoxSensor("src/model/box-model.pt")
    sorter = BoundingBoxSorter()

    imagePath = "src/data/boxes-on-pallets.jpg"
    imageArray = cv2.cvtColor(cv2.imread(imagePath), cv2.COLOR_BGR2RGB)

    detectedBoxes = detector.detectBoxes(imageArray)
    logging.info("Detected Boxes: %s", detectedBoxes)

    detector.visualizeBoxes(imageArray, detectedBoxes)

    sortedBoundedItems = BoundingBoxSorter.sortBoundedItems(imageArray, detectedBoxes)
    logging.info("Sorted Bounded Items: %s", sortedBoundedItems)
