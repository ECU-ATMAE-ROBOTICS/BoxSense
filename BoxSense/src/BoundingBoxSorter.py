import numpy as np
import cv2

from monodepth2 import monodepth2


class BoundingBoxSorter:
    depthdepthModel = monodepth2()

    @staticmethod
    def sortBoundedItemsByDepth(image: np.ndarray, detections: list) -> list:
        """Sorts bounded items by their estimated depth from the camera.

        Args:
            image (np.ndarray): Image array (HxWxC).
            detections (list): List of detections [(x_min, y_min, x_max, y_max, conf, _)].

        Returns:
            list: Sorted list of bounding boxes.
        """
        sortedItems = sorted(
            detections, key=lambda box: BoundingBoxSorter.calculateDepth(image, box)
        )
        return sortedItems

    @staticmethod
    def calculateDepth(image: np.ndarray, box: tuple) -> float:
        x_min, y_min, x_max, y_max, conf, _ = box

        input_image = image[y_min:y_max, x_min:x_max]

        depth = BoundingBoxSorter.depthModel.eval(input_image)

        average_depth = depth.mean()

        return average_depth

    @staticmethod
    def displaySortedItems(image: np.ndarray, sortedDetections: list) -> None:
        """Displays sorted bounding box items on the image with order numbers.

        Args:
            image (np.ndarray): Image array (HxWxC).
            sortedDetections (list): List of sorted detections [(x_min, y_min, x_max, y_max, conf, _)].
        """
        imageWithBoxes = image.copy()
        for idx, box in enumerate(sortedDetections, start=1):
            xMin, yMin, xMax, yMax, conf, _ = box
            boxWidth = xMax - xMin
            boxHeight = yMax - yMin

            cv2.rectangle(
                imageWithBoxes,
                (int(xMin), int(yMin)),
                (int(xMax), int(yMax)),
                (0, 255, 0),
                2,
            )

            text_size = cv2.getTextSize(str(idx), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = int(xMin + (boxWidth - text_size[0]) / 2)
            text_y = int(yMin + (boxHeight + text_size[1]) / 2)

            cv2.putText(
                imageWithBoxes,
                str(idx),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        imageWithBoxesRGB = cv2.cvtColor(imageWithBoxes, cv2.COLOR_BGR2RGB)

        cv2.imshow("Sorted Bounded Items", imageWithBoxesRGB)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
