from ultralytics import YOLO
import torch


class YOLOv8Model:
    """
    A class to encapsulate YOLOv8 model loading and inference.
    """

    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize the YOLOv8 model.

        Parameters:
            model_path (str): Path to the YOLOv8 model file (e.g., 'yolov8n.pt').
                              The ultralytics library will attempt to download standard models
                              if not found locally.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"YOLOv8 model '{model_path}' loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.model = None

    def detect(self, image):
        """
        Perform object detection on an image.

        Parameters:
            image (numpy.ndarray): The input image in BGR format (as read by OpenCV).

        Returns:
            list: A list of detections. Each detection is a dictionary:
                  {'box': [x1, y1, x2, y2],
                   'confidence': conf_score,
                   'class_id': class_id,
                   'class_name': class_name}
                  Returns an empty list if the model is not loaded or no detections are found.
        """
        if self.model is None:
            print("YOLOv8 model is not loaded. Cannot perform detection.")
            return []

        detections = []
        try:
            results = self.model(
                image, verbose=False
            )  # verbose=False to reduce console output

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]

                    detections.append(
                        {
                            "box": [x1, y1, x2, y2],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": class_name,
                        }
                    )
        except Exception as e:
            print(f"Error during YOLOv8 detection: {e}")
            return []

        return detections


if __name__ == "__main__":
    # Example usage (requires an image file 'test_image.jpg' in the same directory)
    import cv2
    import numpy as np

    # Create a dummy image for testing if 'test_image.jpg' doesn't exist
    dummy_image_path = "test_image.jpg"
    if not cv2.imread(dummy_image_path):
        print(f"'{dummy_image_path}' not found, creating a dummy image for testing.")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            dummy_img,
            "Test",
            (200, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(dummy_image_path, dummy_img)

    img = cv2.imread(dummy_image_path)
    if img is not None:
        yolo_detector = YOLOv8Model()  # Uses yolov8n.pt by default
        if yolo_detector.model:  # Check if model loaded
            detected_objects = yolo_detector.detect(img)

            if detected_objects:
                print(f"Detected {len(detected_objects)} objects:")
                for obj in detected_objects:
                    print(
                        f"  Class: {obj['class_name']}, Confidence: {obj['confidence']:.2f}, Box: {obj['box']}"
                    )
                    x1, y1, x2, y2 = obj["box"]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{obj['class_name']}: {obj['confidence']:.2f}"
                    cv2.putText(
                        img,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("YOLOv8 Detections", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("No objects detected in the test image.")
    else:
        print(
            f"Failed to load '{dummy_image_path}'. Please provide a valid image for testing."
        )
