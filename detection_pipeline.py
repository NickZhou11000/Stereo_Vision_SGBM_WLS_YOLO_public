import cv2
import numpy as np
import sys
from enhanced_stereo_vision import (
    EnhancedStereoVision,
)  # Ensure this file is in the same dir or PYTHONPATH
from yolo_v8_model import (
    YOLOv8Model,
)  # Ensure this file is in the same dir or PYTHONPATH


class DetectionPipeline:
    def __init__(
        self, calibration_file="enhanced_stereo_map.xml", yolo_model_path="yolov8n.pt"
    ):
        """
        Initializes the Detection Pipeline.
        Args:
            calibration_file (str): Path to the stereo calibration file.
            yolo_model_path (str): Path to the YOLOv8 model file.
        """
        self.calibration_file = calibration_file
        self.yolo_model_path = yolo_model_path
        self.stereo_system = None
        self.yolo_detector = None
        self.processed_left_image = None
        self.depth_map = None
        self.last_detections_details = []  # Added to store last detection details
        self.last_center_roi_coeff = None  # Added to store the last used coefficient
        # print(f"DetectionPipeline initialized. Calib: {self.calibration_file}, YOLO: {self.yolo_model_path}")

    def setup_stereo(self):
        """
        Sets up the stereo vision system.
        Returns: bool: True if setup was successful, False otherwise.
        """
        print("Initializing Stereo Vision System...")
        try:
            self.stereo_system = EnhancedStereoVision(
                calibration_file=self.calibration_file
            )
            print("Stereo Vision System initialized successfully.")
            return True
        except Exception as e_stereo:
            print(f"Error initializing Stereo Vision System: {e_stereo}")
            self.stereo_system = None
            return False

    def setup_yolo(self):
        """
        Sets up the YOLOv8 object detection model.
        Returns: bool: True if setup was successful, False otherwise.
        """
        print("Initializing YOLOv8 Model...")
        try:
            self.yolo_detector = YOLOv8Model(model_path=self.yolo_model_path)
            if self.yolo_detector.model is None:
                print(
                    "Failed to load YOLOv8 model (model attribute is None after init)."
                )
                self.yolo_detector = None  # Ensure consistent state
                return False
            print("YOLOv8 Model initialized successfully.")
            return True
        except Exception as e_yolo:
            print(f"Error initializing YOLOv8 Model: {e_yolo}")
            self.yolo_detector = None
            return False

    def _calculate_mean_depth_full_box(self, box):
        if self.depth_map is None:
            return 0.0
        x1, y1, x2, y2 = map(int, box)  # Ensure integer coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.depth_map.shape[1], x2), min(self.depth_map.shape[0], y2)
        if x1 >= x2 or y1 >= y2:  # Invalid ROI
            return 0.0
        depth_roi = self.depth_map[y1:y2, x1:x2]
        valid_depth_pixels = depth_roi[depth_roi > 0]
        if valid_depth_pixels.size > 0:
            return np.mean(valid_depth_pixels) / 1000.0  # Convert mm to m
        return 0.0

    def _calculate_mean_depth_center_sq_roi(self, box, coefficient):
        if self.depth_map is None or coefficient <= 0:
            return 0.0, None
        x1, y1, x2, y2 = map(int, box)
        box_width = x2 - x1
        box_height = y2 - y1
        if box_width <= 0 or box_height <= 0:
            return 0.0, None
        roi_size = int(min(box_width, box_height) * coefficient)
        if roi_size <= 0:
            return 0.0, None
        center_x = x1 + box_width // 2
        center_y = y1 + box_height // 2
        roi_x1 = max(0, center_x - roi_size // 2)
        roi_y1 = max(0, center_y - roi_size // 2)
        roi_x2 = min(self.depth_map.shape[1], center_x + roi_size // 2)
        roi_y2 = min(self.depth_map.shape[0], center_y + roi_size // 2)
        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
            return 0.0, None
        center_depth_roi = self.depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        valid_center_depth_pixels = center_depth_roi[center_depth_roi > 0]
        mean_depth = 0.0
        if valid_center_depth_pixels.size > 0:
            mean_depth = np.mean(valid_center_depth_pixels) / 1000.0  # Convert mm to m
        return mean_depth, [roi_x1, roi_y1, roi_x2, roi_y2]

    def process_frame(
        self,
        left_image_input,
        right_image_input,
        center_roi_coeff=0.1,
        min_confidence_threshold=0.3,
    ):
        if not self.stereo_system or not self.yolo_detector:
            print("Error: Stereo system or YOLO detector not initialized.")
            return None, [], None

        self.last_center_roi_coeff = center_roi_coeff  # Store the coefficient

        try:
            if isinstance(left_image_input, str) and isinstance(right_image_input, str):
                self.stereo_system.load_images(left_image_input, right_image_input)
            elif isinstance(left_image_input, np.ndarray) and isinstance(
                right_image_input, np.ndarray
            ):
                self.stereo_system.left_image = left_image_input.copy()
                self.stereo_system.right_image = right_image_input.copy()
                self.stereo_system._process_images()  # Rectify and crop
            else:
                print(
                    "Error: Invalid image input. Provide paths (str) or NumPy arrays."
                )
                return None, [], None
        except Exception as e_img_load:
            print(f"Error processing input images: {e_img_load}")
            return None, [], None

        self.processed_left_image = self.stereo_system.get_processed_left_image()
        if self.processed_left_image is None:
            print("Failed to get processed left image.")
            return None, [], None

        self.stereo_system.get_disparity_map(interactive=False)
        self.depth_map = self.stereo_system.get_depth_map()
        # YOLO detection can proceed even if depth_map is None, depth values will be N/A

        detections = self.yolo_detector.detect(self.processed_left_image)
        detections_with_depth = []
        if detections:
            for det in detections:
                if det["confidence"] >= min_confidence_threshold:
                    box = det["box"]
                    # mean_depth_full_m = 0.0 # Removed: No longer calculating full box depth
                    mean_depth_center_m = 0.0
                    center_roi_box_coords = None
                    if self.depth_map is not None:
                        # mean_depth_full_m = self._calculate_mean_depth_full_box(box) # Removed
                        mean_depth_center_m, center_roi_box_coords = (
                            self._calculate_mean_depth_center_sq_roi(
                                box, center_roi_coeff
                            )
                        )
                    det_info = det.copy()
                    det_info["depth_full_box_m"] = (
                        "N/A (Not Calculated)"  # Indicate full box depth is not calculated
                    )
                    det_info["depth_center_roi_m"] = (
                        f"{mean_depth_center_m:.2f}m"
                        if mean_depth_center_m > 0 and center_roi_coeff > 0
                        else ("N/A" if center_roi_coeff > 0 else "Off")
                    )
                    det_info["center_roi_box_coords"] = center_roi_box_coords
                    detections_with_depth.append(det_info)

        self.last_detections_details = (
            detections_with_depth  # Store for later retrieval
        )
        return self.processed_left_image.copy(), detections_with_depth, self.depth_map

    def draw_detections(
        self, image_to_draw_on, detections_with_depth, center_roi_coeff=0.1
    ):
        output_image = image_to_draw_on.copy()
        for det in detections_with_depth:
            box = det["box"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            center_roi_box_coords = det.get("center_roi_box_coords")
            if center_roi_coeff > 0 and center_roi_box_coords:
                cx1, cy1, cx2, cy2 = map(int, center_roi_box_coords)
                if cx2 > cx1 and cy2 > cy1:
                    cv2.rectangle(output_image, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)
            label = f"{class_name} ({confidence * 100:.0f}%)"
            # depth_text_full = f"FullBoxD: {det['depth_full_box_m']}" # Removed
            depth_text_center = f"CenterD: {det['depth_center_roi_m']}"

            text_y = y1 - 7
            cv2.putText(
                output_image,
                depth_text_center,
                (x1, max(15, text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )
            text_y -= 18
            # cv2.putText(output_image, depth_text_full, (x1, max(15, text_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,165,255), 1, cv2.LINE_AA) # Removed
            # text_y -= 18 # Adjusted: No longer needed if full box depth text is removed
            cv2.putText(
                output_image,
                label,
                (x1, max(15, text_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return output_image

    def get_last_detection_details(self):
        """
        Prints and returns the details of objects detected in the last processed frame.
        Each item in the list is a dictionary containing detection information
        (class_name, box, confidence) and calculated center ROI depth information.
        Full box depth is no longer calculated.
        """
        latest_detections = self.last_detections_details
        if latest_detections:
            print("\\nDetailed Detections from get_last_detection_details():")
            if (
                self.last_center_roi_coeff is not None
            ):  # Check if process_frame has been run
                print(
                    f"  (Using Center ROI Coefficient: {self.last_center_roi_coeff if self.last_center_roi_coeff > 0 else 'Off'})"
                )
            for detection in latest_detections:
                print(
                    f"Detected: {detection['class_name']} (Confidence: {detection['confidence']:.2f})"
                )
                print(f"  Bounding Box: {detection['box']}")
                # print(f"  Depth (Full Box): {detection['depth_full_box_m']}") # Removed
                print(f"  Depth (Center ROI): {detection['depth_center_roi_m']}")
                if (
                    detection.get("center_roi_box_coords")
                    and self.last_center_roi_coeff > 0
                ):  # Also check if ROI was active
                    print(f"  Center ROI Coords: {detection['center_roi_box_coords']}")
        else:
            print(
                "No detections were found in the last processed frame (from get_last_detection_details)."
            )
        return latest_detections

    def print_last_detection_details(self):
        """
        Prints and returns the details of objects detected in the last processed frame.
        Each item in the list is a dictionary containing detection information
        (class_name, box, confidence) and calculated center ROI depth information.
        Full box depth is no longer calculated.
        """
        latest_detections = self.last_detections_details
        if latest_detections:
            print("\\nDetailed Detections (from print_last_detection_details):")
            if (
                self.last_center_roi_coeff is not None
            ):  # Check if process_frame has been run
                print(
                    f"  (Using Center ROI Coefficient: {self.last_center_roi_coeff if self.last_center_roi_coeff > 0 else 'Off'})"
                )
            for detection in latest_detections:
                print(
                    f"  Detected: {detection['class_name']} (Confidence: {detection['confidence']:.2f})"
                )
                print(f"    Bounding Box: {detection['box']}")
                # print(f"    Depth (Full Box): {detection['depth_full_box_m']}") # Removed
                print(f"    Depth (Center ROI): {detection['depth_center_roi_m']}")
                if (
                    detection.get("center_roi_box_coords")
                    and self.last_center_roi_coeff > 0
                ):  # Also check if ROI was active
                    print(
                        f"    Center ROI Coords: {detection['center_roi_box_coords']}"
                    )
        else:
            print(
                "No detections were found in the last processed frame (from print_last_detection_details)."
            )
        return latest_detections


if __name__ == "__main__":
    print("Starting Detection Pipeline example...")
    # Configuration - Update paths if necessary
    CALIBRATION_FILE = "enhanced_stereo_map.xml"
    LEFT_IMAGE_PATH = "./dataset/left/L22.png"
    RIGHT_IMAGE_PATH = "./dataset/right/R22.png"
    YOLO_MODEL_PATH = "yolov8n.pt"  # Standard model, ultralytics lib will try to download if not found
    MIN_CONFIDENCE_MAIN = 0.3

    pipeline = DetectionPipeline(
        calibration_file=CALIBRATION_FILE, yolo_model_path=YOLO_MODEL_PATH
    )

    if not pipeline.setup_stereo():
        print("Exiting: Stereo setup failed.")
        sys.exit(1)
    if not pipeline.setup_yolo():
        print("Exiting: YOLO setup failed.")
        sys.exit(1)

    print(f"Processing images: {LEFT_IMAGE_PATH}, {RIGHT_IMAGE_PATH}")
    processed_img_base, detections_data, d_map = pipeline.process_frame(
        LEFT_IMAGE_PATH,
        RIGHT_IMAGE_PATH,
        center_roi_coeff=0.1,  # Example: 10% center ROI for depth calculation
        min_confidence_threshold=MIN_CONFIDENCE_MAIN,
    )

    if processed_img_base is not None:
        print(f"Frame processed. Detections: {len(detections_data)}")
        pipeline.print_last_detection_details()
        output_display_image = pipeline.draw_detections(
            processed_img_base,
            detections_data,
            center_roi_coeff=0.1,  # For drawing the center ROI box
        )
        try:
            cv2.imshow("Detection Pipeline Output", output_display_image)
            print("Displaying output. Press any key in the OpenCV window to close.")
            cv2.waitKey(0)
        except Exception as e_display:
            print(
                f"cv2.imshow failed: {e_display}. This can happen in headless environments."
            )
            output_filename = "detection_output.png"
            try:
                cv2.imwrite(output_filename, output_display_image)
                print(f"Output image saved to {output_filename}")
            except Exception as e_save:
                print(f"Failed to save image to {output_filename}: {e_save}")
    else:
        print("Frame processing failed. No output to display or save.")

    cv2.destroyAllWindows()
    print("Detection pipeline example finished.")
    # Added a newline here for the tool
