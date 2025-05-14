import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt


class EnhancedStereoCalibration:
    """
    An enhanced class for stereo camera calibration that provides functionality for calibrating stereo cameras,
    analyzing calibration errors, saving calibration parameters, and handling consistent image cropping.
    """

    # Class constants
    TERM_CRITERIA_EPS = cv2.TERM_CRITERIA_EPS
    TERM_CRITERIA_MAX_ITER = cv2.TERM_CRITERIA_MAX_ITER
    CALIB_CB_ADAPTIVE_THRESH = cv2.CALIB_CB_ADAPTIVE_THRESH
    CALIB_CB_FAST_CHECK = cv2.CALIB_CB_FAST_CHECK
    CALIB_CB_NORMALIZE_IMAGE = cv2.CALIB_CB_NORMALIZE_IMAGE
    CALIB_FIX_ASPECT_RATIO = cv2.CALIB_FIX_ASPECT_RATIO
    CALIB_USE_INTRINSIC_GUESS = cv2.CALIB_USE_INTRINSIC_GUESS
    CALIB_SAME_FOCAL_LENGTH = cv2.CALIB_SAME_FOCAL_LENGTH
    CALIB_ZERO_TANGENT_DIST = cv2.CALIB_ZERO_TANGENT_DIST
    CALIB_RATIONAL_MODEL = cv2.CALIB_RATIONAL_MODEL
    CALIB_FIX_K1 = cv2.CALIB_FIX_K1
    CALIB_FIX_K2 = cv2.CALIB_FIX_K2
    CALIB_FIX_K3 = cv2.CALIB_FIX_K3
    CALIB_FIX_K4 = cv2.CALIB_FIX_K4
    CALIB_FIX_K5 = cv2.CALIB_FIX_K5
    CALIB_FIX_K6 = cv2.CALIB_FIX_K6
    CV_16SC2 = cv2.CV_16SC2

    def __init__(
        self, left_img_folder="./dataset/left/", right_img_folder="./dataset/right/"
    ):
        """
        Initialize the EnhancedStereoCalibration object with default parameters.

        Parameters:
            left_img_folder (str): Path to folder containing left camera images
            right_img_folder (str): Path to folder containing right camera images
        """
        # Paths to calibration images
        self.left_img_folder = left_img_folder
        self.right_img_folder = right_img_folder

        # Calibration parameters
        self.chessboard_size = (13, 9)
        self.frame_size = (1640, 1232)
        self.square_size_mm = 20

        # Criteria for corner refinement
        self.criteria = (
            self.TERM_CRITERIA_EPS + self.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        # Arrays to store object points and image points
        self.obj_points = []  # 3D points in real world space
        self.img_points_l = []  # 2D points in left image plane
        self.img_points_r = []  # 2D points in right image plane

        # Flags for stereo calibration
        self.flags = 0
        self.flags |= self.CALIB_FIX_ASPECT_RATIO
        self.flags |= self.CALIB_SAME_FOCAL_LENGTH
        self.flags |= self.CALIB_ZERO_TANGENT_DIST

        # Calibration results
        self.camera_matrix_l = None
        self.dist_l = None
        self.camera_matrix_r = None
        self.dist_r = None
        self.new_camera_matrix_l = None
        self.new_camera_matrix_r = None
        self.rot = None
        self.trans = None
        self.essential_matrix = None
        self.fundamental_matrix = None
        self.rect_l = None
        self.rect_r = None
        self.proj_matrix_l = None
        self.proj_matrix_r = None
        self.q = None
        self.roi_l = None
        self.roi_r = None
        self.stereo_map_l = None
        self.stereo_map_r = None

        # Error metrics
        self.errors_l = []
        self.errors_r = []
        self.mean_error_l = 0
        self.mean_error_r = 0
        self.max_error_l = 0
        self.max_error_r = 0
        self.max_error_idx_l = 0
        self.max_error_idx_r = 0

        # Sample calibration images (for before/after display)
        self.sample_image_l = None
        self.sample_image_r = None
        self.sample_rectified_l = None
        self.sample_rectified_r = None

        # Enhanced cropping information
        self.crop_rect = None  # Will store [x, y, width, height]

        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(12,8,0)
        self.prepare_object_points()

    def prepare_object_points(self):
        """
        Prepare object points for calibration (points on chessboard).
        """
        self.objp = np.zeros(
            (self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32
        )
        self.objp[:, :2] = np.mgrid[
            0 : self.chessboard_size[0], 0 : self.chessboard_size[1]
        ].T.reshape(-1, 2)
        # Scale to actual size in mm
        self.objp = self.objp * self.square_size_mm

    def load_images(self):
        """
        Load calibration images and find chessboard corners.
        """
        # Clear previous data
        self.obj_points = []
        self.img_points_l = []
        self.img_points_r = []

        # Get lists of image paths
        images_l = sorted(glob.glob(os.path.join(self.left_img_folder, "*.png")))
        images_r = sorted(glob.glob(os.path.join(self.right_img_folder, "*.png")))

        if not images_l or not images_r:
            raise ValueError(
                f"No images found in {self.left_img_folder} or {self.right_img_folder}"
            )

        if len(images_l) != len(images_r):
            raise ValueError("Number of left and right images do not match")

        print(f"Found {len(images_l)} image pairs. Processing...")
        success_count = 0

        # Keep one image pair for before/after comparison
        sample_idx = len(images_l) // 2  # Use middle image as sample

        # Process each image pair
        for i, (image_path_l, image_path_r) in enumerate(zip(images_l, images_r)):
            img_l = cv2.imread(image_path_l)
            img_r = cv2.imread(image_path_r)
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

            # Save sample images
            if i == sample_idx:
                self.sample_image_l = img_l.copy()
                self.sample_image_r = img_r.copy()

            # Find chessboard corners
            ret_l, corners_l = cv2.findChessboardCorners(
                gray_l,
                self.chessboard_size,
                flags=self.CALIB_CB_ADAPTIVE_THRESH
                + self.CALIB_CB_FAST_CHECK
                + self.CALIB_CB_NORMALIZE_IMAGE,
            )
            ret_r, corners_r = cv2.findChessboardCorners(
                gray_r,
                self.chessboard_size,
                flags=self.CALIB_CB_ADAPTIVE_THRESH
                + self.CALIB_CB_FAST_CHECK
                + self.CALIB_CB_NORMALIZE_IMAGE,
            )

            # If both found, refine and add to calibration data
            if ret_l and ret_r:
                # Refine corner locations
                cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), self.criteria)
                cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)

                # Add points to collections
                self.obj_points.append(self.objp)
                self.img_points_l.append(corners_l)
                self.img_points_r.append(corners_r)

                # Draw corners on sample images
                if i == sample_idx:
                    cv2.drawChessboardCorners(
                        self.sample_image_l, self.chessboard_size, corners_l, ret_l
                    )
                    cv2.drawChessboardCorners(
                        self.sample_image_r, self.chessboard_size, corners_r, ret_r
                    )

                    # Display sample images with corners
                    cv2.imshow("Left Corners", self.sample_image_l)
                    cv2.imshow("Right Corners", self.sample_image_r)
                    cv2.waitKey(500)  # Show for half a second

                success_count += 1

        cv2.destroyAllWindows()
        print(
            f"Successfully processed {success_count} out of {len(images_l)} image pairs"
        )

        # Store the gray images for later use
        self.gray_l = gray_l
        self.gray_r = gray_r

    def calibrate_mono_cameras(self):
        """
        Calibrate each camera individually and calculate reprojection errors.
        """
        if not self.obj_points:
            raise ValueError("No calibration data. Run load_images() first.")

        # Calibrate left camera
        ret_l, self.camera_matrix_l, self.dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self.obj_points, self.img_points_l, self.frame_size, None, None
        )

        # Calibrate right camera
        ret_r, self.camera_matrix_r, self.dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self.obj_points, self.img_points_r, self.frame_size, None, None
        )

        print("\nCamera Calibration Results:")
        print("-" * 50)
        print("Left Camera Matrix:")
        print(self.camera_matrix_l)
        print("\nLeft Camera Distortion:")
        print(self.dist_l)
        print("\nRight Camera Matrix:")
        print(self.camera_matrix_r)
        print("\nRight Camera Distortion:")
        print(self.dist_r)

        # Calculate reprojection errors
        self.calculate_reprojection_errors(rvecs_l, tvecs_l, rvecs_r, tvecs_r)

    def calibrate_stereo(self):
        """
        Perform stereo calibration to find the relationship between cameras.
        """
        if self.camera_matrix_l is None or self.camera_matrix_r is None:
            raise ValueError(
                "Mono camera calibration required. Run calibrate_mono_cameras() first."
            )

        # Stereo calibration criteria
        criteria_stereo = (
            self.TERM_CRITERIA_EPS + self.TERM_CRITERIA_MAX_ITER,
            100,
            0.00001,
        )

        # Perform stereo calibration
        (
            ret_stereo,
            self.new_camera_matrix_l,
            self.dist_l,
            self.new_camera_matrix_r,
            self.dist_r,
            self.rot,
            self.trans,
            self.essential_matrix,
            self.fundamental_matrix,
        ) = cv2.stereoCalibrate(
            self.obj_points,
            self.img_points_l,
            self.img_points_r,
            self.camera_matrix_l,
            self.dist_l,
            self.camera_matrix_r,
            self.dist_r,
            self.gray_l.shape[::-1],
            criteria=criteria_stereo,
            flags=self.flags,
        )

        print("\nStereo Calibration Results:")
        print("-" * 50)
        print(f"Left Camera Matrix:\n{self.new_camera_matrix_l}")
        print(f"\nLeft Camera Distortion:\n{self.dist_l}")
        print(f"\nRight Camera Matrix:\n{self.new_camera_matrix_r}")
        print(f"\nRight Camera Distortion:\n{self.dist_r}")
        print(f"\nRotation Matrix:\n{self.rot}")
        print(f"\nTranslation Vector:\n{self.trans}")
        print(f"\nEssential Matrix:\n{self.essential_matrix}")
        print(f"\nFundamental Matrix:\n{self.fundamental_matrix}")

    def stereo_rectify(self):
        """
        Compute stereo rectification and generate rectification maps.
        """
        if self.rot is None or self.trans is None:
            raise ValueError(
                "Stereo calibration required. Run calibrate_stereo() first."
            )

        # Stereo rectification
        rectify_scale = 0  # 0=full crop (alpha=0), 1=no crop (alpha=1)
        (
            self.rect_l,
            self.rect_r,
            self.proj_matrix_l,
            self.proj_matrix_r,
            self.q,
            self.roi_l,
            self.roi_r,
        ) = cv2.stereoRectify(
            self.new_camera_matrix_l,
            self.dist_l,
            self.new_camera_matrix_r,
            self.dist_r,
            self.gray_l.shape[::-1],  # size
            self.rot,
            self.trans,
            alpha=rectify_scale,  # Use alpha for clarity, equivalent to rectifyScale=0
            newImageSize=(0, 0),  # Same as input image size
        )

        # Generate rectification maps
        self.stereo_map_l = cv2.initUndistortRectifyMap(
            self.new_camera_matrix_l,
            self.dist_l,
            self.rect_l,
            self.proj_matrix_l,
            self.gray_l.shape[::-1],
            self.CV_16SC2,
        )
        self.stereo_map_r = cv2.initUndistortRectifyMap(
            self.new_camera_matrix_r,
            self.dist_r,
            self.rect_r,
            self.proj_matrix_r,
            self.gray_r.shape[::-1],
            self.CV_16SC2,
        )

        print("\nStereo Rectification Complete!")

        # Determine the crop rectangle using ROI intersection first
        # roi_l and roi_r are (x, y, w, h)
        x_l, y_l, w_l, h_l = map(int, self.roi_l)
        x_r, y_r, w_r, h_r = map(int, self.roi_r)

        inter_x = max(x_l, x_r)
        inter_y = max(y_l, y_r)

        # Calculate the rightmost and bottommost coordinates for each ROI
        x_l_prime = x_l + w_l
        y_l_prime = y_l + h_l
        x_r_prime = x_r + w_r
        y_r_prime = y_r + h_r

        inter_w = min(x_l_prime, x_r_prime) - inter_x
        inter_h = min(y_l_prime, y_r_prime) - inter_y

        if inter_w > 0 and inter_h > 0:
            self.crop_rect = [inter_x, inter_y, inter_w, inter_h]
            print(
                f"\nUsing ROI intersection for crop rectangle: x={inter_x}, y={inter_y}, width={inter_w}, height={inter_h}"
            )
        else:
            print(
                "\nWarning: ROI intersection is not valid. Falling back to contour-based cropping."
            )
            # Fallback: Use the existing contour-based method.
            # Need to remap sample images first to use calculate_optimal_crop_rect if they exist.
            # If not, this path might indicate a more severe issue or require a different fallback.
            temp_rectified_l_for_crop = None
            temp_rectified_r_for_crop = None

            # Try to create temporary rectified images for contour-based cropping
            # This requires self.sample_image_l/r to be available.
            # If not, calculate_optimal_crop_rect cannot be called with image data.
            # A more robust fallback might use self.gray_l/r dimensions if sample images are missing.
            if self.sample_image_l is not None and self.sample_image_r is not None:
                temp_rectified_l_for_crop = cv2.remap(
                    self.sample_image_l,
                    self.stereo_map_l[0],
                    self.stereo_map_l[1],
                    cv2.INTER_LINEAR,
                )
                temp_rectified_r_for_crop = cv2.remap(
                    self.sample_image_r,
                    self.stereo_map_r[0],
                    self.stereo_map_r[1],
                    cv2.INTER_LINEAR,
                )
            elif hasattr(self, "gray_l") and hasattr(
                self, "gray_r"
            ):  # Fallback to gray images if samples not set
                temp_rectified_l_for_crop = cv2.remap(
                    cv2.cvtColor(
                        self.gray_l, cv2.COLOR_GRAY2BGR
                    ),  # Remap needs 3 channels if original was color
                    self.stereo_map_l[0],
                    self.stereo_map_l[1],
                    cv2.INTER_LINEAR,
                )
                temp_rectified_r_for_crop = cv2.remap(
                    cv2.cvtColor(self.gray_r, cv2.COLOR_GRAY2BGR),
                    self.stereo_map_r[0],
                    self.stereo_map_r[1],
                    cv2.INTER_LINEAR,
                )

            if (
                temp_rectified_l_for_crop is not None
                and temp_rectified_r_for_crop is not None
            ):
                self.crop_rect = self.calculate_optimal_crop_rect(
                    temp_rectified_l_for_crop, temp_rectified_r_for_crop
                )
                print(
                    f"Using contour-based crop rectangle: x={self.crop_rect[0]}, y={self.crop_rect[1]}, width={self.crop_rect[2]}, height={self.crop_rect[3]}"
                )
            else:
                # Ultimate fallback: use full image dimensions of one of the gray images
                print(
                    "Warning: Could not determine crop rectangle from ROIs or image data for contour method. Using full image dimensions of gray_l."
                )
                h_img, w_img = self.gray_l.shape[:2]  # Assumes self.gray_l is populated
                self.crop_rect = [0, 0, w_img, h_img]

        # Create rectified samples for display and apply the determined self.crop_rect
        if self.sample_image_l is not None and self.sample_image_r is not None:
            self.sample_rectified_l = cv2.remap(
                self.sample_image_l,
                self.stereo_map_l[0],
                self.stereo_map_l[1],
                cv2.INTER_LINEAR,
            )
            self.sample_rectified_r = cv2.remap(
                self.sample_image_r,
                self.stereo_map_r[0],
                self.stereo_map_r[1],
                cv2.INTER_LINEAR,
            )

            # Apply the determined crop to sample images for display
            if self.crop_rect and self.crop_rect[2] > 0 and self.crop_rect[3] > 0:
                x, y, w, h = self.crop_rect

                # Ensure crop dimensions are within bounds of the full sample_rectified images
                h_s_l, w_s_l = self.sample_rectified_l.shape[:2]
                h_s_r, w_s_r = self.sample_rectified_r.shape[:2]

                # Adjust crop if it goes out of bounds
                # (x,y from ROI are relative to full rectified image, should be fine)
                final_x = max(0, x)
                final_y = max(0, y)
                final_w = min(
                    w, w_s_l - final_x
                )  # Ensure width does not exceed image boundary from x
                final_h = min(
                    h, h_s_l - final_y
                )  # Ensure height does not exceed image boundary from y

                # Also ensure width does not exceed other image boundary
                final_w = min(final_w, w_s_r - final_x)
                final_h = min(final_h, h_s_r - final_y)

                if final_w > 0 and final_h > 0:
                    # Update self.crop_rect if it was adjusted for safety, though ideally it shouldn't need much adjustment
                    # if ROIs were correct. This primarily ensures the sample cropping is safe.
                    # The self.crop_rect to be saved should be the one from ROI intersection or primary fallback.
                    # For display purposes, we use the safe final_x,y,w,h.
                    self.sample_rectified_l = self.sample_rectified_l[
                        final_y : final_y + final_h, final_x : final_x + final_w
                    ]
                    self.sample_rectified_r = self.sample_rectified_r[
                        final_y : final_y + final_h, final_x : final_x + final_w
                    ]
                    print(
                        f"Applied crop to sample images for display: x={final_x}, y={final_y}, width={final_w}, height={final_h}"
                    )
                else:
                    print(
                        "Warning: Determined crop rectangle is invalid for sample image dimensions after adjustment. Samples not cropped for display."
                    )
            else:
                print(
                    "Warning: self.crop_rect is invalid. Sample images will not be cropped for display."
                )

        # Print final crop information that will be saved
        if self.crop_rect and self.crop_rect[2] > 0 and self.crop_rect[3] > 0:
            print(
                f"\nFinal crop rectangle to be saved: x={self.crop_rect[0]}, y={self.crop_rect[1]}, width={self.crop_rect[2]}, height={self.crop_rect[3]}"
            )
        else:
            # This case means self.crop_rect is problematic, assign a default full crop to avoid saving invalid rect
            print(
                "\nWarning: Final crop_rect is invalid. Defaulting to full image for saving."
            )
            h_img, w_img = self.gray_l.shape[:2]
            self.crop_rect = [0, 0, w_img, h_img]

    def calculate_optimal_crop_rect(self, left_image, right_image):
        """
        Calculate the optimal crop rectangle for both images.

        Parameters:
            left_image: The rectified left image
            right_image: The rectified right image

        Returns:
            list: [x, y, width, height] - crop rectangle
        """
        # Convert to grayscale if not already
        if len(left_image.shape) > 2:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image.copy()
            right_gray = right_image.copy()

        # Find non-black regions in both images
        _, left_mask = cv2.threshold(left_gray, 1, 255, cv2.THRESH_BINARY)
        _, right_mask = cv2.threshold(right_gray, 1, 255, cv2.THRESH_BINARY)

        # Combine masks to get common valid region
        common_mask = cv2.bitwise_and(left_mask, right_mask)

        # Find bounding box of non-black region
        contours, _ = cv2.findContours(
            common_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            # Find the largest contour (should be the valid image area)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Removed margin logic: The x, y, w, h from boundingRect are the tightest crop.
            # Ensure width and height are positive
            if w <= 0 or h <= 0:
                print(
                    "Warning: Auto-detected crop rectangle has zero width or height during calibration. Falling back to full image."
                )
                img_h, img_w = left_image.shape[:2]
                return [0, 0, img_w, img_h]

            return [x, y, w, h]
        else:
            # If no valid common region found, return full image dimensions
            print("Warning: Could not find valid common region for cropping")
            h, w = left_image.shape[:2]
            return [0, 0, w, h]

    def save_calibration_parameters(self, filename="enhanced_stereo_map.xml"):
        """
        Save calibration parameters to a file, including crop rectangle.

        Parameters:
            filename (str): Output file name
        """
        if self.stereo_map_l is None or self.stereo_map_r is None:
            raise ValueError(
                "Rectification maps not available. Run stereo_rectify() first."
            )

        cv_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

        cv_file.write("stereoMapL_x", self.stereo_map_l[0])
        cv_file.write("stereoMapL_y", self.stereo_map_l[1])
        cv_file.write("stereoMapR_x", self.stereo_map_r[0])
        cv_file.write("stereoMapR_y", self.stereo_map_r[1])
        cv_file.write("q", self.q)

        # Save crop rectangle
        cv_file.write("cropRect", np.array(self.crop_rect))

        cv_file.release()
        print(f"\nCalibration parameters saved to {filename}")

    def calculate_reprojection_errors(self, rvecs_l, tvecs_l, rvecs_r, tvecs_r):
        """
        Calculate reprojection errors for both cameras.

        Parameters:
            rvecs_l (list): Rotation vectors for left camera
            tvecs_l (list): Translation vectors for left camera
            rvecs_r (list): Rotation vectors for right camera
            tvecs_r (list): Translation vectors for right camera
        """
        # Calculate errors for left camera
        self.errors_l = []
        self.mean_error_l = 0

        for i in range(len(self.obj_points)):
            img_points2_l, _ = cv2.projectPoints(
                self.obj_points[i], rvecs_l[i], tvecs_l[i], self.camera_matrix_l, self.dist_l
            )
            error_l = cv2.norm(self.img_points_l[i], img_points2_l, cv2.NORM_L2) / len(
                img_points2_l
            )
            self.errors_l.append(error_l)
            self.mean_error_l += error_l

        self.mean_error_l /= len(self.obj_points)
        self.max_error_l = max(self.errors_l)
        self.max_error_idx_l = self.errors_l.index(self.max_error_l) + 1

        # Calculate errors for right camera
        self.errors_r = []
        self.mean_error_r = 0

        for i in range(len(self.obj_points)):
            img_points2_r, _ = cv2.projectPoints(
                self.obj_points[i], rvecs_r[i], tvecs_r[i], self.camera_matrix_r, self.dist_r
            )
            error_r = cv2.norm(self.img_points_r[i], img_points2_r, cv2.NORM_L2) / len(
                img_points2_r
            )
            self.errors_r.append(error_r)
            self.mean_error_r += error_r

        self.mean_error_r /= len(self.obj_points)
        self.max_error_r = max(self.errors_r)
        self.max_error_idx_r = self.errors_r.index(self.max_error_r) + 1

        # Report error statistics
        print("\nReprojection Error Analysis:")
        print("-" * 50)
        print(f"Left Camera - Mean Error: {self.mean_error_l:.6f} pixels")
        print(
            f"Left Camera - Max Error: {self.max_error_l:.6f} pixels (Image {self.max_error_idx_l})"
        )
        print(f"Right Camera - Mean Error: {self.mean_error_r:.6f} pixels")
        print(
            f"Right Camera - Max Error: {self.max_error_r:.6f} pixels (Image {self.max_error_idx_r})"
        )

    def show_error_analysis(self):
        """
        Display error analysis graphs for both cameras.
        """
        if not self.errors_l or not self.errors_r:
            raise ValueError("No error data. Run calibrate_mono_cameras() first.")

        img_numbers_l = list(range(1, len(self.errors_l) + 1))
        img_numbers_r = list(range(1, len(self.errors_r) + 1))

        plt.figure(figsize=(12, 8))

        # Left camera error subplot
        plt.subplot(2, 1, 1)
        plt.bar(img_numbers_l, self.errors_l, width=0.6, color="blue")
        plt.axhline(
            y=self.mean_error_l,
            color="red",
            linestyle="--",
            label=f"Average: {self.mean_error_l:.6f}",
        )
        plt.xlabel("Image Number")
        plt.ylabel("Reprojection Error (pixels)")
        plt.title("Left Camera Reprojection Error Analysis")
        plt.legend()
        plt.grid(True)

        # Right camera error subplot
        plt.subplot(2, 1, 2)
        plt.bar(img_numbers_r, self.errors_r, width=0.6, color="green")
        plt.axhline(
            y=self.mean_error_r,
            color="red",
            linestyle="--",
            label=f"Average: {self.mean_error_r:.6f}",
        )
        plt.xlabel("Image Number")
        plt.ylabel("Reprojection Error (pixels)")
        plt.title("Right Camera Reprojection Error Analysis")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("enhanced_stereo_camera_error_analysis.png")
        plt.show()

        print(
            "\nError analysis figure saved as enhanced_stereo_camera_error_analysis.png"
        )

    def show_calibration_results(self):
        """
        Display calibration results visually (before and after rectification).
        """
        if self.sample_image_l is None or self.sample_rectified_l is None:
            raise ValueError("Sample images not available. Run full calibration first.")

        # Create side-by-side comparison of original and rectified images
        before_l = self.sample_image_l.copy()
        before_r = self.sample_image_r.copy()
        after_l = self.sample_rectified_l.copy()
        after_r = self.sample_rectified_r.copy()

        # Draw horizontal lines to show rectification
        h, w = before_l.shape[:2]
        lines = np.linspace(0, h - 1, 20).astype(np.int32)

        for line in lines:
            cv2.line(before_l, (0, line), (w - 1, line), (0, 255, 0), 1)
            cv2.line(before_r, (0, line), (w - 1, line), (0, 255, 0), 1)

        h, w = after_l.shape[:2]
        lines = np.linspace(0, h - 1, 20).astype(np.int32)

        for line in lines:
            cv2.line(after_l, (0, line), (w - 1, line), (0, 255, 0), 1)
            cv2.line(after_r, (0, line), (w - 1, line), (0, 255, 0), 1)

        # Combine images for display
        before = np.hstack((before_l, before_r))
        after = np.hstack((after_l, after_r))

        # Resize after images to match before images (they're cropped so they'll be smaller)
        before_h, before_w = before.shape[:2]
        after = cv2.resize(after, (before_w, after.shape[0]))

        comparison = np.vstack((before, after))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            comparison, "Before Rectification", (10, 30), font, 1, (0, 0, 255), 2
        )
        cv2.putText(
            comparison,
            "After Rectification (Cropped)",
            (10, before_h + 30),
            font,
            1,
            (0, 0, 255),
            2,
        )

        # Resize for display
        max_height = 1000
        if comparison.shape[0] > max_height:
            scale = max_height / comparison.shape[0]
            comparison = cv2.resize(
                comparison, (int(comparison.shape[1] * scale), max_height)
            )

        cv2.imshow("Enhanced Stereo Calibration Results", comparison)
        cv2.imwrite("enhanced_stereo_calibration_result.png", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\nCalibration results saved as enhanced_stereo_calibration_result.png")

    def set_flags(self, **kwargs):
        """
        Set flags for stereo calibration.

        Parameters:
            **kwargs: Flag settings (e.g., fix_aspect_ratio=True)
        """
        # Reset flags
        self.flags = 0

        # Set flags based on input
        if kwargs.get("fix_aspect_ratio", True):
            self.flags |= self.CALIB_FIX_ASPECT_RATIO

        if kwargs.get("use_intrinsic_guess", False):
            self.flags |= self.CALIB_USE_INTRINSIC_GUESS

        if kwargs.get("same_focal_length", True):
            self.flags |= self.CALIB_SAME_FOCAL_LENGTH

        if kwargs.get("zero_tangent_dist", True):
            self.flags |= self.CALIB_ZERO_TANGENT_DIST

        if kwargs.get("rational_model", False):
            self.flags |= self.CALIB_RATIONAL_MODEL

        if kwargs.get("fix_k1", False):
            self.flags |= self.CALIB_FIX_K1

        if kwargs.get("fix_k2", False):
            self.flags |= self.CALIB_FIX_K2

        if kwargs.get("fix_k3", False):
            self.flags |= self.CALIB_FIX_K3

        if kwargs.get("fix_k4", False):
            self.flags |= self.CALIB_FIX_K4

        if kwargs.get("fix_k5", False):
            self.flags |= self.CALIB_FIX_K5

        if kwargs.get("fix_k6", False):
            self.flags |= self.CALIB_FIX_K6

        print("\nCalibration flags have been updated.")

    def calibrate_and_save(self):
        """
        Run the full calibration pipeline and save results.
        """
        try:
            self.load_images()
            self.calibrate_mono_cameras()
            self.calibrate_stereo()
            self.stereo_rectify()
            self.save_calibration_parameters()
            self.show_error_analysis()
            self.show_calibration_results()
        except Exception as e:
            print(f"Error during calibration: {str(e)}")

    def set_chessboard_size(self, width, height):
        """
        Set chessboard dimensions.

        Parameters:
            width (int): Number of inner corners along width
            height (int): Number of inner corners along height
        """
        self.chessboard_size = (width, height)
        self.prepare_object_points()
        print(f"Chessboard size set to {width}x{height}")

    def set_frame_size(self, width, height):
        """
        Set camera frame dimensions.

        Parameters:
            width (int): Image width in pixels
            height (int): Image height in pixels
        """
        self.frame_size = (width, height)
        print(f"Frame size set to {width}x{height}")

    def set_square_size(self, size_mm):
        """
        Set chessboard square size.

        Parameters:
            size_mm (float): Square size in millimeters
        """
        self.square_size_mm = size_mm
        self.prepare_object_points()
        print(f"Square size set to {size_mm} mm")


# Example usage
if __name__ == "__main__":
    # Create calibration object
    stereo_calib = EnhancedStereoCalibration(
        left_img_folder="./dataset2/left/", right_img_folder="./dataset2/right/"
    )

    # Set calibration parameters if needed
    # stereo_calib.set_chessboard_size(13, 9)
    # stereo_calib.set_frame_size(1640, 1232)
    # stereo_calib.set_square_size(20)

    # Set calibration flags
    stereo_calib.set_flags(
        fix_aspect_ratio=True,
        use_intrinsic_guess=False,
        same_focal_length=True,
        zero_tangent_dist=True,
        rational_model=False,
    )

    # Run calibration
    stereo_calib.calibrate_and_save()
