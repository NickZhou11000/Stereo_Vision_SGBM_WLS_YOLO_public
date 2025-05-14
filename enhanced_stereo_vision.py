import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d


class EnhancedStereoVision:
    """
    An enhanced class for stereo vision processing that provides functionality for disparity map computation,
    depth map generation, and 3D point cloud reconstruction with improved cropping consistency.
    """

    # Class constants
    MIN_DISPARITY = 16
    MAX_NUM_DISPARITY = 200
    MAX_BLOCK_SIZE = 200
    MAX_UNIQUENESS_RATIO = 200
    MAX_SPECKLE_WINDOW = 200
    MAX_SPECKLE_RANGE = 200
    MAX_DISP12_MAX_DIFF = 200
    MAX_P1 = 100
    MAX_P2 = 1000
    MAX_LAMBDA = 10000
    MAX_SIGMA = 300
    DEFAULT_OUTPUT_FILE = "enhanced_point_cloud.ply"
    # CROP_MARGIN = 5 # Removed

    #################################
    # Public Methods
    #################################

    def __init__(self, calibration_file="enhanced_stereo_map.xml"):
        """
        Initialize the EnhancedStereoVision object with calibration data.

        Parameters:
            calibration_file (str): Path to the calibration file
        """
        self.calibration_file = calibration_file
        self.stereo_map_left_x = None
        self.stereo_map_left_y = None
        self.stereo_map_right_x = None
        self.stereo_map_right_y = None
        self.reprojection_matrix = None
        self.crop_rect = None  # Will store [x, y, width, height]
        self.left_image = None
        self.right_image = None
        self.rectified_left_image = None
        self.rectified_right_image = None
        self.cropped_left_image = None
        self.cropped_right_image = None
        self.left_image_gray = None
        self.right_image_gray = None
        self.disparity_map = None
        self.disparity_params = {
            "numDisparities": 80,  # Needs to be divisible by 16
            "blockSize": 9,
            "uniquenessRatio": 5,
            "speckleWindowSize": 0,
            "speckleRange": 0,
            "disp12MaxDiff": 0,
            "p1": 30,
            "p2": 1000,
            "useWls": True,
            "lambdaValue": 10000,
            "sigmaValue": 1.6,
        }

        self._load_calibration_data()

    def load_images(self, left_image_path, right_image_path):
        """
        Load stereo image pair from files.

        Parameters:
            left_image_path (str): Path to the left image
            right_image_path (str): Path to the right image
        """
        self.left_image = cv2.imread(left_image_path)
        self.right_image = cv2.imread(right_image_path)

        if self.left_image is None or self.right_image is None:
            raise ValueError("Failed to load images from the provided paths")

        # Rectify and crop images
        self._process_images()

    def show_images(self, delay=0):
        """
        Display the original, rectified, and cropped images.

        Parameters:
            delay (int): Time in milliseconds to show each image (0 = wait for key press)
        """
        # Show the original frames
        cv2.imshow("Original Left Image", self.left_image)
        cv2.imshow("Original Right Image", self.right_image)

        # Show the rectified frames (before cropping)
        cv2.imshow("Rectified Left Image", self.rectified_left_image)
        cv2.imshow("Rectified Right Image", self.rectified_right_image)

        # Show the processed frames (after cropping)
        cv2.imshow("Cropped Left Image", self.cropped_left_image)
        cv2.imshow("Cropped Right Image", self.cropped_right_image)

        cv2.waitKey(delay)

    def set_disparity_parameters(self, **params):
        """
        Set parameters for disparity map computation.

        Parameters:
            **params: Parameters to set (e.g., numDisparities=48, blockSize=9)
        """
        for key, value in params.items():
            if key in self.disparity_params:
                self.disparity_params[key] = value

    def get_disparity_map(self, interactive=False):
        """
        Compute the disparity map from the stereo image pair.

        Parameters:
            interactive (bool): If True, show interactive GUI to adjust parameters

        Returns:
            numpy.ndarray: The computed disparity map
        """
        if interactive:
            self._run_interactive_parameter_adjustment()
        else:
            self._compute_disparity_map()

        return self.disparity_map

    def get_depth_map(self):
        """
        Convert disparity map to depth map.

        Returns:
            numpy.ndarray: The computed depth map in millimeters
        """
        if self.disparity_map is None:
            self._compute_disparity_map()

        # Depth calculation
        # For stereo, Q matrix (reprojection matrix) maps disparity to depth
        # We extract these values from reprojection matrix
        Q = self.reprojection_matrix
        focal_length = Q[2, 3]  # Focal length
        baseline = 1.0 / Q[3, 2]  # Baseline

        # Avoid division by zero and filter invalid disparities
        valid_mask = self.disparity_map > 0

        # Initialize depth map with zeros
        depth_map = np.zeros_like(self.disparity_map, dtype=np.float32)

        # Calculate depth using disparity: depth = (focal_length * baseline) / disparity
        # Since disparity from SGBM is multiplied by 16 internally, we need to account for that
        disparity_scaled = np.divide(self.disparity_map, 16.0)
        depth_map[valid_mask] = (focal_length * baseline) / disparity_scaled[valid_mask]

        # Filter out unrealistic values (too far or too close)
        max_depth = 5000  # Max depth of 5m (5000mm)
        depth_map[depth_map > max_depth] = 0  # Filter values over max depth
        depth_map[depth_map < 0] = 0  # Filter negative values

        return depth_map

    def get_point_cloud(self, enhance_colors=True, output_file=None):
        """
        Generate a 3D point cloud from the disparity map.

        Parameters:
            enhance_colors (bool): Whether to enhance colors in the point cloud
            output_file (str): Path to save the point cloud file (None = don't save)

        Returns:
            o3d.geometry.PointCloud: The generated point cloud
        """
        if self.disparity_map is None:
            self._compute_disparity_map()

        # Convert disparity map to float32 and divide by 16 as required
        # This is needed because SGBM returns fixed-point disparity with 4 fractional bits
        disparity_map_float = np.float32(np.divide(self.disparity_map, 16.0))

        # Reproject points into 3D
        points_3d = cv2.reprojectImageTo3D(
            disparity_map_float, self.reprojection_matrix, handleMissingValues=False
        )

        # Get colors from the LEFT image (RGB)
        colors = cv2.cvtColor(self.cropped_left_image, cv2.COLOR_BGR2RGB)

        # Create a mask for valid disparity values
        mask_map = disparity_map_float > disparity_map_float.min()

        # Apply color enhancement if requested
        if enhance_colors:
            alpha = 1.2  # Contrast control
            beta = 30  # Brightness control
            enhanced_colors = cv2.convertScaleAbs(colors, alpha=alpha, beta=beta)
            output_colors = enhanced_colors[mask_map]
        else:
            output_colors = colors[mask_map]

        # Mask points by valid disparities
        output_points = points_3d[mask_map]

        # Save point cloud file if requested
        if output_file is not None:
            self._create_point_cloud_file(output_points, output_colors, output_file)
            print(f"Point cloud saved as {output_file}")

        # Create and return Open3D point cloud
        pcd = self._create_open3d_point_cloud(output_points, output_colors)
        return pcd

    def visualize_point_cloud(self, pcd=None):
        """
        Visualize a point cloud using Open3D.

        Parameters:
            pcd: Point cloud to visualize (if None, generate new one)

        Returns:
            o3d.geometry.PointCloud: The visualized point cloud
        """
        if pcd is None:
            pcd = self.get_point_cloud()

        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window("Enhanced Stereo Vision Point Cloud")

        # Add point cloud to the visualization
        vis.add_geometry(pcd)

        # Add coordinate frame (XYZ axes)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50.0,  # Size of the coordinate frame
            origin=[0, 0, 0],  # Position at origin
        )
        vis.add_geometry(coordinate_frame)

        # Set background color and rendering options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark gray background
        opt.point_size = 1.0

        # Set view control
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])  # Looking along negative z-axis
        view_control.set_up([0, -1, 0])  # Up vector
        view_control.set_lookat([0, 0, 0])  # Look at origin
        view_control.set_zoom(0.8)

        # Run visualization
        vis.run()
        vis.destroy_window()

        return pcd

    def visualize_disparity_map(self):
        """
        Visualize the disparity map with matplotlib.
        """
        if self.disparity_map is None:
            self._compute_disparity_map()

        # Show disparity map with matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(self.disparity_map, "gray")
        plt.title(
            f"numDisp:{self.disparity_params['numDisparities']}, "
            f"blockSize:{self.disparity_params['blockSize']}, "
            f"WLS:{self.disparity_params['useWls']}, "
            f"Lambda:{self.disparity_params['lambdaValue']}, "
            f"Sigma:{self.disparity_params['sigmaValue']}"
        )
        plt.colorbar()
        plt.show()

    def visualize_depth_map(self):
        """
        Visualize the depth map with matplotlib.
        """
        # Get the depth map
        depth_map = self.get_depth_map()

        # Get the maximum depth from get_depth_map method (5000mm)
        max_depth = 5000

        # Filter out outliers for better visualization
        # Create a mask for reasonable depth values
        valid_mask = (depth_map > 0) & (depth_map <= max_depth)

        # Create a copy for visualization and apply mask
        display_depth = np.copy(depth_map)
        display_depth[~valid_mask] = 0

        # Calculate min and max for better color mapping (excluding zeros)
        non_zero_depth = display_depth[display_depth > 0]
        min_val = np.min(non_zero_depth) if non_zero_depth.size > 0 else 0
        max_val = np.max(non_zero_depth) if non_zero_depth.size > 0 else max_depth

        # Show depth map with matplotlib
        plt.figure(figsize=(10, 6))
        plt.imshow(
            display_depth, "jet", vmin=min_val, vmax=max_val
        )  # Using jet colormap for depth visualization
        plt.title("Enhanced Depth Map (0-5m)")
        plt.colorbar(label="Distance (mm)")
        plt.show()

    def get_processed_left_image(self):
        """
        Returns the processed (rectified and cropped) left image.
        This image is suitable for object detection and corresponds to the
        disparity/depth map and point cloud.

        Returns:
            numpy.ndarray: The processed left image, or None if images not loaded.
        """
        if self.cropped_left_image is None:
            print("Warning: Cropped left image is not available. Load images first.")
        return self.cropped_left_image

    #################################
    # Private Methods
    #################################

    def _load_calibration_data(self):
        """
        Load stereo calibration data from the calibration file.
        """
        calibration_file = cv2.FileStorage()
        calibration_file.open(self.calibration_file, cv2.FileStorage_READ)

        if not calibration_file.isOpened():
            raise ValueError(f"Could not open calibration file: {self.calibration_file}")

        # Get rectification maps for left and right cameras
        self.stereo_map_left_x = calibration_file.getNode("stereoMapL_x").mat()
        self.stereo_map_left_y = calibration_file.getNode("stereoMapL_y").mat()
        self.stereo_map_right_x = calibration_file.getNode("stereoMapR_x").mat()
        self.stereo_map_right_y = calibration_file.getNode("stereoMapR_y").mat()

        # Get reprojection matrix for generating 3D points
        self.reprojection_matrix = calibration_file.getNode("q").mat()

        # Load crop rectangle if available
        crop_rect_node = calibration_file.getNode("cropRect")
        if not crop_rect_node.empty():
            self.crop_rect = crop_rect_node.mat().flatten().astype(int).tolist()
            print(
                f"Loaded crop rectangle: x={self.crop_rect[0]}, y={self.crop_rect[1]}, "
                f"width={self.crop_rect[2]}, height={self.crop_rect[3]}"
            )
        else:
            print(
                "Warning: No crop rectangle found in calibration file, using automatic detection"
            )
            self.crop_rect = None

        calibration_file.release()

    def _process_images(self):
        """
        Apply rectification and cropping to the loaded stereo images.
        """
        # Undistort and rectify images
        self.rectified_left_image = cv2.remap(
            self.left_image,
            self.stereo_map_left_x,
            self.stereo_map_left_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )
        self.rectified_right_image = cv2.remap(
            self.right_image,
            self.stereo_map_right_x,
            self.stereo_map_right_y,
            cv2.INTER_LANCZOS4,
            cv2.BORDER_CONSTANT,
            0,
        )

        # Apply consistent cropping
        self._crop_images()

        # Convert to grayscale for disparity computation
        self.left_image_gray = cv2.cvtColor(self.cropped_left_image, cv2.COLOR_BGR2GRAY)
        self.right_image_gray = cv2.cvtColor(self.cropped_right_image, cv2.COLOR_BGR2GRAY)

    def _crop_images(self):
        """
        Apply consistent cropping to rectified stereo images.
        """
        # If we don't have a stored crop rectangle, calculate it
        if self.crop_rect is None:
            self.crop_rect = self._calculate_optimal_crop_rect(
                self.rectified_left_image, self.rectified_right_image
            )

        # Apply crop to rectified images
        x, y, w, h = self.crop_rect
        self.cropped_left_image = self.rectified_left_image[y : y + h, x : x + w]
        self.cropped_right_image = self.rectified_right_image[y : y + h, x : x + w]

    def _calculate_optimal_crop_rect(self, left_image, right_image):
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
            # Ensure width and height are positive to avoid issues with empty crops
            if w <= 0 or h <= 0:
                print(
                    "Warning: Auto-detected crop rectangle has zero width or height. Falling back to full image."
                )
                img_h, img_w = left_image.shape[:2]
                return [0, 0, img_w, img_h]

            print(f"Auto-detected crop rectangle: x={x}, y={y}, width={w}, height={h}")
            return [x, y, w, h]
        else:
            # If no valid common region found, return full image dimensions
            print("Warning: Could not find valid common region for cropping")
            h, w = left_image.shape[:2]
            return [0, 0, w, h]

    def _compute_disparity_map(self):
        """
        Compute the disparity map using current parameter settings.
        """
        # Get parameters
        num_disp = self.disparity_params["numDisparities"]
        block_size = self.disparity_params["blockSize"]
        uniqueness_ratio = self.disparity_params["uniquenessRatio"]
        speckle_window = self.disparity_params["speckleWindowSize"]
        speckle_range = self.disparity_params["speckleRange"]
        disp12_max_diff = self.disparity_params["disp12MaxDiff"]
        p1 = self.disparity_params["p1"]
        p2 = self.disparity_params["p2"]
        use_wls = self.disparity_params["useWls"]
        lambda_value = self.disparity_params["lambdaValue"]
        sigma_value = self.disparity_params["sigmaValue"]

        # Create left matcher (SGBM)
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.MIN_DISPARITY,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window,
            speckleRange=speckle_range,
            disp12MaxDiff=disp12_max_diff,
            P1=p1,
            P2=p2,
        )

        # Compute left disparity map
        disparity_map_left = left_matcher.compute(self.left_image_gray, self.right_image_gray)

        # Apply WLS filter if enabled
        if use_wls:
            # Create right matcher for WLS filter
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            disparity_map_right = right_matcher.compute(
                self.right_image_gray, self.left_image_gray
            )

            # Initialize the WLS filter
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(lambda_value)
            wls_filter.setSigmaColor(sigma_value)

            # Filter the disparity map
            self.disparity_map = wls_filter.filter(
                disparity_map_left, self.left_image_gray, None, disparity_map_right
            )
        else:
            self.disparity_map = disparity_map_left

    def _run_interactive_parameter_adjustment(self):
        """
        Run an interactive GUI to adjust disparity map parameters.
        """
        # Initial values
        num_disp = self.disparity_params["numDisparities"]
        block_size = self.disparity_params["blockSize"]
        uniqueness_ratio = self.disparity_params["uniquenessRatio"]
        speckle_window = self.disparity_params["speckleWindowSize"]
        speckle_range = self.disparity_params["speckleRange"]
        disp12_max_diff = self.disparity_params["disp12MaxDiff"]
        p1 = self.disparity_params["p1"]
        p2 = self.disparity_params["p2"]
        lambda_value = self.disparity_params["lambdaValue"]
        sigma_value = self.disparity_params["sigmaValue"]
        use_wls = self.disparity_params["useWls"]

        # Create windows for trackbars
        cv2.namedWindow("Disparity Parameters")
        cv2.namedWindow("WLS Filter Parameters")

        # Create trackbars
        cv2.createTrackbar(
            "Num Disparities",
            "Disparity Parameters",
            num_disp // 16,
            self.MAX_NUM_DISPARITY // 16,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Block Size",
            "Disparity Parameters",
            block_size,
            self.MAX_BLOCK_SIZE,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Uniqueness Ratio",
            "Disparity Parameters",
            uniqueness_ratio,
            self.MAX_UNIQUENESS_RATIO,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Speckle Window",
            "Disparity Parameters",
            speckle_window,
            self.MAX_SPECKLE_WINDOW,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Speckle Range",
            "Disparity Parameters",
            speckle_range,
            self.MAX_SPECKLE_RANGE,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Disp12MaxDiff",
            "Disparity Parameters",
            disp12_max_diff,
            self.MAX_DISP12_MAX_DIFF,
            lambda x: None,
        )
        cv2.createTrackbar(
            "P1", "Disparity Parameters", p1, self.MAX_P1, lambda x: None
        )
        cv2.createTrackbar(
            "P2", "Disparity Parameters", p2, self.MAX_P2, lambda x: None
        )

        # WLS Filter trackbars
        cv2.createTrackbar(
            "Lambda",
            "WLS Filter Parameters",
            lambda_value,
            self.MAX_LAMBDA,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Sigma",
            "WLS Filter Parameters",
            int(sigma_value * 10),
            self.MAX_SIGMA,
            lambda x: None,
        )
        cv2.createTrackbar(
            "Use WLS", "WLS Filter Parameters", 1 if use_wls else 0, 1, lambda x: None
        )

        # Main loop for parameter adjustment
        while True:
            # Get current parameter values from trackbars
            num_disp = cv2.getTrackbarPos("Num Disparities", "Disparity Parameters") * 16
            if num_disp < 16:
                num_disp = 16

            block_size = cv2.getTrackbarPos("Block Size", "Disparity Parameters")
            if block_size % 2 == 0:
                block_size += 1  # Block size must be odd
            if block_size < 1:
                block_size = 1

            uniqueness_ratio = cv2.getTrackbarPos(
                "Uniqueness Ratio", "Disparity Parameters"
            )
            speckle_window = cv2.getTrackbarPos("Speckle Window", "Disparity Parameters")
            speckle_range = cv2.getTrackbarPos("Speckle Range", "Disparity Parameters")
            disp12_max_diff = cv2.getTrackbarPos("Disp12MaxDiff", "Disparity Parameters")
            p1 = cv2.getTrackbarPos("P1", "Disparity Parameters")
            p2 = cv2.getTrackbarPos("P2", "Disparity Parameters")

            # Ensure P2 is always greater than P1
            if p2 <= p1:
                p2 = p1 + 1
                cv2.setTrackbarPos("P2", "Disparity Parameters", p2)

            # Get WLS filter parameters
            lambda_value = cv2.getTrackbarPos("Lambda", "WLS Filter Parameters")
            sigma_value = cv2.getTrackbarPos("Sigma", "WLS Filter Parameters") / 10.0
            use_wls = bool(cv2.getTrackbarPos("Use WLS", "WLS Filter Parameters"))

            # Update parameters
            self.disparity_params.update(
                {
                    "numDisparities": num_disp,
                    "blockSize": block_size,
                    "uniquenessRatio": uniqueness_ratio,
                    "speckleWindowSize": speckle_window,
                    "speckleRange": speckle_range,
                    "disp12MaxDiff": disp12_max_diff,
                    "p1": p1,
                    "p2": p2,
                    "useWls": use_wls,
                    "lambdaValue": lambda_value,
                    "sigmaValue": sigma_value,
                }
            )

            # Compute disparity map with current parameters
            self._compute_disparity_map()

            # Normalize the disparity map for visualization
            disparity_norm = cv2.normalize(
                self.disparity_map,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

            # Create resizable window for better visualization
            cv2.namedWindow("Enhanced Disparity Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced Disparity Map", 800, 600)
            cv2.imshow("Enhanced Disparity Map", disparity_norm)

            # Wait for key press - exit loop if ESC is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == 13:  # Enter key to confirm parameters
                print(
                    f"Selected parameters - numDisparities: {num_disp}, blockSize: {block_size}, uniquenessRatio: {uniqueness_ratio}, "
                    f"speckleWindowSize: {speckle_window}, speckleRange: {speckle_range}, disp12MaxDiff: {disp12_max_diff}, "
                    f"P1: {p1}, P2: {p2}, WLS Lambda: {lambda_value}, WLS Sigma: {sigma_value}, Use WLS: {use_wls}"
                )
                break

        cv2.destroyAllWindows()

    def _create_point_cloud_file(self, vertices, colors, filename):
        """
        Create a PLY file from point cloud data.

        Parameters:
            vertices: 3D points
            colors: RGB colors for each point
            filename: Output filename
        """
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices.reshape(-1, 3), colors])

        ply_header = """ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        """
        with open(filename, "w") as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, "%f %f %f %d %d %d")

    def _create_open3d_point_cloud(self, points, colors):
        """
        Create an Open3D point cloud object.

        Parameters:
            points: 3D points
            colors: RGB colors for each point

        Returns:
            o3d.geometry.PointCloud: The created point cloud
        """
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()

        # Set points
        pcd.points = o3d.utility.Vector3dVector(points)

        # Set colors (normalize RGB values to [0, 1] range as required by Open3D)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

        return pcd


def down_sample_image(image, reduce_factor):
    """
    Downsamples image x number (reduce_factor) of times.

    Parameters:
        image: Input image
        reduce_factor: Number of times to downsample

    Returns:
        numpy.ndarray: Downsampled image
    """
    for i in range(0, reduce_factor):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape

        image = cv2.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


def main():
    """
    Main function demonstrating the use of the EnhancedStereoVision class.
    """
    # Create EnhancedStereoVision object
    stereo_vision = EnhancedStereoVision("enhanced_stereo_map.xml")

    # Load images
    stereo_vision.load_images("./dataset/left/L22.png", "./dataset/right/R22.png")

    # Show loaded, rectified, and cropped images
    stereo_vision.show_images(10)

    # Get and visualize disparity map (interactive)
    disparity_map = stereo_vision.get_disparity_map(interactive=False)
    stereo_vision.visualize_disparity_map()

    # Visualize depth map
    depth_map = stereo_vision.get_depth_map()
    stereo_vision.visualize_depth_map()

    # Generate and visualize point cloud
    point_cloud = stereo_vision.get_point_cloud(
        enhance_colors=True, output_file="enhanced_point_cloud.ply"
    )

    # Visualize the point cloud
    stereo_vision.visualize_point_cloud(point_cloud)


if __name__ == "__main__":
    main()
