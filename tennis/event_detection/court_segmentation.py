import cv2
import numpy as np

class CourtSegmentation:
    def __init__(self, court_keypoints):
        """
        Initialize the court segmentation class with court key points.
        The key points will be used to define the court's boundaries.
        
        Args:
        - court_keypoints: List of court key points detected from the court detection model.
        """
        self.court_keypoints = court_keypoints
        self.homography_matrix = None
        self.real_world_court_corners = np.array([
            [0, 0],         # Top-left corner of the court
            [0, 10.97],     # Bottom-left corner of the court (standard tennis court dimensions)
            [23.77, 0],     # Top-right corner of the court
            [23.77, 10.97], # Bottom-right corner of the court
        ], dtype=np.float32)  # Real-world court dimensions in meters

        # Define the pixel coordinates of the court from the detected key points
        # Assuming court_keypoints are [x1, y1, x2, y2, ..., x14, y14]
        self.pixel_court_corners = np.array([
            [court_keypoints[0], court_keypoints[1]],   # Top-left corner in pixels
            [court_keypoints[4], court_keypoints[5]],   # Bottom-left corner in pixels
            [court_keypoints[2], court_keypoints[3]],   # Top-right corner in pixels
            [court_keypoints[6], court_keypoints[7]]    # Bottom-right corner in pixels
        ], dtype=np.float32)

        # Compute homography matrix between pixel coordinates and real-world coordinates
        self.compute_homography()

    def compute_homography(self):
        """
        Compute the homography matrix to map pixel coordinates to real-world court coordinates.
        """
        self.homography_matrix, _ = cv2.findHomography(self.pixel_court_corners, self.real_world_court_corners)

    def map_ball_to_court(self, ball_position):
        """
        Map the ball's pixel position to the court's real-world coordinates.
        
        Args:
        - ball_position: The ball's position in the frame in pixel coordinates [x, y].

        Returns:
        - The ball's real-world position in court coordinates.
        """
        if len(ball_position) == 2:  # Expecting ball_position as [x, y]
            ball_pixel = np.array([[ball_position[0], ball_position[1]]], dtype=np.float32)
            ball_pixel = np.array([ball_pixel])

            # Apply the homography transformation to get real-world court coordinates
            ball_real_world = cv2.perspectiveTransform(ball_pixel, self.homography_matrix)

            # Return the x, y position in real-world coordinates
            return ball_real_world[0][0]
        return None

    def is_ball_out_of_bounds(self, ball_real_world_position):
        """
        Check if the ball's real-world position is out of bounds.
        
        Args:
        - ball_real_world_position: The ball's position in real-world court coordinates [x, y].

        Returns:
        - True if the ball is out of bounds, otherwise False.
        """
        x, y = ball_real_world_position
        court_width = 23.77  # Standard tennis court width in meters
        court_height = 10.97  # Standard tennis court height in meters

        if x < 0 or x > court_width or y < 0 or y > court_height:
            return True
        return False

    def detect_out_of_bounds(self, ball_position):
        """
        Main function to check if the ball is out of bounds based on its pixel position.
        
        Args:
        - ball_position: The ball's position in the frame in pixel coordinates [x, y].

        Returns:
        - True if the ball is out of bounds, otherwise False.
        """
        # Map the ball's pixel position to real-world court coordinates
        ball_real_world_position = self.map_ball_to_court(ball_position)
        
        if ball_real_world_position is not None:
            # Check if the ball is out of bounds
            return self.is_ball_out_of_bounds(ball_real_world_position)
        
        return False