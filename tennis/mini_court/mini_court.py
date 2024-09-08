import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import (
    convert_meters_to_pixel_distance,
    get_center_of_bbox,
)

class MiniCourt():
    def __init__(self, frame, court_keypoints):
        # Set the actual court boundaries from keypoints (top-left and bottom-right corners)
        self.court_start_x = min(court_keypoints[0], court_keypoints[2], court_keypoints[10], court_keypoints[12])
        self.court_start_y = min(court_keypoints[1], court_keypoints[3], court_keypoints[11], court_keypoints[13])
        self.court_end_x = max(court_keypoints[0], court_keypoints[2], court_keypoints[10], court_keypoints[12])
        self.court_end_y = max(court_keypoints[1], court_keypoints[3], court_keypoints[11], court_keypoints[13])

        # Calculate court width and height based on keypoints
        self.court_drawing_width = self.court_end_x - self.court_start_x
        self.court_drawing_height = self.court_end_y - self.court_start_y

        # Define the mini court dimensions relative to the real court
        self.drawing_rectangle_width = 175  # Width of the mini court box
        self.drawing_rectangle_height = int(self.drawing_rectangle_width * self.court_drawing_height / self.court_drawing_width)

        # Initialize buffer and padding
        self.buffer = 50
        self.padding_court = 20

        # Set canvas dimensions
        self.video_height, self.video_width = frame.shape[:2]
        self.set_canvas_background_box_position(frame)

        # Court lines based on keypoints
        self.set_court_lines(court_keypoints)

        self.player_positions = []
        self.ball_position = None

    def set_court_lines(self, court_keypoints):
        # Set court lines dynamically based on keypoints
        self.lines = [
            (court_keypoints[0], court_keypoints[1], court_keypoints[2], court_keypoints[3]),  # Baseline top
            (court_keypoints[10], court_keypoints[11], court_keypoints[12], court_keypoints[13])  # Baseline bottom
        ]

    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        # Adjust the buffer and x, y positions for the mini court
        self.buffer = 50
        shift_left = 70  # Move 100 pixels to the left

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width - shift_left  # Shift the mini court left
        self.start_y = self.end_y - self.drawing_rectangle_height


    def video_to_court_coordinates(self, point, court_keypoints):
        # Convert video point to mini court coordinates
        x, y = point
        court_width = self.court_end_x - self.court_start_x
        court_height = self.court_end_y - self.court_start_y

        # Calculate relative position in the court (based on keypoints)
        rel_x = (x - self.court_start_x) / court_width
        rel_y = (y - self.court_start_y) / court_height

        # Scale to mini court dimensions
        mini_court_x = self.start_x + rel_x * self.drawing_rectangle_width
        mini_court_y = self.start_y + rel_y * self.drawing_rectangle_height

        return int(mini_court_x), int(mini_court_y)

    def update_player_positions(self, detections, court_keypoints):
        # Update player positions in mini court
        self.player_positions = []
        for player in detections.get('players', []):
            bbox = player['bbox']
            center = get_center_of_bbox(bbox)
            court_position = self.video_to_court_coordinates(center, court_keypoints)
            self.player_positions.append(court_position)

    def update_ball_position(self, interpolated_position, court_keypoints):
        if interpolated_position and all(interpolated_position):
            center = get_center_of_bbox(interpolated_position)
            self.ball_position = self.video_to_court_coordinates(center, court_keypoints)
        else:
            self.ball_position = None

    def draw_players_and_ball(self, frame):
        # Draw players on mini court
        for position in self.player_positions:
            cv2.circle(frame, position, 5, (0, 255, 0), -1)  # Green dot for players

        # Draw ball on mini court
        if self.ball_position:
            cv2.circle(frame, self.ball_position, 5, (0, 255, 255), -1)  # Yellow dot for ball

        return frame

    
    def draw_background_rectangle(self, frame, court_keypoints):
        shapes = np.zeros_like(frame, np.uint8)

        # Define scaling factors to enlarge the rectangle around the mini court
        margin_factor_x = 0.2  # Increase by 20% horizontally
        margin_factor_y = 0.3  # Increase by 20% vertically
        shift_left = 0  # Shift mini court left by 30 pixels

        # Calculate new top-left and bottom-right points for the larger rectangle
        court_width = self.end_x - self.start_x
        court_height = self.end_y - self.start_y

        top_left_x = int(self.start_x - court_width * margin_factor_x - shift_left)
        top_left_y = int(self.start_y - court_height * margin_factor_y)
        bottom_right_x = int(self.end_x + court_width * margin_factor_x - shift_left)
        bottom_right_y = int(self.end_y + court_height * margin_factor_y)

        # Ensure the coordinates are valid
        if bottom_right_x > top_left_x and bottom_right_y > top_left_y:
            cv2.rectangle(shapes, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 255, 255), cv2.FILLED)

            # Alpha blending for transparency
            alpha = 0.5
            mask = shapes.astype(bool)
            frame[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return frame


    
    
    def draw_court(self, frame, court_keypoints):
        # Draw the background rectangle based on court keypoints
        frame = self.draw_background_rectangle(frame, court_keypoints)

        mini_court_keypoints = []

        # Scale all the court keypoints based on the mini court dimensions
        for i in range(0, len(court_keypoints), 2):
            x = self.start_x + ((court_keypoints[i] - self.court_start_x) / self.court_drawing_width) * self.drawing_rectangle_width
            y = self.start_y + ((court_keypoints[i + 1] - self.court_start_y) / self.court_drawing_height) * self.drawing_rectangle_height
            mini_court_keypoints.append((int(x), int(y)))

        # Draw the keypoints as circles
        for x, y in mini_court_keypoints:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Define lines based on the scaled mini court keypoints
        lines_to_draw = [
            (0, 1),  # Top baseline
            (4, 5),  # Service line
            (2, 3),  # Bottom baseline
            (8, 9),  # Left doubles alley
            (10, 11),  # Right doubles alley
            (6, 7),  # Center service line
            (0, 2),  # Left baseline connecting point 0 and 2
            (12, 13),  # Bottom court line
            (1, 3)   # Right baseline connecting point 1 and 3
        ]

        # Draw lines based on the keypoints
        for start, end in lines_to_draw:
            start_point = mini_court_keypoints[start]
            end_point = mini_court_keypoints[end]
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Calculate the midpoint between keypoints 8 and 10
        midpoint_0_2 = (
            (mini_court_keypoints[8][0] + mini_court_keypoints[10][0]) // 2,
            (mini_court_keypoints[8][1] + mini_court_keypoints[10][1]) // 2
        )

        # Midpoint between keypoints 9 and 11
        midpoint_1_3 = (
            (mini_court_keypoints[9][0] + mini_court_keypoints[11][0]) // 2,
            (mini_court_keypoints[9][1] + mini_court_keypoints[11][1]) // 2
        )

        # Draw the net line between the two midpoints
        cv2.line(frame, midpoint_0_2, midpoint_1_3, (255, 0, 0), 2)  # Red line for the net

        return frame
    
    def draw_mini_court(self, frames, detections, interpolated_positions, court_keypoints_list):
        output_frames = []
        for i, frame in enumerate(frames):
            frame_copy = frame.copy()
            frame_copy = self.draw_background_rectangle(frame_copy, court_keypoints_list[i])
            frame_copy = self.draw_court(frame_copy, court_keypoints_list[i])

            # Update player and ball positions
            self.update_player_positions(detections[i], court_keypoints_list[i])
            self.update_ball_position(interpolated_positions[i], court_keypoints_list[i])

            # Draw players and ball on mini court
            frame_copy = self.draw_players_and_ball(frame_copy)

            output_frames.append(frame_copy)
        return output_frames

    
    def get_width_of_mini_court(self):
        return self.drawing_rectangle_width

