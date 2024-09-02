import cv2
import numpy as np
from utils import (
    convert_pixel_distance_to_meters, get_foot_position, get_center_of_bbox,
    get_closest_keypoint_index, get_height_of_bbox, measure_xy_distance, convert_meters_to_pixel_distance
)
import constants
from court_line_detector import CourtLineDetector
from player_stats import calculate_player_stats, process_player_stats_data
from trackers import UnifiedTracker


class ImprovedMiniCourt():
    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

        # New: Set up perspective transform
        self.set_perspective_transform()

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    
    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames


    def get_mini_court_coordinates(self, object_position, closest_key_point, closest_key_point_index, bbox_height):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Convert pixel distance to meters using bbox height as a reference
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_court_keypoint = (
            self.drawing_key_points[closest_key_point_index*2],
            self.drawing_key_points[closest_key_point_index*2+1]
        )
        
        mini_court_player_position = (
            closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
            closest_mini_court_keypoint[1] + mini_court_y_distance_pixels
        )

        return mini_court_player_position
    

    def convert_to_mini_court_coordinates(self, point):
        # Convert point to numpy array
        point = np.array([point[0], point[1], 1], dtype=np.float32)

        # Apply perspective transform
        transformed_point = np.dot(self.perspective_matrix, point)
        transformed_point = transformed_point / transformed_point[2]

        return (int(transformed_point[0]), int(transformed_point[1]))
    

    def set_perspective_transform(self):
        # Define source points (actual court corners)
        src_pts = np.float32([
            self.drawing_key_points[0:2],  # top-left
            self.drawing_key_points[2:4],  # top-right
            self.drawing_key_points[4:6],  # bottom-left
            self.drawing_key_points[6:8]   # bottom-right
        ])

        # Define destination points (mini-court corners)
        dst_pts = np.float32([
            [self.court_start_x, self.court_start_y],
            [self.court_end_x, self.court_start_y],
            [self.court_start_x, self.court_end_y],
            [self.court_end_x, self.court_end_y]
        ])

        # Calculate the perspective transform matrix
        self.perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    def convert_detections_to_mini_court_coordinates(self, detections, original_court_key_points):
        output_player_boxes = []
        output_ball_boxes = []

        for detection in detections:
            players = detection.get('players', [])
            balls = detection.get('ball', [])

            output_player_bboxes_dict = {}
            output_ball_bboxes_dict = {}

            for player in players:
                if player['class'] == 1:  # Assuming class 1 is player
                    bbox = player['bbox']
                    foot_position = (bbox[0] + bbox[2]) / 2, bbox[3]  # Use bottom center of bbox
                    mini_court_player_position = self.convert_to_mini_court_coordinates(foot_position)
                    player_id = player['id']
                    output_player_bboxes_dict[player_id] = mini_court_player_position
                    print(f"Player {player_id}: Original position: {foot_position}, Mini court position: {mini_court_player_position}")

            for ball in balls:
                if ball['class'] == 0:  # Assuming class 0 is ball
                    bbox = ball['bbox']
                    ball_position = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2  # Use center of bbox
                    mini_court_ball_position = self.convert_to_mini_court_coordinates(ball_position)
                    ball_id = 'ball'
                    output_ball_bboxes_dict[ball_id] = mini_court_ball_position
                    print(f"Ball: Original position: {ball_position}, Mini court position: {mini_court_ball_position}")

            output_player_boxes.append(output_player_bboxes_dict)
            output_ball_boxes.append(output_ball_bboxes_dict)

        return output_player_boxes, output_ball_boxes
    

    def draw_points_on_mini_court(self, frames, positions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            if frame_num < len(positions):
                for _, position in positions[frame_num].items():
                    if position:  # Check if position is not None or empty
                        x, y = position
                        x = int(x)
                        y = int(y)
                        cv2.circle(frame, (x,y), 5, color, -1)
                        print(f"Drawing point at ({x}, {y}) on frame {frame_num}")
        return frames

    def convert_to_mini_court_coordinates(self, point):
        # Convert point to numpy array
        point = np.array([point[0], point[1], 1], dtype=np.float32)

        # Apply perspective transform
        transformed_point = np.dot(self.perspective_matrix, point)
        transformed_point = transformed_point / transformed_point[2]

        return (int(transformed_point[0]), int(transformed_point[1]))