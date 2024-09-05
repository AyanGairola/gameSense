import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 175
        self.drawing_rectangle_height = 350
        self.buffer = 50
        self.padding_court=20
        self.horizontal_scale_factor = 1
        self.vertical_scale_factor = 2.2 # Increased vertical scaling #Hit and Try
        self.vertical_offset = -15  # Hit and Try

        self.video_height, self.video_width = frame.shape[:2]

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

        self.player_positions = []
        self.ball_position = None


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
        self.court_start_y = self.start_y + self.padding_court + 20
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court + 20
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

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    

    def update_player_positions(self, detections):
        self.player_positions = []
        for player in detections.get('players', []):
            bbox = player['bbox']
            center = get_center_of_bbox(bbox)
            court_position = self.video_to_court_coordinates(center)
            self.player_positions.append(court_position)

    def update_ball_position(self, interpolated_position):
        if interpolated_position and all(interpolated_position):
            center = get_center_of_bbox(interpolated_position)
            self.ball_position = self.video_to_court_coordinates(center)
        else:
            self.ball_position = None

    def video_to_court_coordinates(self, point):
        x, y = point
        
        # Calculate the relative position within the video frame
        rel_x = x / self.video_width
        rel_y = y / self.video_height
        
        # Apply non-linear transformation to y-coordinate
        transformed_y = rel_y ** 2.5  # Adjust exponent as needed # Hit and Try
        
        # Calculate court coordinates with separate scaling factors
        court_x = self.court_start_x + rel_x * self.court_drawing_width * self.horizontal_scale_factor
        court_y = self.court_start_y + transformed_y * (self.court_end_y - self.court_start_y) * self.vertical_scale_factor
        
        # Apply vertical offset
        court_y += self.vertical_offset
        
        # # Ensure the coordinates don't exceed the court boundaries
        # court_x = min(max(court_x, self.court_start_x), self.court_end_x)
        # court_y = min(max(court_y, self.court_start_y), self.court_end_y)

        # court_x = min(max(court_x, self.court_start_x), self.court_end_x)
        # court_y = min(max(court_y, self.court_start_y), self.court_end_y)
        
        return int(court_x), int(court_y)

    def draw_players_and_ball(self, frame):
        for position in self.player_positions:
            cv2.circle(frame, position, 5, (0, 255, 0), -1)  # Green dot for players

        if self.ball_position:
            cv2.circle(frame, self.ball_position, 3, (0, 0, 255), -1)  # Red dot for ball

        return frame

    def draw_mini_court(self, frames, detections, interpolated_positions):
        output_frames = []
        for i, frame in enumerate(frames):
            frame_copy = frame.copy()
            frame_copy = self.draw_background_rectangle(frame_copy)
            frame_copy = self.draw_court(frame_copy)

            # Update player and ball positions
            self.update_player_positions(detections[i])
            self.update_ball_position(interpolated_positions[i])

            # Draw players and ball
            frame_copy = self.draw_players_and_ball(frame_copy)

            output_frames.append(frame_copy)
        return output_frames