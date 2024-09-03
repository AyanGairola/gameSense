from mini_court import MiniCourt
import numpy as np
import cv2
from utils import (
    get_center_of_bbox,
    measure_distance
)

class RallyDetector:
    def __init__(self, mini_court):
        self.mini_court = mini_court
        self.last_ball_position = None
        self.rally_count = 0
        self.rally_ongoing = False

    def update_rally_count(self, ball_position):
        # Get the middle x-coordinate of the mini court (assuming horizontal crossing)
        middle_x = (self.mini_court.get_start_point_of_mini_court()[0] + 
                    self.mini_court.get_start_point_of_mini_court()[0] + 
                    self.mini_court.get_width_of_mini_court()) // 2

        # Check if the ball has crossed the net
        if self.last_ball_position is not None:
            last_x = self.last_ball_position[0]
            current_x = ball_position[0]

            if (last_x < middle_x <= current_x) or (last_x > middle_x >= current_x):
                if not self.rally_ongoing:
                    self.rally_count += 1
                    self.rally_ongoing = True
        else:
            self.rally_ongoing = False

        self.last_ball_position = ball_position

    def get_rally_count(self):
        return self.rally_count

    def reset_rally(self):
        self.rally_count = 0
        self.last_ball_position = None
        self.rally_ongoing = False