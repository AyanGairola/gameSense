from mini_court import MiniCourt
import numpy as np
import cv2

class RallyDetector:
    def __init__(self, mini_court):
        self.mini_court = mini_court
        self.last_ball_position = None
        self.rally_count = 0
        self.rally_ongoing = False

    def update_rally_count(self, ball_position):
        """
        Update the rally count based on whether the ball has crossed the net on the mini court.
        """
        # Get the x-coordinate of the net position on the mini court (middle of the court)
        middle_x = self.mini_court.start_x + (self.mini_court.get_width_of_mini_court() // 2)

        # Check if the ball has crossed the net
        if self.last_ball_position is not None:
            last_x = self.last_ball_position[0]
            current_x = ball_position[0]

            # Detect if the ball has crossed the net (change in position across the net line)
            if (last_x < middle_x and current_x >= middle_x) or (last_x > middle_x and current_x <= middle_x):
                if not self.rally_ongoing:  # Increment rally count only once per crossing
                    self.rally_count += 1
                    self.rally_ongoing = True  # Mark the rally as ongoing after the first crossing
            else:
                self.rally_ongoing = False  # Reset to allow detection of a new rally

        self.last_ball_position = ball_position

    def get_rally_count(self):
        """Return the current rally count."""
        return self.rally_count

    def reset_rally(self):
        """Reset the rally count and state."""
        self.rally_count = 0
        self.last_ball_position = None
        self.rally_ongoing = False
