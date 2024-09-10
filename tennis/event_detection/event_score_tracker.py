import numpy as np
import cv2
import math

class EventScoreTracker:
    def __init__(self, mini_court):
        """
        Initialize the score tracker with the mini court.
        mini_court: instance of MiniCourt class, used to detect fouls based on mini court boundaries.
        """
        self.mini_court = mini_court
        self.player_scores = {1: 0, 2: 0}  # Points for each player in the current game
        self.game_scores = {1: 0, 2: 0}  # Games won by each player in the current set
        self.set_scores = {1: 0, 2: 0}  # Sets won by each player
        self.last_shot_player = None
        self.foul_detected = False  # Track whether a foul was detected in this rally

        # Constants for winning
        self.games_to_win_set = 6  # Number of games required to win a set
        self.winning_margin_in_games = 2  # Player must win a set by 2 games

    def update_score(self, winner):
        """Update the score based on the winner of the point."""
        if self.player_scores[winner] == 0:
            self.player_scores[winner] = 15
        elif self.player_scores[winner] == 15:
            self.player_scores[winner] = 30
        elif self.player_scores[winner] == 30:
            self.player_scores[winner] = 40
        elif self.player_scores[winner] == 40:
            # Player wins the game
            self.game_scores[winner] += 1
            self.player_scores = {1: 0, 2: 0}  # Reset after game win
            self._check_set_winner(winner)

    def _check_set_winner(self, game_winner):
        """Check if a player has won the set."""
        other_player = 2 if game_winner == 1 else 1

        # Check if the player has enough games to win the set
        if (self.game_scores[game_winner] >= self.games_to_win_set and 
            (self.game_scores[game_winner] - self.game_scores[other_player]) >= self.winning_margin_in_games):
            self.set_scores[game_winner] += 1  # Player wins the set
            self.game_scores = {1: 0, 2: 0}  # Reset games for the next set

    def detect_foul(self, ball_position_mini_court, player_hit, net_x, player_positions):
        """
        Detect fouls based on the ball's position on the mini court.
        - Check if the ball has gone out of bounds on the first contact in the opponent's half
        - Check if the ball has hit the net (remains on the same side after a player hits it)
        """
        if self.foul_detected:  # Prevent repeated foul detection
            return None, None

        # Check for net hit
        net_hit = self._is_net_hit(ball_position_mini_court, player_hit, net_x)
        if net_hit:
            self.foul_detected = True
            return "net_hit", self._get_other_player(player_hit)

        # Check for out of bounds on the mini court
        out_of_bounds = self._is_ball_out_of_bounds_on_mini_court(ball_position_mini_court, player_hit)
        if out_of_bounds:
            self.foul_detected = True
            return "out_of_bounds", self._get_other_player(player_hit)

        return None, None  # No foul detected

    def _is_ball_out_of_bounds_on_mini_court(self, ball_position_mini_court, player_hit):
        """Check if the ball is out of bounds using specific lines on the mini court after it has bounced."""
        ball_x, ball_y = ball_position_mini_court

        # Define the boundaries using mini court keypoints (4,6), (4,5), (6,7), and (5,7)
        p4 = self.mini_court_keypoints[4]
        p5 = self.mini_court_keypoints[5]
        p6 = self.mini_court_keypoints[6]
        p7 = self.mini_court_keypoints[7]

        # Create a boundary polygon
        boundary_polygon = np.array([p4, p6, p7, p5])

        # Check if the ball's position is outside this polygon
        result = cv2.pointPolygonTest(boundary_polygon, (ball_x, ball_y), False)

        # If result is -1, the ball is outside the boundary and has bounced out of bounds
        return result == -1

    def _is_net_hit(self, ball_position, player_hit, net_x):
        """Check if the ball has hit the net based on its position relative to the net on the mini court."""
        ball_x, ball_y = ball_position
        if player_hit == 1 and ball_x > net_x:
            return False  # Player 1 hit, ball crossed the net
        if player_hit == 2 and ball_x < net_x:
            return False  # Player 2 hit, ball crossed the net

        # If the ball hasn't crossed the net after being hit
        return True

    def _get_other_player(self, player_hit):
        """Get the opponent player."""
        return 2 if player_hit == 1 else 1

    def reset_foul(self):
        """Reset the foul detection state for a new rally."""
        self.foul_detected = False
        
        
    def track_bounces(self, previous_ball_position, current_ball_position, two_frames_ago_position, net_x, court_height_threshold=80, velocity_threshold=0.1, slope_threshold=0.005, min_y_change=0.2, y_fallback_threshold=5):
        """
        Enhanced bounce detection by analyzing ball height, velocity, and slopes.
        """
        if previous_ball_position is None or current_ball_position is None or two_frames_ago_position is None:
            return False, None  # Can't detect bounce without three valid positions

        # Get y-coordinates and x-coordinates of the ball
        prev_y = previous_ball_position[1]
        curr_y = current_ball_position[1]
        two_frames_ago_y = two_frames_ago_position[1]

        prev_x = previous_ball_position[0]
        curr_x = current_ball_position[0]

        # Log the current ball height
        print(f"[DEBUG] Current Ball Height: {curr_y}")

        # **Height Threshold**: Detect bounces when the ball is lower on the court
        if curr_y > court_height_threshold:
            print(f"[DEBUG] Ball is too high. No bounce detected.")
            return False, None

        # Check if Y-coordinate changes are significant (small bounces may have subtle changes)
        if abs(curr_y - prev_y) < min_y_change and abs(prev_y - two_frames_ago_y) < min_y_change:
            print(f"[DEBUG] Y-coordinate change too small. No bounce detected.")
            return False, None

        # **Velocity Calculation**: Calculate velocity between frames
        velocity_before = two_frames_ago_y - prev_y
        velocity_now = prev_y - curr_y

        # **Slope Calculation**: Detect slope based on x and y coordinates (use if there's significant horizontal movement)
        slope_now = 0
        if abs(curr_x - prev_x) > 0.01:  # To avoid division by zero, and small movements
            slope_now = (curr_y - prev_y) / (curr_x - prev_x)

        print(f"[DEBUG] Velocity Before: {velocity_before}, Velocity Now: {velocity_now}, Slope Now: {slope_now}")

        # **Turning Point Detection**: Ball is falling and now starts rising
        if velocity_before < 0 and velocity_now > 0 and abs(velocity_before - velocity_now) > velocity_threshold:
            print(f"[DEBUG] Bounce detected at {current_ball_position}")
            return True, current_ball_position

        # **Slope-Based Fallback**: If slope changes significantly, it may indicate a bounce (useful for shallow or long bounces)
        if abs(slope_now) > slope_threshold and abs(velocity_now) < velocity_threshold:
            print(f"[DEBUG] Slope-based bounce detection at {current_ball_position}")
            return True, current_ball_position

        # **Small Bounces Detection (Y-change-based fallback)**: Fallback if ball falls below the threshold
        if curr_y < y_fallback_threshold and abs(velocity_now) > 0:
            print(f"[DEBUG] Fallback bounce detection at {current_ball_position}")
            return True, current_ball_position

        return False, None


    def track_bounces_with_ball_trail(self, ball_trail, court_height_threshold=80, velocity_threshold=0.1, slope_threshold=0.005, min_y_change=0.2, y_fallback_threshold=5):
        """
        Bounce detection using the ball trail by analyzing ball height, velocity, and turning points.
        """
        if len(ball_trail) < 3:
            # Not enough points in the ball trail to detect a bounce
            return False, None
        
        # Get the current, previous, and two frames ago positions from the ball trail
        curr_position = ball_trail[-1]
        prev_position = ball_trail[-2]
        two_frames_ago_position = ball_trail[-3]

        # Get y-coordinates of the ball
        curr_y = curr_position[1]
        prev_y = prev_position[1]
        two_frames_ago_y = two_frames_ago_position[1]

        # **Height Threshold**: Only detect bounces when the ball is close to the ground
        if curr_y > court_height_threshold:
            print(f"[DEBUG] Ball is too high. No bounce detected.")
            return False, None

        # Check if Y-coordinate changes are significant (small bounces may have subtle changes)
        if abs(curr_y - prev_y) < min_y_change and abs(prev_y - two_frames_ago_y) < min_y_change:
            print(f"[DEBUG] Y-coordinate change too small. No bounce detected.")
            return False, None

        # **Velocity Calculation**: Calculate the vertical velocity (change in y position)
        velocity_before = two_frames_ago_y - prev_y
        velocity_now = prev_y - curr_y

        # Log velocities for debugging
        print(f"[DEBUG] Velocity Before: {velocity_before}, Velocity Now: {velocity_now}")

        # **Turning Point Detection**: Check if the ball is turning from falling to rising
        if velocity_before < 0 and velocity_now > 0 and abs(velocity_before - velocity_now) > velocity_threshold:
            print(f"[DEBUG] Bounce detected at {curr_position}")
            return True, curr_position

        # Fallback condition for small bounces or missed detections
        if curr_y < y_fallback_threshold and abs(velocity_now) > 0:
            print(f"[DEBUG] Fallback bounce detection at {curr_position}")
            return True, curr_position

        return False, None

    
    def draw_score_on_frame(self, frame):
        """Draw the current score for both players on the frame."""
        frame_height, frame_width = frame.shape[:2]
        score_box_position = (10, frame_height - 120)
        box_width = 410
        box_height = 80

        # Draw score box
        cv2.rectangle(frame, score_box_position, 
                      (score_box_position[0] + box_width, score_box_position[1] + box_height), 
                      (0, 0, 0), -1)

        # Display Player 1 and Player 2 points, games, and sets
        cv2.putText(frame, f"Player 1: {self.player_scores[1]} pts | {self.game_scores[1]} games | {self.set_scores[1]} sets", 
                    (20, frame_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player 2: {self.player_scores[2]} pts | {self.game_scores[2]} games | {self.set_scores[2]} sets", 
                    (20, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


# # Get the net position
#             net_x = mini_court.get_net_position()  # Ensure you have the net's x-coordinate

#             # Track bounces using the ball trail
#             bounce_detected, bounce_position = score_tracker.track_bounces_using_ball_trail(
#                 ball_trail, net_x, min_trail_length=5, curvature_threshold=5
#             )

#             if bounce_detected:
#                 print(f"[DEBUG] Frame {i}: Ball bounced at {bounce_position}")
#                 bounce_positions.append(ball_center)  # Store the bounce position

#             # Draw bounce positions as red dots on the main screen (video frame)
#             for bounce in bounce_positions:
#                 cv2.circle(frame, (int(bounce[0]), int(bounce[1])), 10, (0, 0, 255), -1)  # Red dot for bounce