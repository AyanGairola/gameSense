import numpy as np
import cv2
import math
from scipy.ndimage import gaussian_filter1d


class EventScoreTracker:
    def __init__(self, mini_court, court_keypoints):
        """
        Initialize the score tracker with the mini court.
        mini_court: instance of MiniCourt class, used to detect fouls based on mini court boundaries.
        """
        self.mini_court = mini_court
        self.player_scores = {1: 0, 2: 0}
        self.game_scores = {1: 0, 2: 0}
        self.set_scores = {1: 0, 2: 0}
        self.last_shot_player = None
        self.foul_detected = False
        self.ball_trail = []
        self.last_foul_frame = -1  # Keep track of the last frame a foul was detected
        self.foul_timeout = 20  # Timeout to avoid consecutive fouls
        self.ball_passed_player = False  # To track if ball has passed the opponent player
        self.previous_positions = []
        self.velocity_history = []
        self.court_keypoints = court_keypoints

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

    def detect_foul(self, ball_position_mini_court, player_hit, net_x, player_positions, ball_trail, ball_hit_frames, current_frame, rally_in_progress):
        """
        Detect fouls based on the ball's position on the mini court.
        Focus on two fouls:
        - Detect ball passing the opponent player
        - Detect if the ball bounced out of bounds
        """    

        # If a foul was detected recently, skip foul detection
        if current_frame - self.last_foul_frame <= self.foul_timeout:
            return None, None

        # Check for ball passing the opponent player
        ball_passed_player = self._did_ball_pass_player(
            ball_position=ball_position_mini_court, 
            player_positions=player_positions, 
            rally_in_progress=rally_in_progress
        )

        if ball_passed_player:
            self.last_foul_frame = current_frame
            return "ball_passed_player", self._get_other_player(player_hit)

        # Check for out-of-bounds using bounce detection
        is_bounce, bounce_position = self.track_bounces(
            ball_trail=ball_trail, 
            current_position=ball_position_mini_court, 
            player_positions=player_positions, 
            ball_hit_frames=ball_hit_frames
        )
        
        # if is_bounce and self._is_ball_out_of_bounds_on_actual_court(bounce_position, player_hit):
        #     self.last_foul_frame = current_frame
        #     return "out_of_bounds", self._get_other_player(player_hit)

        return None, None

    
    def reset_foul_state(self):
        """Reset the foul detection state for a new rally."""
        self.foul_detected = False

    def _is_net_hit(self, ball_position, player_hit, net_x, net_height_threshold=50):
        """
        Check if the ball has hit the net based on its position relative to the net and its height.

        :param ball_position: (x, y) position of the ball
        :param player_hit: The player who last hit the ball (1 or 2)
        :param net_x: The x-coordinate of the net
        :param net_height_threshold: The height threshold to consider if the ball hit the net
        :return: True if the ball hit the net, False otherwise
        """
        ball_x, ball_y = ball_position

        # Check if the ball is below the net height threshold
        if ball_y > net_height_threshold:
            return False  # Ball is above the net, no net hit

        # Check if the ball has crossed the net based on x-position
        if player_hit == 1 and ball_x > net_x:
            return False  # Player 1 hit, ball crossed the net
        if player_hit == 2 and ball_x < net_x:
            return False  # Player 2 hit, ball crossed the net

        # If the ball is below the threshold and has not crossed the net, it's a net hit
        return True

    def _get_other_player(self, player_hit):
        """Get the opponent player."""
        return 2 if player_hit == 1 else 1

    def track_bounces(self, ball_trail, current_position, player_positions, window_size=5, vertical_velocity_threshold=1.9, direction_change_threshold=0.9, velocity_slowdown_threshold=0.5, horizontal_threshold=20, min_time_between_bounces=20, ball_hit_frames=[]):
        """
        Detect bounces using velocity changes, direction reversals, and horizontal analysis.

        :param ball_trail: List of previous ball positions
        :param current_position: Current ball position (x, y)
        :param player_positions: List of player positions
        :param window_size: Number of recent positions to consider
        :param vertical_velocity_threshold: Minimum vertical velocity change to consider for a bounce
        :param direction_change_threshold: Threshold for direction change
        :param velocity_slowdown_threshold: Threshold to detect significant slowdown
        :param horizontal_threshold: Threshold for horizontal movement
        :param min_time_between_bounces: Minimum number of frames between detected bounces
        :param ball_hit_frames: List of frames where ball hits are detected
        :return: Tuple (is_bounce, bounce_position)
        """
        current_frame = len(ball_trail)

        # Skip if the current frame is in the ball_hit_frames
        if current_frame in ball_hit_frames:
            return False, None

        # Ensure we have enough positions to perform analysis
        positions = ball_trail[-window_size:] + [current_position]
        if len(positions) < window_size + 1:
            return False, None

        # Calculate velocities between consecutive positions
        velocities = [
            (positions[i+1][0] - positions[i][0], positions[i+1][1] - positions[i][1])
            for i in range(len(positions) - 1)
        ]
        
        # Calculate vertical accelerations and smooth them
        vertical_accelerations = gaussian_filter1d([
            velocities[i+1][1] - velocities[i][1]
            for i in range(len(velocities) - 1)
        ], sigma=1)

        # Horizontal movements between frames
        horizontal_movements = [
            abs(velocities[i][0]) for i in range(len(velocities))
        ]

        # Detect candidates for bounces based on vertical acceleration and direction change
        bounce_candidates = []
        last_bounce_frame = -min_time_between_bounces  # Initialize to ensure the first bounce is detected
        
        for i in range(len(vertical_accelerations) - 1):
            if vertical_accelerations[i] < 0 and vertical_accelerations[i+1] > 0:  # Reversal of direction
                # Check the magnitude of the velocity change and horizontal movement
                velocity_change = abs(velocities[i+1][1] - velocities[i][1])
                horizontal_change = sum(horizontal_movements[i:i+2]) / 2  # Averaging horizontal movements

                # Consider it a bounce if vertical change is large, horizontal change is small, and the bounce occurs far from the player
                if velocity_change > vertical_velocity_threshold and horizontal_change < horizontal_threshold:
                    # Ensure that the bounce is not happening too soon after the previous bounce
                    actual_frame_index = current_frame - (window_size - i)  # Calculate the actual frame index of the potential bounce
                    if (actual_frame_index - last_bounce_frame) > min_time_between_bounces:
                        bounce_candidates.append((actual_frame_index, velocity_change))
                        last_bounce_frame = actual_frame_index  # Update last bounce frame

        if bounce_candidates:
            # Select the bounce with the largest velocity change
            best_bounce = max(bounce_candidates, key=lambda x: x[1])
            bounce_position = positions[best_bounce[0] - (current_frame - len(positions))]  # The bounce happens after the reversal point

            return True, bounce_position

        return False, None

    
    
    def detect_foul(self, ball_position, player_hit, net_x, player_positions, ball_trail, ball_hit_frames, current_frame, rally_in_progress):
        """
        Detect fouls based on the ball's position on the mini court.
        Focus on two fouls:
        - Detect ball passing the opponent player
        - Detect if the ball bounced out of bounds
        """
        

        # If a foul was detected recently, skip foul detection
        if current_frame - self.last_foul_frame <= self.foul_timeout:
            return None, None

        # Check for ball passing the opponent player
        ball_passed_player = self._did_ball_pass_player(
            ball_position=ball_position, 
            player_positions=player_positions, 
            rally_in_progress=rally_in_progress,
            current_frame=current_frame
        )

        if ball_passed_player:
            self.last_foul_frame = current_frame
            return "ball_passed_player", self._get_other_player(player_hit)

        # Check for out-of-bounds using bounce detection
        is_bounce, bounce_position = self.track_bounces(
            ball_trail=ball_trail, 
            current_position=ball_position, 
            player_positions=player_positions, 
            ball_hit_frames=ball_hit_frames
        )
        
        # if is_bounce and self._is_ball_out_of_bounds_on_actual_court(bounce_position, player_hit):
        #     self.last_foul_frame = current_frame
        #     return "out_of_bounds", self._get_other_player(player_hit)

        return None, None
    
    
    def _calculate_angle_between_vectors(self, vector1, vector2):
        """
        Helper function to calculate the angle between two vectors in degrees.

        :param vector1: First vector (x, y)
        :param vector2: Second vector (x, y)
        :return: Angle between the vectors in degrees
        """
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        # Calculate the angle using the dot product formula
        cos_angle = dot_product / (magnitude1 * magnitude2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # To handle any floating-point errors
        angle_radians = np.arccos(cos_angle)

        # Convert to degrees
        return np.degrees(angle_radians)

    
    
    def draw_score_on_frame(self, frame):
        """Draw the current score for both players on the frame."""
        frame_height, frame_width = frame.shape[:2]
        score_box_position = (10, frame_height - 140)
        box_width = 410
        box_height = 80

        # Draw score box
        cv2.rectangle(frame, score_box_position, (score_box_position[0] + box_width, score_box_position[1] + box_height), (0, 0, 0), -1)

        # Display Player 1 and Player 2 points, games, and sets
        cv2.putText(frame, f"Player 1: {self.player_scores[1]} pts | {self.game_scores[1]} games | {self.set_scores[1]} sets", 
                    (20, frame_height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player 2: {self.player_scores[2]} pts | {self.game_scores[2]} games | {self.set_scores[2]} sets", 
                    (20, frame_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
    
    
    def _is_ball_out_of_bounds_on_actual_court(self, bounce_position, player_hit):
        """
        Check if the ball has bounced out of bounds using the actual court boundaries.

        :param bounce_position: The (x, y) position where the ball has bounced on the actual court.
        :param player_hit: Player who last hit the ball (1 or 2).
        :return: Boolean indicating if the ball bounced out of bounds.
        """
        # Extract the necessary keypoints to define the boundaries of the actual court
        p8_x, p8_y = self.court_keypoints[16], self.court_keypoints[17]    # A point on the court
        p9_x, p9_y = self.court_keypoints[18], self.court_keypoints[19]    # A point on the court
        p10_x, p10_y = self.court_keypoints[20], self.court_keypoints[21]  # A point on the court
        p11_x, p11_y = self.court_keypoints[22], self.court_keypoints[23]  # A point on the court
        p12_x, p12_y = self.court_keypoints[24], self.court_keypoints[25]  # A point on the court
        p13_x, p13_y = self.court_keypoints[26], self.court_keypoints[27]  # A point on the court

        # Create a boundary polygon for the actual court using the court keypoints
        boundary_polygon = np.array([
            [p8_x, p8_y], [p9_x, p9_y], [p10_x, p10_y], [p11_x, p11_y], [p12_x, p12_y], [p13_x, p13_y]
        ], dtype=np.float32).reshape((-1, 1, 2))

        # Check if the bounce's position is outside this polygon
        bounce_x, bounce_y = bounce_position
        result = cv2.pointPolygonTest(boundary_polygon, (float(bounce_x), float(bounce_y)), False)

        # If result is -1, the bounce is out of bounds
        return result == -1


    
    def _did_ball_pass_player(self, ball_position, player_positions, rally_in_progress, current_frame, min_frame_gap=150, distance_threshold=50):
        """
        Check if the ball has passed the opponent player after being hit.

        :param ball_position: Current ball position (x, y)
        :param player_positions: List or dictionary of player positions
        :param rally_in_progress: Boolean indicating whether the rally is currently active
        :param current_frame: The current frame number (pass this from the main function)
        :param min_frame_gap: Minimum gap between frames to avoid continuous foul detection
        :param distance_threshold: Distance threshold to confirm if the ball has truly passed the player
        :return: Boolean indicating if the ball has passed the opponent player
        """
        # Get the opponent player
        opponent_player = 2 if self.last_shot_player == 1 else 1

        # Ensure player_positions is a dictionary or a list with valid player positions
        if isinstance(player_positions, dict):
            player_position = player_positions.get(opponent_player, None)
        else:
            player_position = player_positions[opponent_player - 1]  # Adjust index for 1-based to 0-based indexing
        
        if player_position is None:
            return False  # No valid position for the opponent, assume ball hasn't passed

        # Extract ball's center if the position is given as a bounding box (x1, y1, x2, y2)
        if isinstance(ball_position, (list, tuple)) and len(ball_position) == 4:
            ball_x = (ball_position[0] + ball_position[2]) / 2  # Average of x1 and x2
            ball_y = (ball_position[1] + ball_position[3]) / 2  # Average of y1 and y2
        else:
            ball_x, ball_y = ball_position

        # Extract player's x and y position
        player_x, player_y = player_position

        # Use player's orientation to decide if the ball is behind the player
        player_orientation_vector = self._calculate_player_orientation(opponent_player)

        # Calculate the vector from player to the ball
        ball_vector = np.array([ball_x - player_x, ball_y - player_y])

        # Check if the ball has passed behind the player by comparing the ball vector with the player's orientation
        passed_player = np.dot(player_orientation_vector, ball_vector) < 0

        # Check frame gap to avoid multiple fouls in consecutive frames
        if self.last_foul_frame is None or (current_frame - self.last_foul_frame) > min_frame_gap:
            self.last_foul_frame = current_frame  # Update last foul frame
            
            # Apply the distance threshold
            distance = np.linalg.norm(np.array([ball_x, ball_y]) - np.array([player_x, player_y]))
            if distance > distance_threshold and passed_player:
                return True  # Ball has passed the player

        return False  # Ball hasn't passed the player
    
    
    
    
    def _check_ball_movement_over_frames(self, current_ball_position, current_player_position):
        """
        Check if the ball consistently passed the player over multiple frames.
        This prevents false positives from a single frame.

        :param current_ball_position: Ball's current position (x, y)
        :param current_player_position: Player's current position (x, y)
        :return: Boolean indicating whether the ball consistently passed the player
        """
        # Track the last few ball and player positions to ensure consistent movement
        # Store these positions in the instance as history
        self.ball_position_history.append(current_ball_position)
        self.player_position_history.append(current_player_position)

        # Check if the ball consistently moved behind the player
        # We can use a threshold of 3 frames to confirm the ball has truly passed
        if len(self.ball_position_history) >= 3:
            for i in range(-3, 0):
                ball_pos = self.ball_position_history[i]
                player_pos = self.player_position_history[i]

                # Calculate the vector from player to the ball
                ball_vector = np.array([ball_pos[0] - player_pos[0], ball_pos[1] - player_pos[1]])

                # Use the player's orientation to check if the ball is consistently behind
                player_orientation_vector = self._calculate_player_orientation(self.last_shot_player)
                if np.dot(player_orientation_vector, ball_vector) > 0:
                    # If the ball hasn't been consistently behind, clear history and return False
                    self.ball_position_history = []
                    self.player_position_history = []
                    return False

            # If the ball consistently passed over multiple frames, clear history and return True
            self.ball_position_history = []
            self.player_position_history = []
            return True

        return False  # Not enough frames to make a decision yet
    
    
    def _calculate_player_orientation(self, player_idx):
        """
        Calculate the orientation of the player based on their current and previous positions.
        
        :param player_idx: The index of the player (1 or 2)
        :return: Orientation vector of the player
        """
        if len(self.previous_positions) < 2:
            # If not enough frames have passed, assume orientation is straight ahead (default)
            return np.array([1, 0])

        # Get previous and current positions for the player
        previous_pos = self.previous_positions[player_idx - 1]  # Adjust for 1-based indexing
        current_pos = self.current_positions[player_idx - 1]

        # Calculate the orientation vector based on movement direction
        orientation_vector = np.array(current_pos) - np.array(previous_pos)

        # Normalize the orientation vector to unit length
        norm = np.linalg.norm(orientation_vector)
        if norm == 0:  # Prevent division by zero
            return np.array([1, 0])
        
        return orientation_vector / norm  # Return the normalized orientation vector


    def _is_behind_player(self, player_orientation, ball_center_x, ball_center_y, player_pos):
        """
        Determine if the ball is behind the player based on the player's orientation.
        
        :param player_orientation: Orientation vector of the player
        :param ball_center_x: X-coordinate of the ball's center
        :param ball_center_y: Y-coordinate of the ball's center
        :param player_pos: Current position of the player
        :return: Boolean indicating if the ball is behind the player
        """
        ball_vector = np.array([ball_center_x, ball_center_y]) - np.array(player_pos)
        dot_product = np.dot(player_orientation, ball_vector)
        
        # If the dot product is negative, the ball is behind the player
        return dot_product < 0