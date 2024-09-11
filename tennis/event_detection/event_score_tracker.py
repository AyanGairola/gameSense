import numpy as np
import cv2
import math
from scipy.signal import find_peaks
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
        - Detect net hit
        - Detect ball passing the opponent player
        - Detect if the ball bounced out of bounds
        """
        # If a foul was detected recently, skip foul detection
        if current_frame - self.last_foul_frame <= self.foul_timeout:
            return None, None

        # Check for net hit
        net_hit = self._is_net_hit(ball_position_mini_court, player_hit, net_x)
        if net_hit:
            self.last_foul_frame = current_frame
            return "net_hit", self._get_other_player(player_hit)

        # Check for ball passing the opponent player
        ball_passed_player = self._did_ball_pass_player(
            ball_position=ball_position_mini_court, 
            player_positions=player_positions, 
            player_hit=player_hit, 
            rally_in_progress=rally_in_progress  # Pass rally_in_progress flag
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
        
        if is_bounce and self._is_ball_out_of_bounds_on_actual_court(bounce_position, player_hit, ball_trail, player_positions, ball_hit_frames):
            self.last_foul_frame = current_frame
            return "out_of_bounds", self._get_other_player(player_hit)

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


    def calculate_angle(x1, y1, x2, y2):
        """Calculate the angle between two points (in degrees)."""
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    def calculate_slope(x1, y1, x2, y2):
        """Calculate the slope between two points."""
        if x2 - x1 == 0:
            return float('inf')  # Avoid division by zero
        return (y2 - y1) / (x2 - x1)

    
    
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
        
        # Calculate vertical accelerations
        vertical_accelerations = [
            velocities[i+1][1] - velocities[i][1]
            for i in range(len(velocities) - 1)
        ]

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


    def _is_near_player(self, ball_position, player_positions, proximity_threshold=50):
        """
        Check if the ball is near either player, indicating that it's likely a hit and not a bounce.
        
        :param ball_position: Current ball position
        :param player_positions: List of player positions
        :param proximity_threshold: Distance threshold to determine if the ball is near a player
        :return: Boolean indicating if the ball is near a player
        """
        for player_position in player_positions:
            distance = np.linalg.norm(np.array(ball_position) - np.array(player_position))
            if distance < proximity_threshold:
                return True
            
        return False

    def draw_score_on_frame(self, frame):
        """Draw the current score for both players on the frame."""
        frame_height, frame_width = frame.shape[:2]
        score_box_position = (10, frame_height - 120)
        box_width = 410
        box_height = 80

        # Draw score box
        cv2.rectangle(frame, score_box_position,(score_box_position[0] + box_width, score_box_position[1] + box_height), (0, 0, 0), -1)

        # Display Player 1 and Player 2 points, games, and sets
        cv2.putText(frame, f"Player 1: {self.player_scores[1]} pts | {self.game_scores[1]} games | {self.set_scores[1]} sets", 
                    (20, frame_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Player 2: {self.player_scores[2]} pts | {self.game_scores[2]} games | {self.set_scores[2]} sets", 
                    (20, frame_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame
    
    
    def _did_ball_pass_player(self, ball_position, player_positions, player_hit, rally_in_progress):
        """
        Check if the ball has passed the opponent player after being hit.
        
        :param ball_position: Current ball position (x, y)
        :param player_positions: List or dictionary of player positions
        :param player_hit: The player who hit the ball (1 or 2)
        :param rally_in_progress: Boolean indicating whether the rally is currently active
        :return: Boolean indicating if the ball has passed the opponent player
        """
        # Get the opponent player
        opponent_player = 2 if player_hit == 1 else 1

        # Ensure player_positions is a dictionary or a list with valid player positions
        if isinstance(player_positions, dict):
            player_position = player_positions.get(opponent_player, None)
        else:
            player_position = player_positions[opponent_player - 1]  # Adjust index for 1-based to 0-based indexing
        
        if player_position is None:
            return False  # No valid position for the opponent, assume ball hasn't passed

        # Check if the ball has passed the opponent based on its x-coordinate
        ball_x, ball_y = ball_position
        player_x, player_y = player_position

        # Track if ball has passed and reset at the start of each rally
        if not rally_in_progress:  # This means a new rally or shot has started
            self.ball_passed_player = False  # Reset the pass detection flag
        
        if not self.ball_passed_player:
            if player_hit == 1 and ball_x > player_x:
                self.ball_passed_player = True  # Mark that the ball has passed Player 2 (opponent)
                return True
            elif player_hit == 2 and ball_x < player_x:
                self.ball_passed_player = True  # Mark that the ball has passed Player 1 (opponent)
                return True

        return False  # Ball hasn't passed the opponent


