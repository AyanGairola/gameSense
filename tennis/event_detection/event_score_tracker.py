import numpy as np
import cv2

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

    def detect_foul(self, ball_position, player_hit, net_x, player_positions):
        """
        Detect fouls based on the ball's position on the mini court.
        - Check if the ball has gone out of bounds on the first contact in the opponent's half
        - Check if the ball has hit the net (remains on the same side after a player hits it)
        """
        if self.foul_detected:  # Prevent repeated foul detection
            return None, None

        # Check for net hit
        net_hit = self._is_net_hit(ball_position, player_hit, net_x)
        if net_hit:
            self.foul_detected = True
            return "net_hit", self._get_other_player(player_hit)

        # Check for out of bounds on the opponent's side of the court
        out_of_bounds = self._is_ball_out_of_bounds(ball_position, player_hit, net_x)
        if out_of_bounds:
            self.foul_detected = True
            return "out_of_bounds", self._get_other_player(player_hit)

        return None, None  # No foul detected

    def _is_ball_out_of_bounds(self, ball_position, player_hit, net_x):
        """Check if the ball is out of bounds using mini court boundaries."""
        ball_x, ball_y = ball_position

        # Ensure ball has crossed the net and is in the opponent's half
        if (player_hit == 1 and ball_x <= net_x) or (player_hit == 2 and ball_x >= net_x):
            return False  # Ball hasn't reached the opponent's half yet

        # Check if ball position is outside court boundaries
        mini_court_width = self.mini_court.get_width_of_mini_court()
        if ball_x < 0 or ball_x > mini_court_width or ball_y < 0 or ball_y > self.mini_court.drawing_rectangle_height:
            return True  # Ball is out of bounds

        return False

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
