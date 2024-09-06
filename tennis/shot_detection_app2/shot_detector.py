

def detect_shot_type(player_position, ball_position, previous_point_ended):
    """
    Detect the type of shot (Forehand, Backhand, Serve) based on the player and ball position.
    
    :param player_position: Tuple (x, y) representing the player's position.
    :param ball_position: Tuple (x, y) representing the ball's position.
    :param previous_point_ended: Boolean indicating if the previous point ended (for serve detection).
    
    :return: String representing the shot type.
    """
    player_x = player_position[0]
    ball_x = ball_position[0]

    if previous_point_ended:
        return "Serve"
    
    if ball_x > player_x:
        return "Forehand"
    else:
        return "Backhand"