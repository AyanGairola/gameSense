import numpy as np
import pandas as pd
from collections import deque

def calculate_player_stats(detection, ball_position, frame_number, fps=30):
    player_stats = {
        'frame': frame_number,
        'player_1_last_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_1_last_player_speed': 0,
        'player_2_last_player_speed': 0,
    }

    # Calculate player speeds
    for i, player in enumerate(detection.get('players', [])[:2], start=1):
        if player and 'bbox' in player:
            player_speed = calculate_player_speed(player['bbox'], frame_number, fps)
            player_stats[f'player_{i}_last_player_speed'] = player_speed

    # Calculate ball speed (potential shot speed)
    if ball_position and all(ball_position):
        ball_speed = calculate_ball_speed(ball_position, frame_number, fps)
        
        # Assign shot speed to the closest player
        closest_player = get_closest_player(detection.get('players', []), ball_position)
        if closest_player is not None:
            player_stats[f'player_{closest_player}_last_shot_speed'] = ball_speed

    return player_stats

def process_player_stats_data(player_stats_df):
    # Calculate rolling averages
    window_size = 30  # 1 second at 30 fps
    for i in range(1, 3):
        player_stats_df[f'player_{i}_average_shot_speed'] = player_stats_df[f'player_{i}_last_shot_speed'].rolling(window=window_size, min_periods=1).mean()
        player_stats_df[f'player_{i}_average_player_speed'] = player_stats_df[f'player_{i}_last_player_speed'].rolling(window=window_size, min_periods=1).mean()

    # Replace NaN values with 0
    player_stats_df = player_stats_df.fillna(0)

    return player_stats_df

# Helper functions

def calculate_player_speed(bbox, frame_number, fps):
    if not hasattr(calculate_player_speed, "positions"):
        calculate_player_speed.positions = {}
    
    player_id = hash(tuple(bbox))  # Use bbox as a unique identifier for each player
    if player_id not in calculate_player_speed.positions:
        calculate_player_speed.positions[player_id] = deque(maxlen=5)
    
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    calculate_player_speed.positions[player_id].append((frame_number, center))
    
    if len(calculate_player_speed.positions[player_id]) < 2:
        return 0
    
    first_frame, first_pos = calculate_player_speed.positions[player_id][0]
    last_frame, last_pos = calculate_player_speed.positions[player_id][-1]
    
    time_diff = (last_frame - first_frame) / fps
    if time_diff == 0:
        return 0
    
    distance = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
    speed_pixels_per_second = distance / time_diff
    
    # Convert to km/h (assuming 1 pixel = 2.54 cm)
    speed_km_h = (speed_pixels_per_second * 2.54 / 100) * 3.6
    
    return min(max(speed_km_h, 0), 35)  # Cap between 0 and 35 km/h

def calculate_ball_speed(ball_position, frame_number, fps):
    if not hasattr(calculate_ball_speed, "positions"):
        calculate_ball_speed.positions = deque(maxlen=3)
    
    center = ((ball_position[0] + ball_position[2]) / 2, (ball_position[1] + ball_position[3]) / 2)
    calculate_ball_speed.positions.append((frame_number, center))
    
    if len(calculate_ball_speed.positions) < 2:
        return 0
    
    first_frame, first_pos = calculate_ball_speed.positions[0]
    last_frame, last_pos = calculate_ball_speed.positions[-1]
    
    time_diff = (last_frame - first_frame) / fps
    if time_diff == 0:
        return 0
    
    distance = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
    speed_pixels_per_second = distance / time_diff
    
    # Convert to km/h (assuming 1 pixel = 2.54 cm)
    speed_km_h = (speed_pixels_per_second * 2.54 / 100) * 3.6
    
    return min(max(speed_km_h, 0), 220)  # Cap between 0 and 220 km/h

def get_closest_player(players, ball_position):
    if not players or not ball_position:
        return None
    
    ball_center = ((ball_position[0] + ball_position[2]) / 2, (ball_position[1] + ball_position[3]) / 2)
    
    min_distance = float('inf')
    closest_player = None
    
    for i, player in enumerate(players[:2], start=1):
        if player and 'bbox' in player:
            player_center = ((player['bbox'][0] + player['bbox'][2]) / 2, (player['bbox'][1] + player['bbox'][3]) / 2)
            distance = np.sqrt((ball_center[0] - player_center[0])**2 + (ball_center[1] - player_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_player = i
    
    return closest_player

# Debugging function
def debug_print_stats(frame_number, detection, ball_position, stats):
    print(f"Frame {frame_number}:")
    print(f"  Detection: {detection}")
    print(f"  Ball position: {ball_position}")
    print(f"  Calculated stats: {stats}")
    print("---")