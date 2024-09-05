import numpy as np
import pandas as pd
from collections import deque

# Define the court dimensions in meters
COURT_LENGTH_M = 23.77  # Tennis court length in meters
COURT_WIDTH_M = 8.23    # Tennis court width in meters (for singles)

# Known pixel positions for court corners (replace with actual values)
# These should be determined from the video frame
top_left = (100, 200)
top_right = (500, 200)
bottom_left = (100, 800)
bottom_right = (500, 800)

# Calculate pixel distances
pixel_length = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
pixel_width = np.sqrt((bottom_right[0] - bottom_left[0])**2 + (bottom_right[1] - bottom_left[1])**2)

# Conversion factors (pixels to meters)
pixels_per_meter_length = pixel_length / COURT_LENGTH_M
pixels_per_meter_width = pixel_width / COURT_WIDTH_M

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

    # Use bbox as a unique identifier for each player
    player_id = hash(tuple(bbox))
    if player_id not in calculate_player_speed.positions:
        calculate_player_speed.positions[player_id] = deque(maxlen=5)

    # Calculate the center of the bounding box
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    # Append the current frame and center position
    calculate_player_speed.positions[player_id].append((frame_number, center))
    
    # If less than 2 positions, return speed as 0
    if len(calculate_player_speed.positions[player_id]) < 2:
        return 0

    # Calculate speed based on the first and last positions in the deque
    first_frame, first_pos = calculate_player_speed.positions[player_id][0]
    last_frame, last_pos = calculate_player_speed.positions[player_id][-1]
    
    # Calculate time difference
    time_diff = (last_frame - first_frame) / fps
    if time_diff == 0:
        return 0
    
    # Calculate the Euclidean distance between the first and last positions
    distance = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
    
    # Convert the speed to pixels per second
    speed_pixels_per_second = distance / time_diff
    
    # Convert pixels per second to km/h (assuming 1 pixel = 2.54 cm)
    speed_km_h = (speed_pixels_per_second * 2.54 / 100) * 3.6

    print(f"Frame {frame_number}: Player {player_id} bbox {bbox} center {center}")
    print(f"Positions for Player {player_id}: {calculate_player_speed.positions[player_id]}")
    
    # return min(max(speed_km_h, 0), 35)  # Cap between 0 and 35 km/h

    return speed_km_h

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
    
    # Calculate real-world distance in meters
    distance_pixels = np.sqrt((last_pos[0] - first_pos[0])**2 + (last_pos[1] - first_pos[1])**2)
    distance_meters = distance_pixels / pixels_per_meter_length
    
    speed_mps = distance_meters / time_diff  # Speed in meters per second
    
    # Convert to km/h
    speed_kmph = speed_mps * 3.6
    
    # return min(max(speed_kmph, 0), 220)  # Cap between 0 and 220 km/h
    return speed_kmph

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