import pandas as pd
from utils import measure_distance, convert_pixel_distance_to_meters
import constants

def calculate_player_stats(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, mini_court, video_frames):
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]
    
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]

        # Calculate ball shot time in seconds assuming 24fps
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24.0  # Assuming 24fps

        # Ensure the start_frame is within bounds
        if start_frame >= len(player_mini_court_detections) or player_mini_court_detections[start_frame] is None:
            print(f"Warning: Missing or invalid player detections at frame {start_frame}")
            continue  # Skip this iteration if player detections are missing
        
        if start_frame >= len(ball_mini_court_detections) or ball_mini_court_detections[start_frame] is None:
            print(f"Warning: Missing or invalid ball detections at frame {start_frame}")
            continue  # Skip this iteration if ball detections are missing

        # Ensure the end_frame is within bounds
        if end_frame >= len(player_mini_court_detections) or player_mini_court_detections[end_frame] is None:
            print(f"Warning: Missing or invalid player detections at frame {end_frame}")
            continue  # Skip this iteration if player detections are missing
        
        if end_frame >= len(ball_mini_court_detections) or ball_mini_court_detections[end_frame] is None:
            print(f"Warning: Missing or invalid ball detections at frame {end_frame}")
            continue  # Skip this iteration if ball detections are missing

        # Proceed with the calculation if data is valid
        player_positions = player_mini_court_detections[start_frame]
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame], ball_mini_court_detections[end_frame])

        # Convert pixel distance to meters
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Player who shot the ball
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: 
                               measure_distance(player_positions[player_id], ball_mini_court_detections[start_frame]))

        # Opponent player ID
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        speed_of_opponent = convert_pixel_distance_to_meters(
            distance_covered_by_opponent_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        ) / ball_shot_time_in_seconds * 3.6

        # Update player stats
        current_player_stats = player_stats_data[-1].copy()
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # Convert the player stats data to a DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Avoid division by zero for average calculations
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots'].replace(0, 1)
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots'].replace(0, 1)
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_1_number_of_shots'].replace(0, 1)
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_2_number_of_shots'].replace(0, 1)

    return player_stats_data_df
