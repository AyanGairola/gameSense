from copy import deepcopy
import pandas as pd
from utils import measure_distance, convert_pixel_distance_to_meters

def calculate_player_stats(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, mini_court, constants):
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

    print(f"ball_mini_court_detections: {ball_mini_court_detections}")
    print(f"ball_shot_frames: {ball_shot_frames}")

    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]

        if start_frame >= len(ball_mini_court_detections):
            print(f"start_frame {start_frame} out of range for ball_mini_court_detections with length {len(ball_mini_court_detections)}")
            continue

        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        # Get distance covered by the ball
        if 1 not in ball_mini_court_detections[start_frame]:
            print(f"No ball detected in frame {start_frame}")
            continue
        
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                            ball_mini_court_detections[end_frame][1])
        
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                            constants.DOUBLE_LINE_WIDTH,
                                                                            mini_court.get_width_of_mini_court())

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Player who shot the ball
        player_positions = player_mini_court_detections[start_frame]

        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                ball_mini_court_detections[start_frame][1]))

        # Opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        
        if opponent_player_id not in player_mini_court_detections[start_frame] or opponent_player_id not in player_mini_court_detections[end_frame]:
            print(f"Opponent player ID {opponent_player_id} not found in frames {start_frame} or {end_frame}")
            continue
        
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                                constants.DOUBLE_LINE_WIDTH,
                                                                                mini_court.get_width_of_mini_court())

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    return player_stats_data_df

def process_player_stats_data(player_stats_data_df, video_frames):
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']

    return player_stats_data_df