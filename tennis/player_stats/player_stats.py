import pandas as pd
from utils import measure_distance, convert_pixel_distance_to_meters
import constants

def calculate_player_stats(ball_shot_frames, ball_mini_court_detections, player_mini_court_detections, mini_court, video_frames):
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_distance': 0,
        'player_1_current_speed': 0,
        'player_1_average_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_distance': 0,
        'player_2_current_speed': 0,
        'player_2_average_speed': 0,
    }]
    
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]

        ball_shot_time_in_seconds = (end_frame - start_frame) / 24.0

        if (start_frame >= len(player_mini_court_detections) or player_mini_court_detections[start_frame] is None or
            start_frame >= len(ball_mini_court_detections) or ball_mini_court_detections[start_frame] is None or
            end_frame >= len(player_mini_court_detections) or player_mini_court_detections[end_frame] is None or
            end_frame >= len(ball_mini_court_detections) or ball_mini_court_detections[end_frame] is None):
            print(f"Warning: Missing or invalid detections between frames {start_frame} and {end_frame}")
            continue

        player_positions_start = player_mini_court_detections[start_frame]
        player_positions_end = player_mini_court_detections[end_frame]
        ball_start = ball_mini_court_detections[start_frame]
        ball_end = ball_mini_court_detections[end_frame]

        distance_covered_by_ball_pixels = measure_distance(ball_start, ball_end)
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels,
            constants.DOUBLE_LINE_WIDTH,
            mini_court.get_width_of_mini_court()
        )

        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        player_shot_ball = min(player_positions_start.keys(), key=lambda player_id: 
                                measure_distance(player_positions_start[player_id], ball_start))

        current_player_stats = player_stats_data[-1].copy()
        current_player_stats['frame_num'] = start_frame

        for player_id in [1, 2]:
            if player_id in player_positions_start and player_id in player_positions_end:
                distance_covered_pixels = measure_distance(player_positions_start[player_id], player_positions_end[player_id])
                distance_covered_meters = convert_pixel_distance_to_meters(
                    distance_covered_pixels,
                    constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court()
                )
                player_speed = distance_covered_meters / ball_shot_time_in_seconds * 3.6  # km/h

                current_player_stats[f'player_{player_id}_total_distance'] += distance_covered_meters
                current_player_stats[f'player_{player_id}_current_speed'] = player_speed
                current_player_stats[f'player_{player_id}_average_speed'] = (
                    (current_player_stats[f'player_{player_id}_average_speed'] * ball_shot_ind + player_speed) / (ball_shot_ind + 1)
                )

                if player_id == player_shot_ball:
                    current_player_stats[f'player_{player_id}_number_of_shots'] += 1
                    current_player_stats[f'player_{player_id}_total_shot_speed'] += speed_of_ball_shot
                    current_player_stats[f'player_{player_id}_last_shot_speed'] = speed_of_ball_shot

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    for player_id in [1, 2]:
        player_stats_data_df[f'player_{player_id}_average_shot_speed'] = (
            player_stats_data_df[f'player_{player_id}_total_shot_speed'] / 
            player_stats_data_df[f'player_{player_id}_number_of_shots'].replace(0, 1)
        )

    return player_stats_data_df