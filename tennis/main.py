from utils import (
    read_video, save_video, measure_distance, draw_player_stats, 
    convert_pixel_distance_to_meters, get_center_of_bbox, 
)
import constants
from court_line_detector import CourtLineDetector
from mini_court import ImprovedMiniCourt
import cv2
from player_stats import calculate_player_stats, process_player_stats_data
from trackers import UnifiedTracker  # Unified tracker is already imported
import numpy as np

from kalman_Filter import KalmanFilter

def assign_player_ids(players, previous_players=None):
    if not previous_players:
        return {i: player for i, player in enumerate(players)}
    
    new_players = {}
    for player in players:
        min_distance = float('inf')
        closest_id = None
        for prev_id, prev_player in previous_players.items():
            distance = ((player['bbox'][0] - prev_player['bbox'][0])**2 + 
                        (player['bbox'][1] - prev_player['bbox'][1])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                closest_id = prev_id
        if closest_id is not None:
            new_players[closest_id] = player
        else:
            new_id = max(previous_players.keys()) + 1 if previous_players else 0
            new_players[new_id] = player
    return new_players


def main():
    # Read Video
    input_video_path = "input_vods/vod4.mp4"
    video_frames = read_video(input_video_path)

    # Initialize the unified tracker
    unified_tracker = UnifiedTracker(model_path='./models/player_and_ball_detection/best.pt')

    # Detect players and ball using the unified model
    detections = unified_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/unified_detections.pkl")

    # Interpolate ball positions to handle missed detections
    interpolated_positions = unified_tracker.interpolate_ball_positions(detections)

    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)

    # Initialize player ID counter
    next_player_id = 0

    # Process video frame by frame
    output_video_frames = []
    court_keypoints_list = []
    previous_players = {}

    
    for i, (frame, detection) in enumerate(zip(video_frames, detections)):
        # Detect court keypoints for the current frame
        court_keypoints = court_line_detector.predict(frame)
        court_keypoints_list.append(court_keypoints)

        # Draw court keypoints on the frame
        frame = court_line_detector.draw_keypoints(frame, court_keypoints)
        
        # Draw player and ball detections on the frame
        frame = unified_tracker.draw_bboxes([frame], [detection], interpolated_positions=[interpolated_positions[i]])[0]
        
        # Draw the ball path (trailing effect)
        if i > 0:
            prev_pos = interpolated_positions[i - 1]
            curr_pos = interpolated_positions[i]
            if prev_pos and curr_pos and all(prev_pos) and all(curr_pos):
                prev_center = (int((prev_pos[0] + prev_pos[2]) // 2), int((prev_pos[1] + prev_pos[3]) // 2))
                curr_center = (int((curr_pos[0] + curr_pos[2]) // 2), int((curr_pos[1] + curr_pos[3]) // 2))
                cv2.line(frame, prev_center, curr_center, (0, 255, 255), 2)  # Yellow line for the ball path
        
        players_with_ids = {}
        for player in detection.get('players', []):
            if player['class'] == 1:  # Assuming class 1 is player
                player_center = get_center_of_bbox(player['bbox'])
                closest_prev_id = None
                min_distance = float('inf')
                for prev_id, prev_player in previous_players.items():
                    dist = measure_distance(player_center, get_center_of_bbox(prev_player['bbox']))
                    if dist < min_distance:
                        min_distance = dist
                        closest_prev_id = prev_id
                if closest_prev_id is not None and min_distance < 50:  # Threshold for considering it's the same player
                    player_id = closest_prev_id
                else:
                    player_id = next_player_id
                    next_player_id += 1
                players_with_ids[player_id] = player
                player['id'] = player_id

        previous_players = players_with_ids

        # Assign ID to the ball
        for ball in detection.get('ball', []):
            if ball['class'] == 0:  # Assuming class 0 is ball
                ball['id'] = 'ball'

        # Update the detection with the new player and ball IDs
        detection['players'] = list(players_with_ids.values())
        
        print(f"Frame {i}:")
        for player in detection.get('players', []):
            print(f"Player ID: {player.get('id', 'No ID')}, Class: {player['class']}")
        for ball in detection.get('ball', []):
            print(f"Ball ID: {ball.get('id', 'No ID')}, Class: {ball['class']}")
        
        output_video_frames.append(frame)

    # MiniCourt Initialization
    mini_court = ImprovedMiniCourt(video_frames[0])

    # Process the unified detections for mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_detections_to_mini_court_coordinates(detections, court_keypoints_list[0])

    # Initialize Kalman filters for players and ball
    player_kalman_filters = {}
    ball_kalman_filter = None

    # Process and smooth the detections
    smoothed_player_detections = []
    smoothed_ball_detections = []

    for frame_idx, (player_detections, ball_detection) in enumerate(zip(player_mini_court_detections, ball_mini_court_detections)):
        smoothed_player_frame = {}
        for player_id, position in player_detections.items():
            if player_id not in player_kalman_filters:
                player_kalman_filters[player_id] = KalmanFilter([position[0], position[1], 0, 0])
            else:
                player_kalman_filters[player_id].predict()
            smoothed_position = player_kalman_filters[player_id].update(np.array(position))
            smoothed_player_frame[player_id] = tuple(smoothed_position)
        smoothed_player_detections.append(smoothed_player_frame)

        if 'ball' in ball_detection:
            if ball_kalman_filter is None:
                ball_kalman_filter = KalmanFilter([ball_detection['ball'][0], ball_detection['ball'][1], 0, 0])
            else:
                ball_kalman_filter.predict()
            smoothed_ball_position = ball_kalman_filter.update(np.array(ball_detection['ball']))
            smoothed_ball_detections.append({'ball': tuple(smoothed_ball_position)})
        else:
            smoothed_ball_detections.append({})

    # Ensure that player_mini_court_detections and ball_mini_court_detections
    # have the same length as output_video_frames
    num_frames = len(output_video_frames)
    player_mini_court_detections = player_mini_court_detections[:num_frames]
    ball_mini_court_detections = ball_mini_court_detections[:num_frames]

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections, color=(0, 255, 0))  # Green for players
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))  # Yellow for ball

    # Draw frame number on top left corner of each frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the processed video
    save_video(output_video_frames, "./output_vods/oaaa6.mp4")

if __name__ == "__main__":
    main()