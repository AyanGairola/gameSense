from utils import (
    read_video, save_video, measure_distance, draw_player_stats, 
    convert_pixel_distance_to_meters, get_center_of_bbox, 
)
import constants
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from player_stats import calculate_player_stats
from trackers import UnifiedTracker
import numpy as np
import pandas as pd
from rally import RallyDetector
from shot_detection_app2.shot_detector import detect_shot_type
from event_detection import TacticalAnalysis

def main():
    # Read Video
    input_video_path = "input_vods/vod4.mp4"
    video_frames = read_video(input_video_path)

    # Initialize the UnifiedTracker for detecting players and ball
    unified_tracker = UnifiedTracker(model_path='./models/player_and_ball_detection/best.pt')

    # Detect players and ball using the unified model
    detections = unified_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/unified_detections.pkl")
    print(f"Type of detections: {type(detections)}")
    print(f"Number of frames with detections: {len(detections)}")

    # Interpolate ball positions to handle missed detections
    interpolated_positions = unified_tracker.interpolate_ball_positions(detections)
    ball_hit_frames = unified_tracker.get_ball_shot_frames(detections)

    # Initialize Court Line Detector
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    
    # Initialize MiniCourt for the aerial view
    mini_court = MiniCourt(video_frames[0])

    # Get court keypoints from the first frame and initialize Tactical Analysis
    court_keypoints = court_line_detector.predict(video_frames[0])
    tactical_analysis = TacticalAnalysis(court_keypoints)

    # Initialize RallyDetector
    rally_detector = RallyDetector(mini_court)

    output_video_frames = []
    ball_mini_court_detections = []
    player_mini_court_detections = []

    previous_point_ended = False
    ball_trail = []
    rally_count = 0
    net_x = None  # Initialize net x-position

    for i, (frame, detection) in enumerate(zip(video_frames, detections)):
        # Get court keypoints for the current frame
        court_keypoints = court_line_detector.predict(frame)
        
        # Calculate net position (middle of keypoints 0 and 2)
        if net_x is None:
            if len(court_keypoints) >= 4:  # Ensure there are enough points
                net_x = (court_keypoints[0] + court_keypoints[2]) // 2

        # Process player positions
        if 'players' in detection and len(detection['players']) >= 2:
            player_1_position = get_center_of_bbox(detection['players'][0]['bbox'])  # First player's position
            player_2_position = get_center_of_bbox(detection['players'][1]['bbox'])  # Second player's position
        else:
            print(f"Warning: Player positions not available for frame {i}")
            continue  # Skip this frame if there are not enough players

        # Draw court keypoints on the frame
        frame = court_line_detector.draw_keypoints(frame, court_keypoints)

        # Draw player and ball detections on the frame
        frame = unified_tracker.draw_bboxes([frame], [detection], interpolated_positions=[interpolated_positions[i]])[0]

        # Ensure ball position exists and process it
        ball_position = interpolated_positions[i]
        if ball_position:
            ball_mini_court = mini_court.video_to_court_coordinates(get_center_of_bbox(ball_position))
            ball_mini_court_detections.append(ball_mini_court)

            # Update rally count if the ball crosses the net
            if net_x is not None:
                if i > 0:
                    prev_ball_position = interpolated_positions[i-1]
                    if prev_ball_position:
                        # Check if the ball crossed the net
                        if (prev_ball_position[0] < net_x and ball_position[0] >= net_x) or \
                                (prev_ball_position[0] > net_x and ball_position[0] <= net_x):
                            rally_count += 1

            # Add current ball position to the ball trail
            ball_trail.append(get_center_of_bbox(ball_position))
            if len(ball_trail) > 10:  # Keep the ball trail within the last 10 positions
                ball_trail.pop(0)

            # Draw the ball trail on the frame
            for j in range(1, len(ball_trail)):
                if ball_trail[j - 1] is None or ball_trail[j] is None:
                    continue
                prev_center = (int(ball_trail[j - 1][0]), int(ball_trail[j - 1][1]))
                curr_center = (int(ball_trail[j][0]), int(ball_trail[j][1]))
                cv2.line(frame, prev_center, curr_center, (0, 255, 255), 2)  # Yellow trail

        else:
            print(f"Warning: Ball position not available for frame {i}")
            ball_mini_court_detections.append(None)
            continue

        # Detect shot type for both players
        shot_type_player_1 = detect_shot_type(player_1_position, ball_position, previous_point_ended)
        shot_type_player_2 = detect_shot_type(player_2_position, ball_position, previous_point_ended)

        # Display shot types on the frame
        cv2.putText(frame, f"Player 1 Shot: {shot_type_player_1}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Player 2 Shot: {shot_type_player_2}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Perform Tactical Analysis for ball and player positions
        if ball_position:
            player_positions = [player_1_position, player_2_position]
            if i in ball_hit_frames:  # Check if the ball hit the ground in this frame
                bounce_position = get_center_of_bbox(ball_position)
                ball_zone = tactical_analysis.analyze_ball_bounce(bounce_position)  # Use ball bounce for zone
                cv2.putText(frame, f"Ball Zone: {ball_zone}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Analyze player positions relative to the net
            player_zone_stats = tactical_analysis.analyze_player_positions(player_positions)

            # Display tactical insights on the frame
            cv2.putText(frame, f"Player Zone Stats: {player_zone_stats}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display rally count on the frame
        cv2.putText(frame, f"Rally: {rally_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_players = {}
        if 'players' in detection and len(detection['players']) >= 2:
            for j, player in enumerate(detection['players'][:2], start=1):
                player_mini_court = mini_court.video_to_court_coordinates(get_center_of_bbox(player['bbox']))
                frame_players[j] = player_mini_court
        player_mini_court_detections.append(frame_players)

        output_video_frames.append(frame)

    # Process player stats
    player_stats = calculate_player_stats(ball_hit_frames, ball_mini_court_detections, player_mini_court_detections, mini_court, video_frames)

    # Draw Mini Court with players and ball for all frames
    output_video_frames = mini_court.draw_mini_court(output_video_frames, detections, interpolated_positions)
    output_video_frames = draw_player_stats(output_video_frames, player_stats)

    # Draw frame number on the top left corner of each frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the processed video with tactical analysis
    save_video(output_video_frames, "./output_vods/output_with_tactical.mp4")

if __name__ == "__main__":
    main()
