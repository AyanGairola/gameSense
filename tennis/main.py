from utils import (read_video, save_video, measure_distance, draw_player_stats, convert_pixel_distance_to_meters)
import constants
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from player_stats import calculate_player_stats, process_player_stats_data
from trackers import UnifiedTracker  # Import your new unified tracker

def main():
    # Read Video
    input_video_path = "input_vods/vod4.mp4"
    video_frames = read_video(input_video_path)

    # Initialize the unified tracker
    unified_tracker = UnifiedTracker(model_path='./models/player_and_ball_detection/best.pt')

    # Detect players and ball using the new unified model
    detections = unified_tracker.detect_frames(video_frames)

    # Interpolate ball positions to handle missed detections
    interpolated_positions = unified_tracker.interpolate_ball_positions(detections)

    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)

    # Process video frame by frame
    output_video_frames = []
    for i, (frame, detection) in enumerate(zip(video_frames, detections)):
        # Detect court keypoints for the current frame
        court_keypoints = court_line_detector.predict(frame)

        # Draw court Keypoints on the frame
        frame = court_line_detector.draw_keypoints(frame, court_keypoints)
        
        # Draw player and ball detections on the frame
        frame = unified_tracker.draw_bboxes([frame], [detection], interpolated_positions=[interpolated_positions[i]])[0]
        
        # Draw the ball path (trailing effect)
        if i > 0:
            prev_pos = interpolated_positions[i-1]
            curr_pos = interpolated_positions[i]
            if prev_pos and curr_pos and all(prev_pos) and all(curr_pos):
                prev_center = (int((prev_pos[0] + prev_pos[2]) // 2), int((prev_pos[1] + prev_pos[3]) // 2))
                curr_center = (int((curr_pos[0] + curr_pos[2]) // 2), int((curr_pos[1] + curr_pos[3]) // 2))
                cv2.line(frame, prev_center, curr_center, (0, 255, 255), 2)  # Greenish-yellow line for the ball path
        
        output_video_frames.append(frame)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        detections, 
        detections,  # Assuming you have players and ball detections in the same structure
        court_keypoints
    )

    # Calculate player stats
    player_stats_data_df = calculate_player_stats(
        [],  # No ball_shot_frames passed since it's not used now
        ball_mini_court_detections, 
        player_mini_court_detections, 
        mini_court, 
        constants
    )

    player_stats_data_df = process_player_stats_data(player_stats_data_df, video_frames)

    # Draw Player Stats on the video frames
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df)

    # Draw frame number on top left corner of each frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the processed video
    save_video(output_video_frames, "./output_vods/oaaa.mp4")

if __name__ == "__main__":
    main()