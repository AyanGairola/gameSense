from utils import (
    read_video, save_video, measure_distance, draw_player_stats, 
    convert_pixel_distance_to_meters, get_center_of_bbox, 
)
import constants
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from player_stats import calculate_player_stats, process_player_stats_data
from trackers import UnifiedTracker  # Unified tracker is already imported
import numpy as np
from rally import RallyDetector

def main():
    # Read Video
    input_video_path = "input_vods/vod4.mp4"
    video_frames = read_video(input_video_path)

    unified_tracker = UnifiedTracker(model_path='./models/player_and_ball_detection/best.pt')

    # Detect players and ball using the unified model
    detections = unified_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/unified_detections.pkl")

    # Print the type and length of detections
    print("Type of detections:", type(detections))
    print("Number of frames with detections:", len(detections))
    
    # Print the detections for the first few frames to understand the structure
    for i in range(min(5, len(detections))):
        print(f"Detections for frame {i}:")
        print(detections[i])
        print("Type of detection data for frame:", type(detections[i]))

    # Interpolate ball positions to handle missed detections
    interpolated_positions = unified_tracker.interpolate_ball_positions(detections)
    
    # Print the type and structure of interpolated positions
    print("Type of interpolated_positions:", type(interpolated_positions))
    print("Sample interpolated positions:", interpolated_positions[:5])

    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    
    # MiniCourt Initialization
    mini_court = MiniCourt(video_frames[0])

    # RallyDetector Initialization
    rally_detector = RallyDetector(mini_court)
    rally_count = 0
    net_x = None

    # Process video frame by frame
    output_video_frames = []
    court_keypoints_list = []

    for i, (frame, detection) in enumerate(zip(video_frames, detections)):
        # Detect court keypoints for the current frame
        court_keypoints = court_line_detector.predict(frame)
        court_keypoints_list.append(court_keypoints)

        # Debugging: Print the structure of court_keypoints
        print(f"Court Keypoints for Frame {i}: {court_keypoints}")
        
        # Calculate net position (middle of keypoints 0 and 2)
        if net_x is None:
            if len(court_keypoints) >= 4:  # Ensure there are enough points
                net_x = (court_keypoints[0] + court_keypoints[2]) // 2

        # Draw court keypoints on the frame
        frame = court_line_detector.draw_keypoints(frame, court_keypoints)
        
        # Draw player and ball detections on the frame
        frame = unified_tracker.draw_bboxes([frame], [detection], interpolated_positions=[interpolated_positions[i]])[0]

        # Update rally count if the ball crosses the net
        if net_x is not None:
            ball_position = interpolated_positions[i]
            if ball_position[0] is not None:  # Ensure ball position is valid
                if i > 0:
                    prev_ball_position = interpolated_positions[i-1]
                    # Check if the ball crossed the net
                    if (prev_ball_position[0] < net_x and ball_position[0] >= net_x) or \
                       (prev_ball_position[0] > net_x and ball_position[0] <= net_x):
                        rally_count += 1
        
        # Draw the ball path (trailing effect)
        if i > 0:
            prev_pos = interpolated_positions[i-1]
            curr_pos = interpolated_positions[i]
            if prev_pos and curr_pos and all(prev_pos) and all(curr_pos):
                prev_center = (int((prev_pos[0] + prev_pos[2]) // 2), int((prev_pos[1] + prev_pos[3]) // 2))
                curr_center = (int((curr_pos[0] + curr_pos[2]) // 2), int((curr_pos[1] + curr_pos[3]) // 2))
                cv2.line(frame, prev_center, curr_center, (255, 0, 0), 2)  # Blue line for the ball path
        
        # Draw rally count on the frame
        cv2.putText(frame, f"Rally: {rally_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        output_video_frames.append(frame)

    # Draw Mini Court with players and ball for all frames
    output_video_frames = mini_court.draw_mini_court(output_video_frames, detections, interpolated_positions)

    # Draw frame number on top left corner of each frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the processed video
    save_video(output_video_frames, "./output_vods/oaaa6.mp4")

if __name__ == "__main__":
    main()