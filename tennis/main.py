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
from rally import RallyDetector
from shot_detection_app2.shot_detector import detect_shot_type
from event_detection import EventScoreTracker
from commentary_generator.generator import CommentaryGenerator
from collections import deque

def add_caption_to_frame(frame, caption):
    """Adds a caption to the video frame at the bottom."""
    # Get the height and width of the frame
    frame_height, frame_width = frame.shape[:2]

    # Set the position to the bottom of the frame
    text_position = (50, frame_height - 30)  # 50 pixels from the bottom

    # Draw the caption on the frame
    cv2.putText(frame, caption, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame

def estimate_missing_player_position(player_positions_history, player_index, fallback_position):
    """
    Estimate the missing player position based on previous frames.
    If no reliable position is found, return the fallback_position.
    """
    if len(player_positions_history) == 0:
        # If no history, use the fallback
        return fallback_position

    # Go back through history to find the last available position for this player
    for past_positions in reversed(player_positions_history):
        if past_positions.get(player_index) is not None:
            return past_positions[player_index]

    # If no valid position is found, return fallback
    return fallback_position


def main():
    # Read Video
    input_video_path = "input_vods/new_vod.mp4"
    video_frames = read_video(input_video_path)

    # Initialize the UnifiedTracker for detecting players and ball
    unified_tracker = UnifiedTracker(model_path='./models/player_and_ball_detection/best.pt')

    # Detect players and ball using the unified model
    detections = unified_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/new_vod.pkl")
    print(f"Type of detections: {type(detections)}")
    print(f"Number of frames with detections: {len(detections)}")

    # Interpolate ball positions to handle missed detections
    interpolated_positions = unified_tracker.interpolate_ball_positions(detections)
    ball_hit_frames = unified_tracker.get_ball_shot_frames(detections)

    # Initialize Court Line Detector
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)

    # Get court keypoints from the first frame and initialize MiniCourt
    court_keypoints = court_line_detector.predict(video_frames[0])
    mini_court = MiniCourt(video_frames[0], court_keypoints)

    # Initialize EventScoreTracker
    score_tracker = EventScoreTracker(mini_court)

    # Initialize RallyDetector
    rally_detector = RallyDetector(mini_court)

    # Initialize the commentary generator
    commentary_generator = CommentaryGenerator()
    all_commentary = []  # Collect all commentaries
    caption_queue = deque(maxlen=3)
    MAX_CAPTION_DURATION = 180  # 6 seconds (assuming 30 fps)

    output_video_frames = []
    ball_mini_court_detections = []
    player_mini_court_detections = []  # Fix: Make sure this is updated with players' positions.
    court_keypoints_list = []  # Collect court keypoints for all frames

    previous_point_ended = False
    ball_trail = []
    rally_count = 0
    net_x = None  # Initialize net x-position

    # Variables to track shot types
    previous_shot_type_player_1 = None
    previous_shot_type_player_2 = None

    # History of previous player positions to handle missing detections
    player_positions_history = []

    for i, (frame, detection) in enumerate(zip(video_frames, detections)):
        # Get court keypoints for the current frame
        court_keypoints = court_line_detector.predict(frame)
        court_keypoints_list.append(court_keypoints)

        # Calculate net position (middle of keypoints 0 and 2)
        if net_x is None:
            if len(court_keypoints) >= 4:  # Ensure there are enough points
                net_x = (court_keypoints[0] + court_keypoints[2]) // 2

        # Initialize variables for player positions
        frame_players = {}

        # Process player positions, estimate if missing
        if 'players' in detection and len(detection['players']) >= 2:
            player_1_position = get_center_of_bbox(detection['players'][0]['bbox'])  # First player's position
            player_2_position = get_center_of_bbox(detection['players'][1]['bbox'])  # Second player's position

            # Save current frame positions for history
            frame_players[1] = player_1_position
            frame_players[2] = player_2_position
        else:
            # Estimate missing player positions based on history
            fallback_position_1 = np.array([net_x - 100, court_keypoints[3]])  # Fallback to near baseline
            fallback_position_2 = np.array([net_x + 100, court_keypoints[3]])  # Opponent fallback
            player_1_position = estimate_missing_player_position(player_positions_history, 1, fallback_position_1)
            player_2_position = estimate_missing_player_position(player_positions_history, 2, fallback_position_2)

            # Assign estimated positions
            frame_players[1] = player_1_position
            frame_players[2] = player_2_position

        # Store player positions for history
        player_positions_history.append(frame_players)

        # Draw court keypoints on the frame
        frame = court_line_detector.draw_keypoints(frame, court_keypoints)

        # Draw player and ball detections on the frame
        frame = unified_tracker.draw_bboxes([frame], [detection], interpolated_positions=[interpolated_positions[i]])[0]

        # Initialize foul detection and rally state
        foul_detected = False
        rally_in_progress = False

        # Ensure ball position exists and process it
        ball_position = interpolated_positions[i]
        if ball_position:
            # Convert the ball's position to the mini court coordinates
            ball_mini_court = mini_court.video_to_court_coordinates(get_center_of_bbox(ball_position), court_keypoints)
            ball_mini_court_detections.append(ball_mini_court)

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

            # Determine which player hit the ball based on the ball's position relative to the net
            if ball_position[0] < net_x:
                player_hit = 1  # Player 1 hit the ball
            else:
                player_hit = 2  # Player 2 hit the ball

            # Detect fouls (net hit or out of bounds) only if the rally is in progress
            if rally_in_progress:
                foul_type, foul_player = score_tracker.detect_foul(ball_mini_court, player_hit, net_x, [player_1_position, player_2_position])

                # Only update score and print message when a foul is detected, and avoid repeating the same foul
                if foul_type and not foul_detected:
                    print(f"Foul detected: {foul_type} by Player {foul_player}")
                    score_tracker.update_score(3 - foul_player)  # Award point to the other player

                    # Set the foul_detected flag to True to avoid repeated detections
                    foul_detected = True

                    # Mark the rally as completed
                    rally_in_progress = False

            # Detect the start of a new rally when the ball crosses the net (and no foul is detected)
            if not rally_in_progress:
                if player_hit == 1 and ball_mini_court[0] > net_x:
                    # Player 1 hit and ball crossed the net, starting a new rally
                    rally_in_progress = True
                    foul_detected = False  # Reset foul detection for the new rally
                    score_tracker.reset_foul()  # Reset foul detection state
                elif player_hit == 2 and ball_mini_court[0] < net_x:
                    # Player 2 hit and ball crossed the net, starting a new rally
                    rally_in_progress = True
                    foul_detected = False  # Reset foul detection for the new rally
                    score_tracker.reset_foul()  # Reset foul detection state

            rally_detector.update_rally_count(ball_mini_court)  # Update rally count
            rally_count = rally_detector.get_rally_count()  # Get the current rally count

            # --- Draw the rally count on the frame ---
            cv2.putText(frame, f"Rally Count: {rally_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw the score on the frame at all times
            frame = score_tracker.draw_score_on_frame(frame)

            # --- Collect player positions for stats ---
            # Fix: Ensure player positions are recorded on the mini court.
            player_mini_court_detections.append({
                1: mini_court.video_to_court_coordinates(player_1_position, court_keypoints),
                2: mini_court.video_to_court_coordinates(player_2_position, court_keypoints)
            })

            # --- Generate simplified commentary ---
            shot_type_player_1 = detect_shot_type(player_1_position, ball_position, previous_point_ended)
            shot_type_player_2 = detect_shot_type(player_2_position, ball_position, previous_point_ended)

            if previous_shot_type_player_1 != shot_type_player_1 or previous_shot_type_player_2 != shot_type_player_2:
                commentary = f"Player 1 executes a {shot_type_player_1}, Player 2 executes a {shot_type_player_2}, Rally count: {rally_count}"
                previous_shot_type_player_1 = shot_type_player_1
                previous_shot_type_player_2 = shot_type_player_2

                if commentary:
                    caption_queue.append([commentary, 0])
                    all_commentary.append(commentary)

            # Add commentary captions to the frame
            if caption_queue:
                caption_queue[0][1] += 1  # Increment frame count for current caption
                frame = add_caption_to_frame(frame, caption_queue[0][0])

                # Remove caption after the duration has passed
                if caption_queue[0][1] > MAX_CAPTION_DURATION:
                    caption_queue.popleft()

        else:
            print(f"Warning: Ball position not available for frame {i}")
            ball_mini_court_detections.append(None)

        output_video_frames.append(frame)

    # Process player stats (Fix: Make sure player_mini_court_detections is correctly populated)
    player_stats = calculate_player_stats(ball_hit_frames, ball_mini_court_detections, player_mini_court_detections, mini_court, video_frames)

    # Draw Mini Court with players and ball for all frames
    output_video_frames = mini_court.draw_mini_court(output_video_frames, detections, interpolated_positions, court_keypoints_list)
    output_video_frames = draw_player_stats(output_video_frames, player_stats)

    # Draw frame number on the top left corner of each frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the processed video with tactical analysis and commentary
    save_video(output_video_frames, "./output_vods/new_vod.mp4")

if __name__ == "__main__":
    main()



