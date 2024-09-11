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
from gemini_commentary import CommentaryGenerator
from collections import deque

import cv2

def add_caption_to_frame(frame, caption, font_scale=1):
    """Adds a caption with a semi-transparent black overlay background to the video frame at the bottom."""
    # Get the height and width of the frame
    frame_height, frame_width = frame.shape[:2]

    # Create an overlay
    overlay = frame.copy()
    overlay_height = 80  # Height of the overlay

    # Draw a black rectangle for the overlay
    cv2.rectangle(overlay, (0, frame_height - overlay_height), (frame_width, frame_height), (0, 0, 0), -1)

    # Set the font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2

    # Get the text size
    text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

    # Calculate text position (centered horizontally, 20 pixels from the bottom)
    text_x = (frame_width - text_size[0]) // 2
    text_y = frame_height - 20

    # Draw the caption text on the overlay
    cv2.putText(overlay, caption, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Combine the overlay with the original frame
    alpha = 0.7  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

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
    input_video_path = "input_vods/input_video1.mp4"
    video_frames = read_video(input_video_path)

    # Initialize the UnifiedTracker for detecting players and ball
    unified_tracker = UnifiedTracker(model_path='./models/player_and_ball_detection/best.pt')

    # Detect players and ball using the unified model
    detections = unified_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/input_video1.pkl")
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
    score_tracker = EventScoreTracker(mini_court, court_keypoints)

    # Initialize RallyDetector
    total_frames = len(video_frames)


    api_key = ""    # Replace with your actual API key
    commentary_generator = CommentaryGenerator(api_key)

    all_commentary = []  # Collect all commentaries
    caption_queue = deque(maxlen=3)
    MAX_CAPTION_DURATION = 144  # 6 seconds (assuming 24 fps)

    output_video_frames = []
    ball_mini_court_detections = []
    player_mini_court_detections = []  # Fix: Make sure this is updated with players' positions.
    court_keypoints_list = []  # Collect court keypoints for all frames

    previous_point_ended = False
    ball_trail = []
    # Initialize RallyDetector
    rally_detector = RallyDetector(mini_court)
    rally_in_progress = False
    foul_detected = False

    # Variables to track shot types
    previous_shot_type_player_1 = None
    previous_shot_type_player_2 = None
    
    previous_ball_position = None
    two_frames_ago_position = None
    bounce_positions = []  # To store bounce positions
    
    distance_threshold = 10
    # History of previous player positions to handle missing detections
    player_positions_history = []

    for i, (frame, detection) in enumerate(zip(video_frames, detections)):
        # Get court keypoints for the current frame
        court_keypoints = court_line_detector.predict(frame)

        # Ensure there are enough keypoints (you need at least 10 keypoints for this check)
        if court_keypoints is None or len(court_keypoints) <= 14:
            print(f"Skipping court keypoints drawing on frame {i} due to insufficient keypoints.")
            
            # Reset rally if keypoints are not available or insufficient
            rally_detector.reset_rally()
            
            print(f"Rally reset called on frame {i}. Current rally count: {rally_detector.get_rally_count()}")
            
            # Append None for this frame to indicate no keypoints
            court_keypoints_list.append(None)
            
            # Skip further processing for this frame
            continue

        # Extract the keypoints
        p0 = np.array([court_keypoints[0], court_keypoints[1]])  # Keypoint 0
        p4 = np.array([court_keypoints[8], court_keypoints[9]])  # Keypoint 4
        p6 = np.array([court_keypoints[12], court_keypoints[13]]) # Keypoint 6
        p9 = np.array([court_keypoints[18], court_keypoints[19]]) # Keypoint 9
        p8 = np.array([court_keypoints[16], court_keypoints[17]]) # Keypoint 8
        p1 = np.array([court_keypoints[2], court_keypoints[3]])   # Keypoint 1

        # Calculate the distances between the specified keypoints
        distance_0_4 = np.linalg.norm(p0 - p4)
        distance_6_9 = np.linalg.norm(p6 - p9)
        distance_4_8 = np.linalg.norm(p4 - p8)
        distance_6_1 = np.linalg.norm(p6 - p1)

        # Check if any of the distances are below the threshold
        draw_keypoints = True
        if (distance_0_4 < distance_threshold or
            distance_6_9 < distance_threshold or
            distance_4_8 < distance_threshold or
            distance_6_1 < distance_threshold):
            draw_keypoints = False  # Don't draw the keypoints

        # Draw the keypoints if they pass the distance check
        if draw_keypoints:
            frame = court_line_detector.draw_keypoints(frame, court_keypoints)
            
            # DRAWING NET
            padding_net = 20
            # Calculate left net point
            left_net_x = ((court_keypoints[16] + court_keypoints[20]) / 2) 
            left_net_y = ((court_keypoints[17] + court_keypoints[21]) / 2) - padding_net

            # Calculate right net point
            right_net_x = ((court_keypoints[18] + court_keypoints[22]) / 2) 
            right_net_y = ((court_keypoints[19] + court_keypoints[23]) / 2) - padding_net

            # Draw the net as a line between the left and right net points
            cv2.line(frame, (int(left_net_x), int(left_net_y)), (int(right_net_x), int(right_net_y)), (255, 0, 0), 2) 
            
            
            # DRAWING COURT GRID
            p4_x, p4_y = court_keypoints[8], court_keypoints[9]
            p5_x, p5_y = court_keypoints[10], court_keypoints[11]  
            p6_x, p6_y = court_keypoints[12], court_keypoints[13]  
            p7_x, p7_y = court_keypoints[14], court_keypoints[15]  
            
            p8_x, p8_y = court_keypoints[16], court_keypoints[17]  
            p9_x, p9_y = court_keypoints[18], court_keypoints[19]  
            p10_x, p10_y = court_keypoints[20], court_keypoints[21]  
            p11_x, p11_y = court_keypoints[22], court_keypoints[23]  
            p12_x, p12_y = court_keypoints[24], court_keypoints[25]  
            p13_x, p13_y = court_keypoints[26], court_keypoints[27]  

            # BOUNDARY
            cv2.line(frame, (int(p4_x), int(p4_y)), (int(p5_x), int(p5_y)), (0, 0, 0), 2)   
            cv2.line(frame, (int(p6_x), int(p6_y)), (int(p7_x), int(p7_y)), (0, 0, 0), 2)  
            cv2.line(frame, (int(p4_x), int(p4_y)), (int(p6_x), int(p6_y)), (0, 0, 0), 2)  
            cv2.line(frame, (int(p5_x), int(p5_y)), (int(p7_x), int(p7_y)), (0, 0, 0), 2)  
            
            # INNER LINES
            
            cv2.line(frame, (int(p8_x), int(p8_y)), (int(p12_x), int(p12_y)), (0, 0, 0), 2)   
            cv2.line(frame, (int(p8_x), int(p8_y)), (int(p10_x), int(p10_y)), (0, 0, 0), 2)   
            cv2.line(frame, (int(p12_x), int(p12_y)), (int(p9_x), int(p9_y)), (0, 0, 0), 2)   
            cv2.line(frame, (int(p12_x), int(p12_y)), (int(p13_x), int(p13_y)), (0, 0, 0), 2)   
            cv2.line(frame, (int(p11_x), int(p11_y)), (int(p13_x), int(p13_y)), (0, 0, 0), 2)   
            cv2.line(frame, (int(p10_x), int(p10_y)), (int(p13_x), int(p13_y)), (0, 0, 0), 2)   
            cv2.line(frame, (int(p9_x), int(p9_y)), (int(p11_x), int(p11_y)), (0, 0, 0), 2) 
        else:
            print(f"Not displaying court keypoints on frame {i} due to close keypoints.")

        # Append the court keypoints or None based on whether we are drawing them
        court_keypoints_list.append(court_keypoints if draw_keypoints else None)
        net_x = ((court_keypoints[16] + court_keypoints[20]) / 2)
        
        

        if i in ball_hit_frames:
            cv2.putText(frame, "BALL HIT", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        

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


        # Draw player and ball detections on the frame
        frame = unified_tracker.draw_bboxes([frame], [detection], interpolated_positions=[interpolated_positions[i]])[0]

        # Initialize foul detection and rally state
        foul_detected = False
        rally_in_progress = False

        # Ensure ball position exists and process it
        
        ball_position = interpolated_positions[i]
        if ball_position:
            # Convert the ball's position to mini court coordinates
            ball_mini_court = mini_court.video_to_court_coordinates(get_center_of_bbox(ball_position), court_keypoints)
            ball_mini_court_detections.append(ball_mini_court)
            ball_center = get_center_of_bbox(ball_position)

            # Add current ball position to the ball trail
            ball_trail.append(get_center_of_bbox(ball_position))
            if len(ball_trail) > 2:  # Keep the ball trail within the last 10 positions
                ball_trail.pop(0)

            # Draw the ball trail on the frame
            for j in range(1, len(ball_trail)):
                if ball_trail[j - 1] is None or ball_trail[j] is None:
                    continue
                prev_center = (int(ball_trail[j - 1][0]), int(ball_trail[j - 1][1]))
                curr_center = (int(ball_trail[j][0]), int(ball_trail[j][1]))
                cv2.line(frame, prev_center, curr_center, (0, 255, 255), 2)  # Yellow trail
                
            
            


                
            player_positions = [player_1_position, player_2_position]
            
            shot_type_player_1 = detect_shot_type(player_1_position, ball_position, previous_point_ended)

            # Detect shot type for player 2
            shot_type_player_2 = detect_shot_type(player_2_position, ball_position, previous_point_ended)

            # Display the detected shot types on the screen
            cv2.putText(frame, f"Shot Predicted for P1: {shot_type_player_1}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(frame, f"Shot Predicted for P2: {shot_type_player_2}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            
    
            
            is_bounce, bounce_position = score_tracker.track_bounces(ball_trail, ball_position, player_positions, ball_hit_frames=ball_hit_frames)

            # if is_bounce:
            #     # Handle the bounce (e.g., mark it on the frame)
            #     print(f"BOUNCE DETECTED AT FRAME {i}")


             # Update and draw rally count
            rally_detector.update_rally_count(ball_mini_court)  # Update rally count
            rally_count = rally_detector.get_rally_count()  # Get the current rally count
            cv2.putText(frame, f"Rally Count: {rally_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            
            
            # Determine which player hit the ball based on its position relative to the net
            player_hit = 1 if ball_position[0] < net_x else 2

            # Detect fouls (out of bounds, ball passed player)
            foul_type, winner = score_tracker.detect_foul(
                ball_position=ball_position, 
                player_hit=player_hit, 
                net_x=net_x, 
                player_positions=player_positions, 
                ball_trail=ball_trail, 
                ball_hit_frames=ball_hit_frames, 
                current_frame=i, 
                rally_in_progress=rally_in_progress
            )

            # Handle fouls if any are detected
            if foul_type and not foul_detected:
                print(f"Foul detected: {foul_type} by Player {player_hit} at frame {i}")
                
                # Update the score and reset the rally
                score_tracker.update_score(winner)  # Award point to the other player
                
                foul_detected = True  # Mark that a foul has been detected
                rally_in_progress = False  # End rally since a foul was detected
                
                # Reset foul state for the next rally
                score_tracker.reset_foul_state()

                # Reset for the next rally (after the foul)
                rally_in_progress = True  # Ready for the next rally
                foul_detected = False  # Reset foul detection for the next rally
            else:
                # If no foul, reset the foul state at the end of a valid rally
                score_tracker.reset_foul_state()

            if i < 60 or i >= total_frames - 60:
                # Draw the score on the frame only in the first 60 or last 60 frames
                frame = score_tracker.draw_score_on_frame(frame)


            # --- Collect player positions for stats ---
            # Fix: Ensure player positions are recorded on the mini court.
            player_mini_court_detections.append({
                1: mini_court.video_to_court_coordinates(player_1_position, court_keypoints),
                2: mini_court.video_to_court_coordinates(player_2_position, court_keypoints)
            })

            # # --- Generate simplified commentary ---
            shot_type_player_1 = detect_shot_type(player_1_position, ball_position, previous_point_ended)
            shot_type_player_2 = detect_shot_type(player_2_position, ball_position, previous_point_ended)


            # Generate commentary using the Gemini-powered generator
            if i % 5 == 0:  # Generate commentary every 5 frames to avoid hitting rate limits
                # player_stats_dict = {player: stats.to_dict() for player, stats in player_stats.iterrows()} if player_stats is not None else {}
                commentary = commentary_generator.generate_frame_commentary(
                    ball_hit_frames,
                    (shot_type_player_1, shot_type_player_2),
                    rally_count,
                    # player_stats_dict,
                    i
                )
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
            
        # frame = score_tracker.draw_score_on_frame(frame)

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
    save_video(output_video_frames, "./output_vods/op_vd1.mp4")

if __name__ == "__main__":
    main()






