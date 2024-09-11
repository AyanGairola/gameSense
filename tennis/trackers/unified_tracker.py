import numpy as np
import cv2
import pickle
import pandas as pd
from ultralytics import YOLO
from scipy.spatial import distance


class UnifiedTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.previous_player_positions = [None, None]  

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        all_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                all_detections = pickle.load(f)
            return all_detections

        for frame in frames:
            detections = self.detect_frame(frame)
            all_detections.append(detections)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(all_detections, f)

        return all_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        detections = {'ball': [], 'players': []}
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            cls = int(box.cls[0])
            confidence = float(box.conf[0])

            detection = {
                'bbox': (x1, y1, x2, y2),
                'class': cls,
                'confidence': confidence
            }

            if cls == 0:  
                detections['ball'].append(detection)
            else:  
                detections['players'].append(detection)
                
        return detections

    def draw_bboxes(self, video_frames, detections, interpolated_positions=None):
        output_video_frames = []
        for i, (frame, detection) in enumerate(zip(video_frames, detections)):
            players = detection['players']

            if len(players) >= 2:
                # Track players based on previous positions
                player_1, player_2 = self.track_players(players)
            else:
                output_video_frames.append(frame)
                continue

            self.draw_player_bbox(frame, player_1, player_label="Player 1")

            self.draw_player_bbox(frame, player_2, player_label="Player 2")

            # Draw ball path if interpolated positions are available
            if interpolated_positions:
                prev_pos = interpolated_positions[i-1] if i > 0 else None
                curr_pos = interpolated_positions[i]
                if prev_pos and curr_pos and all(prev_pos) and all(curr_pos):
                    prev_center = (int((prev_pos[0] + prev_pos[2]) // 2), int((prev_pos[1] + prev_pos[3]) // 2))
                    curr_center = (int((curr_pos[0] + curr_pos[2]) // 2), int((curr_pos[1] + curr_pos[3]) // 2))
                    cv2.line(frame, prev_center, curr_center, (0, 255, 255), 2)  # Yellow line for ball path
            
            output_video_frames.append(frame)
        return output_video_frames

    def draw_player_bbox(self, frame, player, player_label):
        x1, y1, x2, y2 = map(int, player['bbox'])
        color = (0, 255, 0)  # Green for players
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{player_label}: {player['confidence']:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def track_players(self, players):
        
        # Track players across frames using Euclidean distance to previous positions.

        if self.previous_player_positions[0] is None or len(players) < 2:
            # First frame: assign players based on horizontal position (left to right)
            players = sorted(players, key=lambda p: p['bbox'][0])
            self.previous_player_positions = [players[0]['bbox'], players[1]['bbox']]
            return players[0], players[1]
        
        current_centers = [self.get_center(player['bbox']) for player in players]

        # Calculate distances between previous players and current players
        prev_centers = [self.get_center(self.previous_player_positions[0]),
                        self.get_center(self.previous_player_positions[1])]

        # Assign players based on minimum distance
        distances = distance.cdist(prev_centers, current_centers, 'euclidean')
        min_indices = distances.argmin(axis=1)

        # Update previous positions
        self.previous_player_positions[0] = players[min_indices[0]]['bbox']
        self.previous_player_positions[1] = players[min_indices[1]]['bbox']

        return players[min_indices[0]], players[min_indices[1]]

    @staticmethod
    def get_center(bbox):
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def interpolate_ball_positions(self, detections, max_gap=5):        
        # Interpolates ball positions to handle missing values with a moderate level of robustness.
        # Interpolates for small gaps and forward/backward fills larger gaps.
        
        # Extract ball positions
        ball_positions = [self.extract_ball_position(det) for det in detections]
        
        # Create DataFrame for interpolation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Mark rows where no ball was detected
        df_ball_positions['missing'] = df_ball_positions.isna().any(axis=1)

        # Identify groups of consecutive frames with missing data
        df_ball_positions['group'] = (~df_ball_positions['missing']).cumsum()
        gaps = df_ball_positions['group'].value_counts().sort_index()

        # Apply linear interpolation for gaps smaller than the max_gap
        for group, size in gaps.items():
            if size <= max_gap:
                df_ball_positions.loc[df_ball_positions['group'] == group, ['x1', 'y1', 'x2', 'y2']] = \
                    df_ball_positions.loc[df_ball_positions['group'] == group, ['x1', 'y1', 'x2', 'y2']].interpolate()

        # Forward fill and backward fill for the rest
        df_ball_positions[['x1', 'y1', 'x2', 'y2']] = df_ball_positions[['x1', 'y1', 'x2', 'y2']].ffill().bfill()

        # Convert DataFrame to list of positions
        interpolated_positions = df_ball_positions[['x1', 'y1', 'x2', 'y2']].to_numpy().tolist()

        return interpolated_positions



    def extract_ball_position(self, detection):
        if detection['ball']:
            # Take the first detected ball if multiple are detected
            return detection['ball'][0]['bbox']
        return [None, None, None, None]  # Return a placeholder if no ball is found

    def get_ball_shot_frames(self, detections):
        ball_positions = []
        for frame, detection in enumerate(detections):
            if detection['ball']:
                x1, y1, x2, y2 = detection['ball'][0]['bbox']
                ball_positions.append([frame, (x1 + x2) / 2, (y1 + y2) / 2])
            else:
                ball_positions.append([frame, None, None])

        df_ball_positions = pd.DataFrame(ball_positions, columns=['frame', 'x', 'y'])
        df_ball_positions['y'] = df_ball_positions['y'].interpolate()
        df_ball_positions['y_rolling_mean'] = df_ball_positions['y'].rolling(window=5, min_periods=1, center=True).mean()
        df_ball_positions['delta_y'] = df_ball_positions['y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25
        frame_nums_with_ball_hits = []

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i+1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i+1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    frame_nums_with_ball_hits.append(df_ball_positions['frame'].iloc[i])

        return frame_nums_with_ball_hits

    def display_ball_hits(self, video_frames, detections):
        ball_hit_frames = self.get_ball_shot_frames(detections)
        
        output_video_frames = []
        for i, (frame, detection) in enumerate(zip(video_frames, detections)):

            if i in ball_hit_frames:
                cv2.putText(frame, "BALL HIT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames, ball_hit_frames
    
    def get_ball_shot_frames_w_o_interpolations(self, ball_positions):
        
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        if df_ball_positions.empty:
            print("No valid ball positions found.")
            return []

        # Initialize 'ball_hit' column and calculate the midpoint y-coordinates
        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2

        # Rolling mean of mid_y to smooth the data
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1).mean()

        # Calculate the change (delta) in mid_y_rolling_mean between consecutive frames
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        # Parameters for detecting ball hits
        minimum_change_frames_for_hit = 25

        # Iterate through frames and detect significant changes in delta_y
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            # If we detect a significant change in position, we consider it as a ball hit
            if negative_position_change or positive_position_change:
                change_count = 0

                # Ensure the change persists across a number of frames
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_following_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_following_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_following_change:
                        change_count += 1
                    elif positive_position_change and positive_following_change:
                        change_count += 1

                # If the change is significant over a range of frames, mark as a ball hit
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        # Extract frames where ball hits are detected
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits