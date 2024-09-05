from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class UnifiedTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

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

            if cls == 0:  # Assuming class 0 is the ball
                detections['ball'].append(detection)
            else:  # Assuming other classes are players
                detections['players'].append(detection)
                
        return detections

    def draw_bboxes(self, video_frames, detections, interpolated_positions=None):
        output_video_frames = []
        for i, (frame, detection) in enumerate(zip(video_frames, detections)):
            for det in detection['players']:
                x1, y1, x2, y2 = det['bbox']
                cls = det['class']
                color = (0, 255, 0)  # Green for players
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = 'Player'
                cv2.putText(frame, f"{label}: {det['confidence']:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            if interpolated_positions:
                prev_pos = interpolated_positions[i-1] if i > 0 else None
                curr_pos = interpolated_positions[i]
                if prev_pos and curr_pos and all(prev_pos) and all(curr_pos):
                    prev_center = (int((prev_pos[0] + prev_pos[2]) // 2), int((prev_pos[1] + prev_pos[3]) // 2))
                    curr_center = (int((curr_pos[0] + curr_pos[2]) // 2), int((curr_pos[1] + curr_pos[3]) // 2))
                    cv2.line(frame, prev_center, curr_center, (0, 255, 255), 2)  # Greenish-yellow line for the ball path
            
            output_video_frames.append(frame)
        return output_video_frames

    def interpolate_ball_positions(self, detections):
        ball_positions = [self.extract_ball_position(det) for det in detections]

        # Create DataFrame with possible NaN for interpolation
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values (linear interpolation by default)
        df_ball_positions = df_ball_positions.interpolate(method='linear')

        # Forward fill
        df_ball_positions = df_ball_positions.ffill()

        # Backward fill
        df_ball_positions = df_ball_positions.bfill()

        # Convert DataFrame to list of positions
        interpolated_positions = df_ball_positions.to_numpy().tolist()

        return interpolated_positions

    def extract_ball_position(self, detection):
        if detection['ball']:
            # Taking the first detected ball if multiple are detected
            return detection['ball'][0]['bbox']
        return [None, None, None, None]  # Return a placeholder if no ball is found

    def get_ball_shot_frames(self, interpolated_positions):
        # Extract ball positions ensuring each entry has 4 values
        ball_positions = interpolated_positions

        # Check if all positions are None or if the list is empty
        if not ball_positions or all(pos == [None, None, None, None] for pos in ball_positions):
            print("No valid ball positions found. Exiting the function.")
            return []  # Return an empty list if no valid ball positions are found

        # Create DataFrame from ball positions
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # If the DataFrame is empty after filtering, handle it
        if df_ball_positions.empty:
            print("DataFrame is empty after filtering ball positions. Exiting the function.")
            return []

        # Process DataFrame to detect ball shots
        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        minimum_change_frames_for_hit = 25

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits
    

    def get_ball_shot_frames_w_o_interpolations(self, ball_positions):
        # Convert ball positions to a pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Check if DataFrame is empty
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

        # Parameters for detecting ball hit
        minimum_change_frames_for_hit = 25

        # Iterate through frames and detect significant changes in delta_y
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_following = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_following = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_following:
                        change_count += 1
                    elif positive_position_change and positive_following:
                        change_count += 1

                # Mark frame as ball hit if significant change is detected
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        # Extract frames where ball hits are detected
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        return frame_nums_with_ball_hits