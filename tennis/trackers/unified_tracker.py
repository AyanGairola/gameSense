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
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'class': cls,
                'confidence': confidence
            })
        return detections

    def draw_bboxes(self, video_frames, detections):
        output_video_frames = []
        for frame, detection in zip(video_frames, detections):
            for det in detection:
                x1, y1, x2, y2 = det['bbox']
                cls = det['class']
                color = (0, 255, 0) if cls == 0 else (0, 0, 255)  # Assuming class 0 is ball and 1 is player
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = 'Ball' if cls == 0 else 'Player'
                cv2.putText(frame, f"{label}: {det['confidence']:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            output_video_frames.append(frame)
        return output_video_frames

    def get_ball_shot_frames(self, detections):
        ball_positions = [self.extract_ball_position(det) for det in detections]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

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

    def extract_ball_position(self, detection):
        for det in detection:
            if det['class'] == 0:  # Assuming class 0 is the ball
                return det['bbox']
        return [None, None, None, None]  # Return a placeholder if no ball is found