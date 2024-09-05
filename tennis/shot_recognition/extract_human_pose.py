from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2
import os
from ultralytics import YOLO
from trackers import UnifiedTracker  # Import the UnifiedTracker class

# Dynamically get the base directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use the base directory to construct the path to the movenet model
movenet_model_path = os.path.join(BASE_DIR, "../models/movenet.tflite")

try:
    interpreter = tf.lite.Interpreter(model_path=movenet_model_path)
    interpreter.allocate_tensors()
    print("MoveNet model loaded successfully")
except Exception as e:
    print(f"Failed to load MoveNet model: {e}")

class RoI:
    """
    Define the Region of Interest (RoI) around the tennis player.
    """
    def __init__(self, shape):
        self.frame_width = shape[1]
        self.frame_height = shape[0]
        self.width = self.frame_width
        self.height = self.frame_height
        self.center_x = shape[1] // 2
        self.center_y = shape[0] // 2
        self.valid = False

    def extract_subframe(self, frame):
        """Extract the RoI from the original frame."""
        subframe = frame.copy()
        return subframe[
            self.center_y - self.height // 2 : self.center_y + self.height // 2,
            self.center_x - self.width // 2 : self.center_x + self.width // 2,
        ]

    def transform_to_subframe_coordinates(self, keypoints_from_tf):
        """Convert keypoints from normalized coordinates to subframe pixel coordinates."""
        return np.squeeze(np.multiply(keypoints_from_tf, [self.width, self.height, 1]))

    def transform_to_frame_coordinates(self, keypoints_from_tf):
        """Convert keypoints from subframe coordinates to full-frame coordinates."""
        keypoints_pixels_subframe = self.transform_to_subframe_coordinates(keypoints_from_tf)
        keypoints_pixels_frame = keypoints_pixels_subframe.copy()
        
        keypoints_pixels_frame[:, 0] += self.center_x - self.width // 2
        keypoints_pixels_frame[:, 1] += self.center_y - self.height // 2

        return keypoints_pixels_frame

    def update_from_bbox(self, bbox):
        """Update the RoI using bounding box from player detection model."""
        x_min, y_min, x_max, y_max = bbox
        self.center_x = (x_min + x_max) // 2
        self.center_y = (y_min + y_max) // 2
        self.width = int((x_max - x_min) * 1.3)
        self.height = int((y_max - y_min) * 1.3)
        self.valid = True

    def reset(self):
        """Reset the RoI to the full image dimensions."""
        self.width = self.frame_width
        self.height = self.frame_height
        self.center_x = self.frame_width // 2
        self.center_y = self.frame_height // 2
        self.valid = False


class HumanPoseExtractor:
    """
    Defines mapping between movenet key points and human readable body points
    with realistic edges to be drawn.
    """
    EDGES = { ... }  # Define edges similar to the original script
    COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

    def __init__(self, shape):
        # Initialize the TFLite interpreter for MoveNet
        self.interpreter = tf.lite.Interpreter(model_path=movenet_model_path)
        self.interpreter.allocate_tensors()
        self.roi = RoI(shape)

    def extract(self, frame):
        """Run inference model on the RoI from player detection"""
        # Extract the RoI based on the bounding box from the player detector
        subframe = self.roi.extract_subframe(frame)

        img = tf.image.resize_with_pad(np.expand_dims(subframe, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.uint8)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        self.interpreter.invoke()

        self.keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])
        self.keypoints_pixels_frame = self.roi.transform_to_frame_coordinates(self.keypoints_with_scores)

    def discard(self, list_of_keypoints):
        """Discard unnecessary keypoints like eyes or ears"""
        for keypoint in list_of_keypoints:
            self.keypoints_with_scores[0, 0, self.KEYPOINT_DICT[keypoint], 2] = 0
            self.keypoints_pixels_frame[self.KEYPOINT_DICT[keypoint], 2] = 0

def main():
    parser = ArgumentParser(description="Display human pose and track player in a video")
    parser.add_argument("video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()

    ret, frame = cap.read()
    human_pose_extractor = HumanPoseExtractor(frame.shape)
    unified_tracker = UnifiedTracker(model_path="../models/player_and_ball_detection/best.pt")  # Initialize the tracker

    FRAME_ID = 0
    video_frames = []

    # Read video frames for batch processing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(frame)

    # Use the tracker to detect players and ball in each frame
    detections = unified_tracker.detect_frames(video_frames, read_from_stub=False)

    for i, frame in enumerate(video_frames):
        FRAME_ID += 1

        detection = detections[i]

        # Use the first player's bounding box to update the RoI
        if detection['players']:
            first_player_bbox = detection['players'][0]['bbox']
            human_pose_extractor.roi.update_from_bbox(first_player_bbox)

        # Extract human pose based on the new RoI
        human_pose_extractor.extract(frame)

        # Optionally discard irrelevant keypoints
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        # Draw the results (pose keypoints and edges) on the frame
        human_pose_extractor.draw_results_frame(frame)

        # Optionally draw player and ball detections
        frame = unified_tracker.draw_bboxes([frame], [detection])[0]
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
