import time
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from shot_recognition.extract_human_pose import HumanPoseExtractor


# Initialize GPU if available
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Num GPUs Available: ", len(physical_devices))


class ShotCounter:
    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_history = 30
        self.probs = np.zeros(4)

        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0

        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

        self.results = []

    def update(self, probs, frame_id):
        """Update current state with shot probabilities"""

        if len(probs) == 4:
            self.probs = probs
        else:
            self.probs[0:3] = probs

        if (
            probs[0] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            probs[1] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            len(probs) > 3
            and probs[3] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

        self.frames_since_last_shot += 1

    def display(self, frame):
        """Display shot counter"""
        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (20, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "backhand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (20, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "forehand" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Serves = {self.nb_serves}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0)
            if (self.last_shot == "serve" and self.frames_since_last_shot < 30)
            else (0, 0, 255),
            thickness=2,
        )


def draw_probs(frame, probs):
    """Draw vertical bars representing probabilities"""
    cv2.putText(
        frame, "S", (1075, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
    )
    cv2.putText(
        frame, "B", (1130, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
    )
    cv2.putText(
        frame, "N", (1185, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
    )
    cv2.putText(
        frame, "F", (1240, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
    )

    # Draw vertical bars representing probabilities
    bars_x = [1070, 1130, 1190, 1250]
    for i, prob in enumerate(probs):
        cv2.rectangle(
            frame,
            (bars_x[i], 230),
            (bars_x[i] + 30, int(230 - 170 * prob)),
            color=(0, 0, 255),
            thickness=-1,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Track tennis player and display shot probabilities")
    parser.add_argument("video")
    parser.add_argument("model")
    parser.add_argument("--evaluate", help="Path to annotation file")
    parser.add_argument("-f", type=int, help="Forward to")
    parser.add_argument("--left-handed", action="store_const", const=True, default=False, help="If player is left-handed")
    args = parser.parse_args()

    shot_counter = ShotCounter()

    if args.evaluate is not None:
        gt = GT(args.evaluate)

    # Load the RNN model
    model = keras.models.load_model(args.model)

    # Open video capture
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened()

    ret, frame = cap.read()
    human_pose_extractor = HumanPoseExtractor(frame.shape)

    NB_IMAGES = 30
    FRAME_ID = 0
    features_pool = []
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        FRAME_ID += 1
        if args.f is not None and FRAME_ID < args.f:
            continue

        human_pose_extractor.extract(frame)
        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

        # If the player is left-handed, adjust the features
        if args.left_handed:
            features[:, 1] = 1 - features[:, 1]

        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 26)
        features_pool.append(features)

        # Sliding window
        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            probs = model.predict(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)
            features_pool = features_pool[1:]

        draw_probs(frame, shot_counter.probs)
        shot_counter.display(frame)

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        draw_fps(frame, fps)
        draw_frame_id(frame, FRAME_ID)

        # Write frame to file or show
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
