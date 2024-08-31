import cv2 # type: ignore

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames

def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("No frames to save.")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    if not out.isOpened():
        raise ValueError(f"Error opening output video file: {output_video_path}")
    try:
        for frame in output_video_frames:
            out.write(frame)
    finally:
        out.release()