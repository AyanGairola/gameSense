import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    # Generate unique output video path by incrementing the number if file exists
    output_video_path = generate_unique_filename(output_video_path)

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    # Write each frame to the video
    for frame in output_video_frames:
        out.write(frame)

    # Release the VideoWriter
    out.release()

def generate_unique_filename(output_video_path):
    base_path, ext = os.path.splitext(output_video_path)
    counter = 1

    # Check if the file exists, and increment the number in the filename
    while os.path.exists(output_video_path):
        output_video_path = f"{base_path}{counter}{ext}"
        counter += 1

    return output_video_path
