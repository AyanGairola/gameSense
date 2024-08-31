from utils import (read_video, save_video)
from trackers import PlayerTracker

def main():

    #Read Video
    input_video_path = "input_vods/input_video1.mp4"
    video_frames = read_video(input_video_path)

    #Detect Players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs/player_detections.pkl")

    #Draw Output

    ##Draw Player Bouding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)


    save_video(video_frames, "output_vods/output_video.mp4")

if __name__ == "__main__":
    main()
