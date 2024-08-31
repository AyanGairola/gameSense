from utils import read_video,save_video
from trackers import PlayerTracker


def main():
    # Read Video
    input_video_path = "./input_vods/vod1.mp4"
    video_frames = read_video(input_video_path)
    
    # Detecting Players
    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames)

    # Save Video
    save_video(video_frames, "./output_vods/output_video.avi")
    
    
if __name__ == "__main__":
    main()
    
