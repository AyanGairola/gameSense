# from ultralytics import YOLO


# #TODO: need to install it for everyone
# model8 = YOLO('yolov8x')  

# model5 = YOLO('models/yolo5_last.pt')

# # result = model5.predict('./input_vods/input_video1.mp4', save=True)  

# # print(result)
# # print("boxes: ")

# # for box in result[0].boxes:
# #     print(box)

# result = model8.track('./input_vods/input_video1.mp4', save=True)  

from ultralytics import YOLO
model = YOLO('./models/player_and_ball_detection/best.pt')
result = model.predict('./input_vods/vod4.mp4',conf=0.2, save=True) 