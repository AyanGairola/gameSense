from ultralytics import YOLO
model = YOLO('./models/court-and-net-detection/new_last.pt')
result = model.predict('./input_vods/vod2.mp4',conf=0.2, save=True) 
