from ultralytics import YOLO

# Load a model


model = YOLO("./models/yolov8x-pose-p6.pt")  # load an official model

# Predict with the model
result = model.predict('./input_vods/image1.png',conf=0.01, save=True) 