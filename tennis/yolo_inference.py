from ultralytics import YOLO

model = YOLO('yolov8x')  # Load the YOLOv8 model
model.predict('./input_vods/image1.png', save=True)  # Predict and save results