from inference_sdk import InferenceHTTPClient
import cv2 as cv
import os

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="bpAIy1OT3GHg1p7gWwcm"
)

def infer_frame(image_path, client, model_id):
    # Send the image file path to the model
    result = client.infer(image_path, model_id=model_id)
    
    # Load the image to draw results
    image = cv.imread(image_path)
    
    # Process the result
    if result['predictions']:
        for prediction in result['predictions']:
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            cv.rectangle(image, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
            cv.putText(image, prediction['class'], (int(x - w / 2), int(y - h / 2) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def process_video(input_video_path, output_video_path, client, model_id):
    cap = cv.VideoCapture(input_video_path)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame to a temporary image file
        temp_image_path = "temp_frame.jpg"
        cv.imwrite(temp_image_path, frame)
        
        # Perform inference on the frame using the file path
        processed_frame = infer_frame(temp_image_path, client, model_id)
        
        # Initialize the video writer once we know the frame size
        if out is None:
            out = cv.VideoWriter(output_video_path, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
        
        out.write(processed_frame)
    
    cap.release()
    out.release()
    
    # Clean up the temporary file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    
    print(f"Processed video saved to {output_video_path}")

# Paths
input_video = './input_vods/vod1.mp4'
output_video = './output_vods/vod1_processed.mp4'
model_id = "segtest-lfsgo/1"  # Replace with your specific model ID

# Process the video using the Roboflow model
process_video(input_video, output_video, CLIENT, model_id)