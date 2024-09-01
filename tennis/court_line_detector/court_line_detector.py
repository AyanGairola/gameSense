import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        
        self.model.eval()  # Ensure the model is in evaluation mode
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        
        # Use float32 precision
        image_tensor = image_tensor.float()
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] = np.clip(keypoints[::2] * (original_w / 224.0), 0, original_w)
        keypoints[1::2] = np.clip(keypoints[1::2] * (original_h / 224.0), 0, original_h)

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Plot keypoints on the image
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image
    
    def process_video(self, video_frames):
        output_video_frames = []
        for frame in video_frames:
            # Detect keypoints for the current frame
            keypoints = self.predict(frame)
            
            # Draw the detected keypoints on the frame
            frame_with_keypoints = self.draw_keypoints(frame, keypoints)
            
            # Store the processed frame
            output_video_frames.append(frame_with_keypoints)
        
        return output_video_frames