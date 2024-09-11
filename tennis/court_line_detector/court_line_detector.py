import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np

class CourtLineDetector:
    def __init__(self, model_path):
        # Load pre-trained weights for ResNet50 model on ImageNet data
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        # Change the final fully connected layer to predict 28 (14 keypoints with x and y coordinates)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()  # Ensure the model is in evaluation mode
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize image to 224x224 (input size for ResNet50)
            transforms.ToTensor(),
            # Normalize with ImageNet mean and std (important for pre-trained models)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.reference_keypoints = None
        self.previous_keypoints = None
        self.image_shape = None

    def update_reference_keypoints(self, keypoints, update_threshold=0.1):
        if self.reference_keypoints is None:
            self.reference_keypoints = keypoints.reshape(-1, 2)
        else:
            # Calculate the average movement of keypoints
            movement = np.mean(np.abs(keypoints.reshape(-1, 2) - self.reference_keypoints))
            if movement > update_threshold:
                self.reference_keypoints = keypoints.reshape(-1, 2)

    def adaptive_homography(self, current_keypoints, reference_keypoints):
        # Calculate distances from the center of the image
        h, w = self.image_shape[:2]
        center = np.array([w/2, h/2])
        distances = np.linalg.norm(current_keypoints - center, axis=1)
        
        # Calculate weights based on distances (closer to center = higher weight)
        weights = 1 / (distances + 1)
        
        # Use weighted RANSAC for homography calculation
        H, status = cv2.findHomography(current_keypoints, reference_keypoints, 
                                    cv2.RANSAC, 5.0, None, maxIters=2000, 
                                    confidence=0.995, weights=weights)
        return H
    
    def local_warping(self, keypoints, image_shape):
        h, w = image_shape[:2]
        grid_size = 4  # Divide the court into a 4x4 grid
        warped_keypoints = keypoints.copy()

        for i in range(grid_size):
            for j in range(grid_size):
                # Define region boundaries
                x_start, x_end = i*w//grid_size, (i+1)*w//grid_size
                y_start, y_end = j*h//grid_size, (j+1)*h//grid_size
                
                # Find keypoints in this region
                mask = ((keypoints[:, 0] >= x_start) & (keypoints[:, 0] < x_end) &
                        (keypoints[:, 1] >= y_start) & (keypoints[:, 1] < y_end))
                local_keypoints = keypoints[mask]
                local_reference = self.reference_keypoints[mask]
                
                if len(local_keypoints) >= 4:  # Need at least 4 points for homography
                    H = cv2.findHomography(local_keypoints, local_reference)[0]
                    warped_keypoints[mask] = cv2.perspectiveTransform(
                        local_keypoints.reshape(-1, 1, 2), H).reshape(-1, 2)

        return warped_keypoints
    

    def filter_keypoints(self, keypoints, previous_keypoints, max_movement=50):
        filtered_keypoints = keypoints.copy()
        movement = np.linalg.norm(keypoints - previous_keypoints, axis=1)
        mask = movement > max_movement
        filtered_keypoints[mask] = previous_keypoints[mask]
        return filtered_keypoints
    
    def smooth_keypoints(self, keypoints, previous_keypoints, alpha=0.7):
        return alpha * keypoints + (1 - alpha) * previous_keypoints

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        image_tensor = image_tensor.float()
        with torch.no_grad():
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        # Scale the points to match the original image
        keypoints[::2] = np.clip(keypoints[::2] * (original_w / 224.0), 0, original_w)
        keypoints[1::2] = np.clip(keypoints[1::2] * (original_h / 224.0), 0, original_h)
        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def set_reference_keypoints(self, keypoints):
        self.reference_keypoints = keypoints.reshape(-1, 2)

    def apply_homography(self, keypoints, H):
        keypoints_homogeneous = np.column_stack((keypoints.reshape(-1, 2), np.ones(len(keypoints) // 2)))
        transformed_keypoints = np.dot(H, keypoints_homogeneous.T).T
        transformed_keypoints = transformed_keypoints[:, :2] / transformed_keypoints[:, 2:]
        return transformed_keypoints.flatten()

    def process_video(self, video_frames):
        output_video_frames = []
        for i, frame in enumerate(video_frames):
            self.image_shape = frame.shape
            keypoints = self.predict(frame)
            
            if i == 0:
                self.set_reference_keypoints(keypoints)
                reconstructed_keypoints = keypoints
            else:
                # Update reference keypoints if necessary
                self.update_reference_keypoints(keypoints)
                
                # Filter keypoints based on previous frame
                if self.previous_keypoints is not None:
                    keypoints = self.filter_keypoints(keypoints, self.previous_keypoints)
                
                # Apply local warping
                warped_keypoints = self.local_warping(keypoints.reshape(-1, 2), frame.shape)
                
                # Apply temporal smoothing
                if self.previous_keypoints is not None:
                    reconstructed_keypoints = self.smooth_keypoints(warped_keypoints, self.previous_keypoints)
                else:
                    reconstructed_keypoints = warped_keypoints
            
            self.previous_keypoints = reconstructed_keypoints
            
            frame_with_keypoints = self.draw_keypoints(frame, reconstructed_keypoints.flatten())
            output_video_frames.append(frame_with_keypoints)
        
        return output_video_frames