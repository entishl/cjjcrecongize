import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import List

class FeatureExtractor:
    def __init__(self, model_name: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extracts feature vectors from a batch of images.
        
        Args:
            images: A list of images as numpy arrays (H, W, C) in BGR format.
        
        Returns:
            A numpy array of shape (N, D), where N is the number of images
            and D is the feature dimension.
        """
        # Convert BGR (OpenCV) to RGB (PIL) and apply transformations
        pil_images = [Image.fromarray(cv_img[:, :, ::-1]) for cv_img in images]
        image_tensors = torch.stack([self.transform(img) for img in pil_images]).to(self.device)

        features = self.model(image_tensors)
        
        # L2 Normalize the features
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()