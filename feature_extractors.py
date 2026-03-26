import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import *

class DinoV2FeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "dinov2_vitb14"):
        super().__init__()
        self.model_name = model_name
        print(f"[*] Loading DINOv2 via Torch Hub: {model_name}")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).eval()
        self.transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def _prepare_image(self, image: Union[torch.Tensor, List[Image.Image]], size: int = 518) -> torch.Tensor:
        if isinstance(image, list):
            processed = []
            for i in image:
                i = i.convert('RGB').resize((size, size), Image.LANCZOS)
                arr = np.array(i).astype(np.float32) / 255.0
                processed.append(torch.from_numpy(arr).permute(2, 0, 1))
            image = torch.stack(processed)
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
            
        return image.to(next(self.parameters()).device)

    @torch.no_grad()
    def forward(self, image: Union[torch.Tensor, List[Image.Image]]) -> torch.Tensor:
        img = self._prepare_image(image, size=518)
        img = self.transform(img)
        
        # Ambil target dtype dari model (biasanya BFloat16 di LAB-AI-04)
        target_dtype = next(self.model.parameters()).dtype
        img = img.to(target_dtype)
        
        features = self.model.forward_features(img)["x_prenorm"]
        # Pastikan output layer norm juga dalam dtype yang benar
        out = F.layer_norm(features, (features.shape[-1],))
        return out.to(target_dtype)