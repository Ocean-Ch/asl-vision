import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from dataset import WLASLDataset
import os
from tqdm import tqdm
from device import get_device
import config

# CONFIG
DEVICE = get_device()

class FeatureExtractor(nn.Module):
    """
    Feature extractor model for extracting features from videos.
    """
    def __init__(self):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        self.feature_extractor = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Forward pass through the model.
        """
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = self.pool(x)
            return x.flatten(1)

def extract():
    """
    Extract features from the videos and save them to a file.
    """
    print(f"🚀 Extracting features to Memory Map... Target Size: ~2GB")
    model = FeatureExtractor().to(DEVICE)
    model.eval()

    # Dictionary to store EVERYTHING: { 'video_path_string': tensor_data }
    feature_cache = {}

    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split}...")
        ds = WLASLDataset(config.JSON_PATH, config.VIDEO_DIR, split=split, frames_per_clip=32, use_cached_features=False)
        
        # num_workers=0 is crucial for Windows/OpenCV stability
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        
        # We need to map extracted features back to the specific video
        # Since loader maintains order, we iterate through the dataset indices in parallel
        global_idx = 0
        
        for videos, _ in tqdm(loader):
            # Move to GPU and Extract
            b, f, c, h, w = videos.shape
            flat_videos = videos.to(DEVICE).view(b * f, c, h, w)
            
            features = model(flat_videos) # (B*F, 1280)
            features = features.view(b, f, -1).cpu() # (B, 32, 1280) Move back to RAM
            
            # Map batch back to video paths
            for i in range(b):
                # Get the absolute path from the dataset samples list
                # logic: current_batch_index + i
                dataset_idx = global_idx + i
                video_path = ds.samples[dataset_idx]['video_path']
                
                # Normalize path string to avoid slash/backslash issues later
                key = os.path.normpath(video_path)
                
                # Store in dict
                feature_cache[key] = features[i].clone()
                
            global_idx += b

    print(f"💾 Saving {len(feature_cache)} videos to {config.FEATURE_FILE}...")
    torch.save(feature_cache, config.FEATURE_FILE)
    print("✅ Features saved")

if __name__ == "__main__":
    extract()