import torch
from dataset import WLASLDataset
from torch.utils.data import DataLoader
import numpy as np
import sys

# Config
JSON_PATH = "data/WLASL_v0.3.json"
VIDEO_DIR = "data/videos"
CHECK_LIMIT = 50  # Check first 50 videos only to save time

def check_integrity():
    print(f"🕵️  Inspecting the first {CHECK_LIMIT} videos in the Training Set...")
    
    # Initialize Dataset
    try:
        ds = WLASLDataset(JSON_PATH, VIDEO_DIR, split='train')
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    dataloader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    bad_videos = 0
    
    print("\n" + "="*60)
    print(f"{'Video Index':<12} | {'Label':<20} | {'Status':<20}")
    print("-" * 60)

    for i, (video, label) in enumerate(dataloader):
        if i >= CHECK_LIMIT:
            break
        
        # Calculate standard deviation. 
        # A valid video has variance (changes in color/motion). 
        # A "Ghost" video (zeros or flat color) has near 0.0 variance.
        std_dev = torch.std(video).item()
        
        gloss = ds.glosses[label.item()]
        
        if std_dev < 0.01:
            status = "❌ BLACK / FLAT"
            bad_videos += 1
        else:
            status = "✅ OK"
            
        print(f"{i:<12} | {gloss:<20} | {status}")
        sys.stdout.flush()

    print("="*60)
    print(f"Summary: {bad_videos}/{CHECK_LIMIT} videos appear to be corrupted/unreadable.")
    
    if bad_videos > CHECK_LIMIT * 0.5:
        print("\n⚠️  CRITICAL: More than 50% of your data is unreadable.")
    elif bad_videos > 0:
        print(f"\n⚠️  Warning: {bad_videos} corrupted videos found.")
    else:
        print("\n✅ Data looks good! If training still fails, check your Learning Rate.")

if __name__ == "__main__":
    check_integrity()
