import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# CONFIG
JSON_PATH = "data/WLASL_v0.3.json"
VIDEO_DIR = "data/videos"

def visualize():
    if not os.path.exists(JSON_PATH):
        print(f"❌ JSON not found at {JSON_PATH}")
        return

    print("Loading JSON...")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Stats containers
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    missing_counts = {'train': 0, 'val': 0, 'test': 0}
    class_counts = [] # Store (gloss, count) tuples

    print("Scanning dataset structure...")
    
    for entry in data:
        gloss = entry['gloss']
        valid_instances_for_gloss = 0
        
        for inst in entry['instances']:
            split = inst['split']
            video_id = inst['video_id']
            path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
            
            if os.path.exists(path):
                split_counts[split] += 1
                valid_instances_for_gloss += 1
            else:
                missing_counts[split] += 1
        
        class_counts.append((gloss, valid_instances_for_gloss))

    # --- REPORTING ---
    print("\n" + "="*40)
    print("DATASET STATISTICS (On Disk)")
    print("="*40)
    print(f"{'Split':<10} | {'Found':<10} | {'Missing':<10} | {'% Present':<10}")
    print("-" * 46)
    
    total_found = 0
    for split in ['train', 'val', 'test']:
        found = split_counts[split]
        missing = missing_counts[split]
        total = found + missing
        percent = (found / total * 100) if total > 0 else 0
        total_found += found
        print(f"{split.upper():<10} | {found:<10} | {missing:<10} | {percent:.1f}%")
    
    print("-" * 46)
    print(f"TOTAL VALID VIDEOS: {total_found}")
    print("="*40)

    # --- PLOTTING ---
    # Sort classes by frequency
    class_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Get top 50 classes for plotting
    top_n = 50
    top_classes = class_counts[:top_n]
    labels = [x[0] for x in top_classes]
    values = [x[1] for x in top_classes]

    plt.figure(figsize=(15, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Gloss (Sign Word)')
    plt.ylabel('Number of Video Instances')
    plt.title(f'Top {top_n} Most Frequent Signs in Your Dataset')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    
    print(f"\nPlotting distribution for top {top_n} classes...")
    plt.show()
    
    # Distribution Analysis
    zero_samples = sum(1 for x in class_counts if x[1] == 0)
    print(f"\n⚠️ Warning: {zero_samples} classes have 0 video samples on disk.")
    print("These classes will never be predicted correctly.")

if __name__ == "__main__":
    visualize()