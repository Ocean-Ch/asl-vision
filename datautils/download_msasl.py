"""
Script to download MS-ASL videos.

This script downloads videos from the MS-ASL dataset, filtering to the top N classes
and trimming videos to the specific sign segments using yt-dlp and ffmpeg.

Usage:
    python datautils/download_msasl.py
"""

import pandas as pd
import os
import subprocess
from config import MSASL_TRAIN_JSON, MSASL_OUTPUT_DIR, NUM_CLASSES


def download_msasl_100():
    """
    Download MS-ASL videos for the top N classes (default: 100).
    
    The script:
    1. Loads the MS-ASL train JSON file
    2. Filters to keep only the top NUM_CLASSES classes (labels 0 to NUM_CLASSES-1)
    3. Downloads each video using yt-dlp, trimming to the specific sign segment
    4. Saves videos to MSASL_OUTPUT_DIR
    """
    if not os.path.exists(MSASL_OUTPUT_DIR):
        os.makedirs(MSASL_OUTPUT_DIR)

    # Load dataset
    print("Reading JSON...")
    if not os.path.exists(MSASL_TRAIN_JSON):
        print(f"Error: {MSASL_TRAIN_JSON} not found. Please download it from:")
        print("https://github.com/microsoft/MS-ASL/blob/master/MS-ASL_train.json")
        return
    
    df = pd.read_json(MSASL_TRAIN_JSON)
    
    # FILTER: Keep only Top N classes (Labels 0 to NUM_CLASSES-1)
    # MS-ASL labels are already sorted by frequency (0 = most frequent)
    df_filtered = df[df['label'] < NUM_CLASSES]
    
    print(f"Total videos to download: {len(df_filtered)}")
    
    downloaded_count = 0
    failed_count = 0
    
    for index, row in df_filtered.iterrows():
        url = row['url']
        start_time = row['start_time']
        end_time = row['end_time']
        label = row['label']
        video_id = f"class_{label}_{index}"
        
        output_path = os.path.join(MSASL_OUTPUT_DIR, f"{video_id}.mp4")
        
        if os.path.exists(output_path):
            downloaded_count += 1
            continue
            
        print(f"Downloading {video_id} ({row['text']})...")
        
        if not url.startswith(("http://", "https://")):
            url = "https://" + url


        # Quote it so PowerShell doesn't glob the asterisk
        section = f"*{start_time}-{end_time}"

        cmd = [
            "yt-dlp",
            "--force-keyframes-at-cuts",
            "--download-sections", section,

            # Prefer 720p video; fallback to best available
            "-f", "bv*[height=720]/bv*+ba/b[height=720]/b",

            "--no-audio",                 # strip audio track completely
            "--remux-video", "mp4",       # ensure MP4 output

            "-o", output_path,
            url
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            downloaded_count += 1
        except Exception as e:
            print(f"Failed {video_id}: {e}")
            failed_count += 1
    
    print(f"\nDownload summary:")
    print(f"  Successfully downloaded: {downloaded_count}")
    print(f"  Failed to download: {failed_count}")
    print(f"  Total processed: {downloaded_count + failed_count}")


if __name__ == "__main__":
    download_msasl_100()

