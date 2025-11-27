import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from config import VIDEO_DIR

def check_video(path):
    """Returns 1 if valid, 0 if corrupt"""
    try:
        if os.path.getsize(path) == 0:
            return 0
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0
        # Read one frame to be sure
        ret, _ = cap.read()
        cap.release()
        return 1 if ret else 0
    except:
        return 0

def main():
    print(f"Scanning {VIDEO_DIR}...")
    files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    total = len(files)
    
    if total == 0:
        print("❌ No files found! Check your path.")
        return

    print(f"Found {total} files. Checking integrity (this takes a moment)...")
    
    # Check 500 files in parallel to get a quick sample
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(check_video, files[:500]))
    
    valid_count = sum(results)
    corrupt_count = len(results) - valid_count
    
    print("-" * 30)
    print(f"Sample Size: {len(results)}")
    print(f"✅ Valid:   {valid_count}")
    print(f"❌ Corrupt: {corrupt_count}")
    print("-" * 30)
    
    if corrupt_count > 20:
        print("🚨 CRITICAL: Your dataset download is broken. You MUST re-download.")
    else:
        print("Data looks okay. The issue might be elsewhere.")

if __name__ == "__main__":
    main()