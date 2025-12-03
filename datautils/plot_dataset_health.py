import json
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import config

def analyze_dataset(
    json_path: str = config.TRAIN_JSON_PATH,
    video_dir: str = config.VIDEO_DIR,
    top_n: int = config.NUM_CLASSES):
    """
    Analyzes the WLASL dataset to compare metadata counts vs physical file counts.
    """
    
    # 1. Load Data
    print(f"📖 Loading JSON from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: JSON file not found.")
        return

    # Handle dictionary wrapper edge case
    if isinstance(raw_data, dict):
        if 'data' in raw_data: raw_data = raw_data['data']
        elif 'values' in raw_data: raw_data = raw_data['values']
        else: raw_data = list(raw_data.values())

    stats = []
    print(f"🕵️  Scanning {len(raw_data)} classes (this may take a moment)...")

    # 2. Scan Directory
    for i, entry in enumerate(raw_data):
        gloss = entry['gloss']
        
        # Count expected (according to JSON)
        expected_count = len(entry['instances'])
        
        # Count actual (checking disk)
        actual_count = 0
        for inst in entry['instances']:
            video_id = inst['video_id']
            # Check both .mp4 and standard casing just in case
            path = os.path.join(video_dir, f"{video_id}.mp4")
            if os.path.exists(path):
                actual_count += 1
        
        stats.append({
            'gloss': gloss,
            'expected': expected_count,
            'actual': actual_count
        })

        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1} classes...")

    # Sort by EXPECTED count (to show the "ideal" top classes)
    stats.sort(key=lambda x: x['expected'], reverse=True)

    # --- PLOT 1: Top N Comparison ---
    plot_top_n_comparison(stats, top_n)

    # --- PLOT 2: Distribution Histogram ---
    plot_distribution_histogram(stats)

def plot_top_n_comparison(stats, top_n):
    """
    Plots a dual bar chart: Expected vs Actual for the top N classes.
    """
    subset = stats[:top_n]
    
    glosses = [x['gloss'] for x in subset]
    expected = [x['expected'] for x in subset]
    actual = [x['actual'] for x in subset]

    x = np.arange(len(glosses))
    width = 0.35

    plt.figure(figsize=(15, 6))
    
    # Plot 'Expected' as a lighter, background bar
    plt.bar(x, expected, width=0.8, label='Expected (JSON Metadata)', color='lightgray', alpha=0.8)
    
    # Plot 'Actual' as a foreground bar
    # We use a slightly narrower bar to make the "gap" visible
    plt.bar(x, actual, width=0.5, label='Actual (Found on Disk)', color='#1f77b4', alpha=0.9)

    plt.title(f'Data Availability for Top {top_n} Classes (by Frequency)')
    plt.ylabel('Number of Videos')
    plt.xticks(x, glosses, rotation=90, fontsize=9)
    plt.legend()
    plt.tight_layout()
    
    filename = 'wlasl_top_n_health.png'
    plt.savefig(filename)
    print(f"✅ Saved Top-N comparison graph to {filename}")
    plt.close()

def plot_distribution_histogram(stats):
    """
    Plots histograms showing the distribution of class sizes.
    """
    expected_sizes = [x['expected'] for x in stats]
    actual_sizes = [x['actual'] for x in stats]

    plt.figure(figsize=(12, 6))

    # We limit the range to 0-100 to make the graph readable 
    # (since the tail is long and small)
    bins = range(0, 80, 2) 

    plt.hist(expected_sizes, bins=bins, alpha=0.5, label='Expected Distribution', color='gray')
    plt.hist(actual_sizes, bins=bins, alpha=0.7, label='Actual Distribution', color='red')

    plt.title('Distribution of Class Sizes (How many classes have X videos?)')
    plt.xlabel('Number of Videos in Class')
    plt.ylabel('Number of Classes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = 'wlasl_distribution_health.png'
    plt.savefig(filename)
    print(f"✅ Saved distribution histogram to {filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WLASL Dataset Health")
    parser.add_argument("--json_path", type=str, default=config.TRAIN_JSON_PATH, help="Path to WLASL_v0.3.json")
    parser.add_argument("--video_dir", type=str, default=config.VIDEO_DIR, help="Path to the videos folder")
    parser.add_argument("--top", type=int, default=config.NUM_CLASSES, help="Number of top classes to plot")
    
    args = parser.parse_args()
    
    analyze_dataset(args.json_path, args.video_dir, args.top)
