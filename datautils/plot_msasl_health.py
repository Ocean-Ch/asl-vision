import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import config

def analyze_msasl(
    json_path: str = config.MSASL_TRAIN_JSON,
    video_dir: str = config.MSASL_OUTPUT_DIR,
    top_n: int = config.NUM_CLASSES
):
    """
    Analyzes the MS-ASL dataset availability.
    Matches the naming convention: class_{label}_{index}.mp4
    """
    
    # 1. Load Data
    print(f"📖 Loading MS-ASL metadata from {json_path}...")
    if not os.path.exists(json_path):
        print("❌ JSON file not found.")
        return

    df = pd.read_json(json_path)
    
    # Filter for the classes we care about (0 to Top N)
    # MS-ASL labels are 0-indexed and sorted by frequency (0 is most frequent)
    df_subset = df[df['label'] < top_n].copy()
    
    print(f"🕵️  Scanning {len(df_subset)} video entries for Top {top_n} classes...")

    # 2. Check Disk Availability
    # We need to reconstruct the EXACT filename used by download_msasl.py
    # format: class_{label}_{original_dataframe_index}.mp4
    
    actual_counts = {}   # {label: count}
    expected_counts = {} # {label: count}
    gloss_map = {}       # {label: text_name}

    # Initialize counters
    unique_labels = df_subset['label'].unique()
    for label in unique_labels:
        actual_counts[label] = 0
        expected_counts[label] = 0

    # Scan
    for index, row in df_subset.iterrows():
        label = row['label']
        gloss = row['text']
        
        # Store gloss name for plotting later
        if label not in gloss_map:
            gloss_map[label] = gloss
            
        expected_counts[label] += 1
        
        # Construct filename based on your download script's logic
        filename = f"class_{label}_{index}.mp4"
        path = os.path.join(video_dir, filename)
        
        if os.path.exists(path):
            actual_counts[label] += 1

    # 3. Aggregate Stats for Plotting
    stats = []
    for label in sorted(unique_labels):
        stats.append({
            'gloss': gloss_map[label],
            'label': label,
            'expected': expected_counts[label],
            'actual': actual_counts[label]
        })

    # Sort by expected count (should already be sorted by label, but just to be safe)
    stats.sort(key=lambda x: x['expected'], reverse=True)
    
    # Print summary of the "Best" class vs "Worst" class in this subset
    if stats:
        print(f"\n📊 Summary for Top {top_n}:")
        print(f"   Best Available: '{stats[0]['gloss']}' ({stats[0]['actual']}/{stats[0]['expected']})")
        print(f"   Worst Available: '{stats[-1]['gloss']}' ({stats[-1]['actual']}/{stats[-1]['expected']})")

    # --- PLOT 1: Top N Comparison ---
    plot_top_n_comparison(stats, top_n)

    # --- PLOT 2: Distribution Histogram ---
    plot_distribution_histogram(stats)

def plot_top_n_comparison(stats, top_n):
    """
    Plots Expected vs Actual.
    """
    # Just take the top 50 for the bar chart so it's readable, 
    # even if we analyzed 100.
    display_limit = min(50, len(stats))
    subset = stats[:display_limit]
    
    glosses = [x['gloss'] for x in subset]
    expected = [x['expected'] for x in subset]
    actual = [x['actual'] for x in subset]

    x = np.arange(len(glosses))
    
    plt.figure(figsize=(15, 6))
    
    # Expected (Background)
    plt.bar(x, expected, width=0.8, label='Expected (JSON)', color='lightgray', alpha=0.8)
    
    # Actual (Foreground)
    plt.bar(x, actual, width=0.5, label='Downloaded (Disk)', color='#1f77b4', alpha=0.9)

    plt.title(f'MS-ASL Data Availability (Top {display_limit} Displayed)')
    plt.ylabel('Number of Videos')
    plt.xlabel('Sign Class')
    plt.xticks(x, glosses, rotation=90, fontsize=8)
    plt.legend()
    plt.tight_layout()
    
    filename = 'msasl_health_check.png'
    plt.savefig(filename)
    print(f"✅ Saved comparison graph to {filename}")
    plt.close()

def plot_distribution_histogram(stats):
    """
    Plots distribution of available videos per class.
    """
    actual_sizes = [x['actual'] for x in stats]

    plt.figure(figsize=(10, 6))
    
    # Create bins dynamically based on data range
    if not actual_sizes:
        return

    max_val = max(actual_sizes)
    bins = range(0, max_val + 5, 5) 

    plt.hist(actual_sizes, bins=bins, alpha=0.7, color='green', edgecolor='black')

    plt.title('MS-ASL Actual Density (How many videos do we actually have?)')
    plt.xlabel('Number of Videos per Class')
    plt.ylabel('Number of Classes')
    plt.grid(True, alpha=0.3)
    
    filename = 'msasl_distribution.png'
    plt.savefig(filename)
    print(f"✅ Saved distribution histogram to {filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MS-ASL Dataset Health")
    parser.add_argument("--top", type=int, default=config.NUM_CLASSES, help="Number of classes to analyze")
    
    args = parser.parse_args()
    
    analyze_msasl(top_n=args.top)