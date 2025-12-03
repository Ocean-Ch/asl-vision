import json
import os
import torch
import random
import config
from typing import List, Dict


def find_top_n_classes(
    json_path: str = config.JSON_PATH,
    video_dir: str = config.VIDEO_DIR,
    num_classes: int = config.NUM_CLASSES,
    use_cached: bool = config.USE_CACHED_FEATURES,
    cached_features: Dict = None
) -> List[Dict]:
    """
    Finds the top N classes with the most valid video files across ALL splits (train/val/test).
    
    Args:
        json_path: Path to the JSON metadata file
        video_dir: Directory containing video files
        num_classes: Number of top classes to select
        use_cached: Whether to use cached features for validation
        cached_features: Dictionary of cached features (if use_cached is True)
    
    Returns:
        List of entry dictionaries for the top N classes, with all instances consolidated
    """
    print(f"\n🕵️  Scanning classes for valid video files across ALL splits...")
    
    # Load cached features if needed
    if use_cached and cached_features is None:
        print(f"💾 Loading cached features from {config.FEATURE_FILE}...")
        if os.path.exists(config.FEATURE_FILE):
            cached_features = torch.load(config.FEATURE_FILE)
            print(f"💾 Loaded {len(cached_features)} features into cache")
        else:
            print(f"⚠️  Feature file not found: {config.FEATURE_FILE}")
            print("   Falling back to disk-based validation")
            use_cached = False
    
    # Load JSON metadata
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
    
    class_stats = []
    
    # Count valid videos for every class across ALL splits
    for entry in raw_data:
        gloss = entry['gloss']
        valid_count = 0
        
        # Count valid videos across all splits (train, val, test)
        for inst in entry['instances']:
            video_id = inst['video_id']
            path = os.path.normpath(os.path.join(video_dir, f"{video_id}.mp4"))
            
            is_valid = False
            
            if use_cached:
                # Check dictionary lookup (fast) + content check
                if path in cached_features:
                    # check if tensor is not empty/zero
                    if torch.sum(cached_features[path]) != 0:
                        is_valid = True
            else:
                # Check disk (slower)
                if os.path.exists(path):
                    is_valid = True
            
            if is_valid:
                valid_count += 1
        
        # Store stats for this class
        class_stats.append({
            'count': valid_count,
            'gloss': gloss,
            'entry': entry
        })
    
    # Sort by count (descending)
    class_stats.sort(key=lambda x: x['count'], reverse=True)
    
    # Get top N classes
    top_n_stats = class_stats[:num_classes]
    
    # Print statistics
    print(f"📊 Dataset Statistics (Top {num_classes} Classes):")
    print(f"{'Rank':<6} {'Gloss':<20} {'Valid Videos':<15}")
    print("-" * 45)
    
    # Print Top 5
    for i in range(min(5, len(top_n_stats))):
        item = top_n_stats[i]
        print(f"#{i+1:<5} {item['gloss']:<20} {item['count']:<15}")
    
    if len(top_n_stats) > 10:
        print(f"{'...':<6} {'...':<20} {'...':<15}")
    
    # Print Bottom 5 (of the selected set)
    start_idx = max(5, len(top_n_stats) - 5)
    for i in range(start_idx, len(top_n_stats)):
        item = top_n_stats[i]
        print(f"#{i+1:<5} {item['gloss']:<20} {item['count']:<15}")
    
    print("-" * 45)
    
    total_videos = sum(x['count'] for x in top_n_stats)
    print(f"✅ Total videos in top {num_classes} classes: {total_videos}")
    print(f"📉 Cutoff threshold: Classes must have at least {top_n_stats[-1]['count']} videos.\n")
    
    # Return consolidated entries (all instances from all splits)
    consolidated_entries = []
    for stat in top_n_stats:
        entry = stat['entry'].copy()
        # Keep all instances, but we'll filter to only valid ones
        valid_instances = []
        for inst in entry['instances']:
            video_id = inst['video_id']
            path = os.path.normpath(os.path.join(video_dir, f"{video_id}.mp4"))
            
            is_valid = False
            if use_cached:
                if path in cached_features:
                    if torch.sum(cached_features[path]) != 0:
                        is_valid = True
            else:
                if os.path.exists(path):
                    is_valid = True
            
            if is_valid:
                valid_instances.append(inst)
        
        entry['instances'] = valid_instances
        consolidated_entries.append(entry)
    
    return consolidated_entries


def resplit_dataset(
    consolidated_entries: List[Dict],
    train_percent: float = config.TRAIN_SPLIT_PERCENT,
    val_percent: float = config.VAL_SPLIT_PERCENT,
    test_percent: float = config.TEST_SPLIT_PERCENT,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Resplits consolidated video instances into train/val/test sets.
    
    Args:
        consolidated_entries: List of entry dictionaries with all instances
        train_percent: Percentage of videos for training (default: 0.7)
        val_percent: Percentage of videos for validation (default: 0.15)
        test_percent: Percentage of videos for testing (default: 0.15)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing a list of entries
    """
    # Validate percentages
    total_percent = train_percent + val_percent + test_percent
    if abs(total_percent - 1.0) > 1e-6:
        raise ValueError(f"Split percentages must sum to 1.0, got {total_percent}")
    
    print(f"\n🔄 Resplitting dataset with ratios: Train={train_percent:.1%}, Val={val_percent:.1%}, Test={test_percent:.1%}")
    
    random.seed(seed)
    
    resplit_data = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # For each class, resplit its instances
    for entry in consolidated_entries:
        gloss = entry['gloss']
        instances = entry['instances']
        
        # Shuffle instances for this class
        shuffled_instances = instances.copy()
        random.shuffle(shuffled_instances)
        
        total_instances = len(shuffled_instances)
        train_count = int(total_instances * train_percent)
        val_count = int(total_instances * val_percent)
        # Remaining go to test
        
        train_instances = shuffled_instances[:train_count]
        val_instances = shuffled_instances[train_count:train_count + val_count]
        test_instances = shuffled_instances[train_count + val_count:]
        
        # Create entries for each split
        # Preserve all fields from original instances, but update the 'split' field
        if train_instances:
            train_instances_updated = []
            for inst in train_instances:
                new_inst = inst.copy()
                new_inst['split'] = 'train'
                train_instances_updated.append(new_inst)
            resplit_data['train'].append({
                'gloss': gloss,
                'instances': train_instances_updated
            })
        
        if val_instances:
            val_instances_updated = []
            for inst in val_instances:
                new_inst = inst.copy()
                new_inst['split'] = 'val'
                val_instances_updated.append(new_inst)
            resplit_data['val'].append({
                'gloss': gloss,
                'instances': val_instances_updated
            })
        
        if test_instances:
            test_instances_updated = []
            for inst in test_instances:
                new_inst = inst.copy()
                new_inst['split'] = 'test'
                test_instances_updated.append(new_inst)
            resplit_data['test'].append({
                'gloss': gloss,
                'instances': test_instances_updated
            })
    
    # Print statistics
    print(f"\n📊 Resplit Statistics:")
    for split_name in ['train', 'val', 'test']:
        total_videos = sum(len(entry['instances']) for entry in resplit_data[split_name])
        num_classes = len(resplit_data[split_name])
        print(f"  {split_name.upper():<6}: {total_videos:>5} videos across {num_classes:>3} classes")
    
    return resplit_data


def save_resplit_json(
    resplit_data: Dict[str, List[Dict]],
    output_dir: str = config.RESPLIT_OUTPUT_DIR,
    num_classes: int = config.NUM_CLASSES
):
    """
    Saves the resplit data as JSON files.
    
    Args:
        resplit_data: Dictionary with 'train', 'val', 'test' keys
        output_dir: Directory to save JSON files
        num_classes: Number of classes (for filename)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save consolidated data (all splits combined)
    consolidated_path = os.path.join(output_dir, f"consolidated_top_{num_classes}.json")
    consolidated_data = []
    for split_name in ['train', 'val', 'test']:
        consolidated_data.extend(resplit_data[split_name])
    
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)
    print(f"✅ Saved consolidated data to: {consolidated_path}")
    
    # Save individual split files
    for split_name in ['train', 'val', 'test']:
        split_path = os.path.join(output_dir, f"{split_name}_top_{num_classes}.json")
        with open(split_path, 'w') as f:
            json.dump(resplit_data[split_name], f, indent=2)
        print(f"✅ Saved {split_name} split to: {split_path}")


def main():
    """
    Main function to consolidate top N classes and resplit them.
    """
    # Step 1: Find top N classes with most valid videos across all splits
    consolidated_entries = find_top_n_classes(
        json_path=config.JSON_PATH,
        video_dir=config.VIDEO_DIR,
        num_classes=config.NUM_CLASSES,
        use_cached=config.USE_CACHED_FEATURES
    )
    
    # Save intermediate result: consolidated entries (before resplitting)
    os.makedirs(config.RESPLIT_OUTPUT_DIR, exist_ok=True)
    consolidated_before_path = os.path.join(
        config.RESPLIT_OUTPUT_DIR, 
        f"consolidated_before_resplit_top_{config.NUM_CLASSES}.json"
    )
    with open(consolidated_before_path, 'w') as f:
        json.dump(consolidated_entries, f, indent=2)
    print(f"✅ Saved consolidated entries (before resplit) to: {consolidated_before_path}\n")
    
    # Step 2: Resplit the consolidated videos
    resplit_data = resplit_dataset(
        consolidated_entries,
        train_percent=config.TRAIN_SPLIT_PERCENT,
        val_percent=config.VAL_SPLIT_PERCENT,
        test_percent=config.TEST_SPLIT_PERCENT
    )
    
    # Step 3: Save resplit JSON files
    save_resplit_json(resplit_data, config.RESPLIT_OUTPUT_DIR, config.NUM_CLASSES)
    
    print(f"\n✅ Done! All files saved to: {config.RESPLIT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

