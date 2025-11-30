import torch
from torch.utils.data import Dataset
import cv2
import json
import os
import numpy as np
from torchvision import transforms
from typing import Tuple
import config


class WLASLDataset(Dataset):
    """
    A PyTorch Dataset for loading WLASL video data.
    
    This class handles:
    - Loading video file paths and labels from JSON metadata
    - Extracting frames from videos using uniform sampling
    - Preprocessing frames (resize to 224x224, normalize)
    - Converting sign language words (glosses) to numeric IDs
    
    Args:
        json_path (str): JSON file containing video metadata
        video_dir (str): directory containing the video files (.mp4)
        frames_per_clip (int): frames to extract from each video (default: 32)
        split (str): dataset split to use ('train', 'val', or 'test') (default: 'train')
    
    Attributes:
        samples (list): List of dicts, each containing 'video_path' and 'gloss'
        glosses (list): All unique glosses in dataset (sorted)
        gloss_to_id (dict): map from gloss -> numeric ID
        transform: PyTorch transform pipeline for image preprocessing
    """
    def __init__(
        self, 
        json_path: str, 
        video_dir: str, 
        frames_per_clip: int = config.FRAMES_PER_CLIP, 
        split: str = 'train',
        debug_mode: bool = False,
        use_cached_features: bool = True,
        num_classes: int = None,
        augment: bool = config.AUGMENT_ENABLED
    ):
        """
        Initialize the dataset by loading metadata and setting up preprocessing.
        
        The initialization process:
        1. Loads JSON file containing video metadata (gloss labels, video IDs, splits)
        2. Filters videos by the specified split (train/val/test)
        3. Creates a vocabulary mapping from sign language words to numeric IDs
        4. Sets up image preprocessing pipeline
        
        Expected JSON/dict format for metadata file:
        [
            {
                "gloss": "hello",
                "instances": [
                    {
                        "video_id": "12345",
                        "split": "train"
                    },
                    {
                        "video_id": "67890",
                        "split": "val"
                    }
                ]
            }
        ]
        """
        self.video_dir = video_dir
        self.frames_per_clip = frames_per_clip
        self.use_cached_features = use_cached_features
        # only augment data if split is train, and if augment is enabled
        self.augment = augment and split == 'train'

        # load cache if enabled
        self.cached_features = {}
        if self.use_cached_features:
            print(f"💾 Loading cached features from {config.FEATURE_FILE}...")
            if os.path.exists(config.FEATURE_FILE):
                self.cached_features = torch.load(config.FEATURE_FILE)
                print(f"💾 Loaded {len(self.cached_features)} features into cache from disk")
            else:
                raise FileNotFoundError(f"Feature file not found: {config.FEATURE_FILE}, run extract_to_memory.py to create it")
        
        # load JSON metadata file
        # contains entries with 'gloss' (sign language word) and 'instances' (video examples)
        with open(json_path, 'r') as f:
            raw_data = json.load(f)

        # filter for top N most frequent classes
        if num_classes is not None:
            raw_data = filter_by_availability(
                raw_data,
                video_dir,
                num_classes,
                use_cached_features,
                self.cached_features
            )
            
        # create vocabulary: map each unique gloss to a numeric ID (dupes removed, sorted)
        self.glosses = sorted([entry['gloss'] for entry in raw_data])
        self.gloss_to_id = {g: i for i, g in enumerate(self.glosses)}
        print(f"🔍 Created vocabulary with {len(self.glosses)} unique glosses")

        self.samples = []  # store all valid video samples

        num_videos = 0
        num_videos_not_found = 0
        num_dead_videos = 0 # videos that are not found or are empty
        
        # parse JSON to extract video paths and corresponding labels
        for entry in raw_data:
            # extract gloss from entry
            gloss = entry['gloss']
            for inst in entry['instances']:
                # only use videos from the specified split (train/val/test)
                if inst['split'] == split:
                    video_id = inst['video_id']
                    # need to normalize path for windows compatibility
                    path = os.path.normpath(os.path.join(video_dir, f"{video_id}.mp4"))

                    # if cached features are enabled, use them to check if the video is valid
                    if self.use_cached_features:
                        if path in self.cached_features:
                            tensor = self.cached_features[path]
                            if torch.sum(tensor) == 0:
                                num_dead_videos += 1
                                continue
                            self.samples.append({
                                'video_path': path,
                                'gloss': gloss
                            })
                            num_videos += 1
                        else:
                            num_videos_not_found += 1
                        continue

                    # ===== CACHED FEATURES DISABLED =====
                    # only add if video file exists on disk
                    if os.path.exists(path):
                        num_videos += 1
                        self.samples.append({
                            'video_path': path,
                            'gloss': gloss
                        })
                    else:
                        num_videos_not_found += 1
                        # log file not found
                        print(f"Video file not found: {path}")


        
        if self.use_cached_features:
            self.cached_features = torch.load(config.FEATURE_FILE)
            # need to check if feature file exists
            print(f"💾 Loaded {len(self.cached_features)} features into cache from disk")
            print(f"[{split.upper()}] Loaded {num_videos} videos from {json_path}")
            print(f"[{split.upper()}] Of which, {num_videos_not_found} videos were not found")
            print(f"[{split.upper()}] Of which, {num_dead_videos} videos were dead")
            return
        
        # set up image preprocessing pipeline
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # np array -> PIL Image (required for torchvision transforms)
            transforms.Resize((224, 224)),  # resize to 224x224 (standard size for pretrained models like MobileNet)
            transforms.ToTensor(),  # PIL Image -> PyTorch tensor and scale pixel values to [0, 1]
            
            # normalize using ImageNet dataset statistics (mean and std for RGB channels)
            # helps training by centering the data around 0 and scaling it
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        print(f"[{split.upper()}] Loaded {num_videos} videos from {json_path}")
        print(f"[{split.upper()}] However, {num_videos_not_found} videos were not found")

    def __len__(self):
        """
        Returns the number of samples in the dataset. 
        Required by PyTorch's Dataset interface to know how many samples are available.
        """
        return len(self.samples)

    def load_video_frames(self, video_path: str, frames_per_clip: int) -> torch.Tensor:
        """
        Loads and processes frames from a video file at `video_path`.

        Args:
            video_path (str): Path to the video file.
            frames_per_clip (int): Number of frames to sample from the video.

        Returns:
            torch.Tensor: Tensor of shape (frames_per_clip, 3, 224, 224).
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # handle broken or empty videos
        if total_frames < 1:
            cap.release()
            return torch.zeros((frames_per_clip, 3, 224, 224))

        indices = np.linspace(0, total_frames - 1, frames_per_clip).astype(int)
        frames = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frames.append(frame)
            else:
                frames.append(torch.zeros(3, 224, 224))

        cap.release()
        return torch.stack(frames)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads and returns a single video sample.
        Called automatically by DataLoader for each sample in a batch.
        """
        item = self.samples[idx]
        
        if self.use_cached_features:
            # Load the cached features
            features = self.cached_features[item['video_path']]
            
            # Apply speed augmentation if enabled
            if self.augment:
                features = self._apply_speed_augmentation(features)

            return features, self.gloss_to_id[item['gloss']]

        # ===== CACHED FEATURES DISABLED =====
        frames = self.load_video_frames(item['video_path'], self.frames_per_clip)
        return frames, self.gloss_to_id[item['gloss']]

    def _apply_speed_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Applies speed augmentation (temporal resampling) to video features.
        
        This augmentation randomly changes the playback speed of the video by:
        1. Choosing a random speed factor (0.5x to 1.5x)
        2. Resampling frames using linear interpolation
        3. Padding or truncating to match the required frame count
        
        Args:
            features (torch.Tensor): Input features tensor of shape (num_frames, feature_dim)
            
        Returns:
            torch.Tensor: Augmented features tensor of shape (frames_per_clip, feature_dim)
        """
        # 1. Choose a random speed factor (0.5x to 1.5x)
        # < 1.0 means fast (fewer frames), > 1.0 means slow (more frames)
        speed_factor = np.random.uniform(0.5, 1.5)
        
        # 2. Determine the new length
        original_len = features.shape[0]
        target_len = int(original_len * speed_factor)
        
        # Ensure we have at least a few frames
        target_len = max(5, target_len)
        
        # 3. Resample using linear indices
        # np.linspace gives us `target_len` evenly spaced indices from 0 to original_len-1
        indices = np.linspace(0, original_len - 1, target_len).astype(int)
        features = features[indices]
        
        # 4. Pad or Truncate to match self.frames_per_clip
        # This is CRITICAL because DataLoader requires consistent batch shapes
        current_len = features.shape[0]
        
        if current_len > self.frames_per_clip:
            # Truncate: take the first N frames
            features = features[:self.frames_per_clip]
        elif current_len < self.frames_per_clip:
            # Pad: append zeros to the end
            padding_len = self.frames_per_clip - current_len
            padding = torch.zeros((padding_len, features.shape[1]))
            features = torch.cat((features, padding), dim=0)
        
        return features


def filter_by_availability(raw_data, video_dir, num_classes, use_cached, cached_features):
    """
    Scans the dataset to find the top N classes with the most VALID video files.
    """
    print(f"\n🕵️  Scanning {len(raw_data)} classes for valid video files...")
    
    class_stats = []

    # 1. Count actual valid videos for every class
    for entry in raw_data:
        gloss = entry['gloss']
        valid_count = 0
        
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
        
        # Store tuple: (count, gloss, full_entry_data)
        class_stats.append({
            'count': valid_count,
            'gloss': gloss,
            'entry': entry
        })

    # 2. Sort by Count (Descending)
    class_stats.sort(key=lambda x: x['count'], reverse=True)

    # 3. Slice the Top N
    top_n_stats = class_stats[:num_classes]
    
    # 4. Print Detailed Metrics
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
    print(f"✅ Total videos in this subset: {total_videos}")
    print(f"📉 Cutoff threshold: Classes must have at least {top_n_stats[-1]['count']} videos.\n")

    # Return only the raw data entries for the selected classes
    return [x['entry'] for x in top_n_stats]