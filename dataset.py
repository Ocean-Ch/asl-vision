import torch
from torch.utils.data import Dataset
import cv2
import json
import os
import numpy as np
from torchvision import transforms
from typing import Tuple


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
    def __init__(self, json_path: str, video_dir: str, frames_per_clip: int = 32, split: str = 'train'):
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
        
        # load JSON metadata file
        # contains entries with 'gloss' (sign language word) and 'instances' (video examples)
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.samples = []  # store all valid video samples
        
        # parse JSON to extract video paths and corresponding labels
        for entry in raw_data:
            # extract gloss from entry
            gloss = entry['gloss']
            for inst in entry['instances']:
                # only use videos from the specified split (train/val/test)
                if inst['split'] == split:
                    video_id = inst['video_id']
                    path = os.path.join(video_dir, f"{video_id}.mp4")
                    # only add if video file exists on disk
                    if os.path.exists(path):
                        self.samples.append({
                            'video_path': path,
                            'gloss': gloss
                        })
                    else:
                        # log file not found
                        print(f"Video file not found: {path}")

        # create vocabulary: map each unique gloss to a numeric ID (dupes removed, sorted)
        self.glosses = sorted(list(set(s['gloss'] for s in self.samples)))
        # create reverse mapping: gloss -> ID (e.g., "hello" -> 0, "thank you" -> 1)
        self.gloss_to_id = {g: i for i, g in enumerate(self.glosses)}
        
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        # set up image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # np array -> PIL Image (required for torchvision transforms)
            transforms.Resize((224, 224)),  # resize to 224x224 (standard size for pretrained models like MobileNet)
            transforms.ToTensor(),  # PIL Image -> PyTorch tensor and scale pixel values to [0, 1]
            
            # normalize using ImageNet dataset statistics (mean and std for RGB channels)
            # helps training by centering the data around 0 and scaling it
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        """
        Returns the number of samples in the dataset. 
        Required by PyTorch's Dataset interface to know how many samples are available.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Loads and returns a single video sample.
        Called automatically by DataLoader for each sample in a batch.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: (video_tensor, label_id)
                - video_tensor: PyTorch tensor of shape (frames, 3, 224, 224)
                  representing the video frames (uniformly sampled)
                - label_id: Integer ID of the sign language word (gloss)
        
        Process:
        1. Open video file using OpenCV
        2. Uniformly sample frames across the video
        3. Apply self.transform to each frame (resize, normalize)
        4. Flatten frames into a single tensor
        5. Return frames tensor and corresponding label ID
        """
        item = self.samples[idx]
        
        # open video file using OpenCV's VideoCapture
        cap = cv2.VideoCapture(item['video_path'])
        
        # get total number of frames in the video for uniform sampling
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # handle broken or empty videos
        # return a zero tensor if video is invalid
        # shape: (frames, channels=3, height=224, width=224)
        if total_frames < 1:
            return torch.zeros((self.frames_per_clip, 3, 224, 224)), self.gloss_to_id[item['gloss']]

        # uniform sampling to evenly distribute frame indices across the video
        # np.linspace creates evenly spaced numbers from start to end
        # ex: total_frames=100 and frames_per_clip=32, then indices = [0, 3, 6, 9, ...]
        indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)
        frames = []
        
        # extract and process each sampled frame
        for i in indices:
            # seek to the specific frame position in the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            
            # read the frame at that position (ret = True -> frame read successfully)
            ret, frame = cap.read()
            if ret:
                # OpenCV reads images in BGR, convert to RGB for PyTorch
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # apply preprocessing: resize, normalize, convert to tensor
                frame = self.transform(frame)
                frames.append(frame)
            else:
                # if frame couldn't be read, add a zero tensor as placeholder
                # shape: (channels=3, height=224, width=224)
                frames.append(torch.zeros(3, 224, 224))
        
        # release video file to free up resources
        cap.release()
        
        # stack all frames into a single tensor
        # torch.stack combines a list of tensors along a new dimension
        # shape: (frames_per_clip, 3, 224, 224)
        # also return the numeric label ID for this video
        return torch.stack(frames), self.gloss_to_id[item['gloss']]