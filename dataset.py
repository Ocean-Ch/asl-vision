import torch
from torch.utils.data import Dataset
import cv2
import json
import os
import numpy as np
from torchvision import transforms

class WLASLDataset(Dataset):
    def __init__(self, json_path, video_dir, frames_per_clip=32, split='train'):
        self.video_dir = video_dir
        self.frames_per_clip = frames_per_clip
        
        # load json
        with open(json_path, 'r') as f:
            raw_data = json.load(f)
            
        self.samples = []
        
        # parse json
        for entry in raw_data:
            gloss = entry['gloss']
            for inst in entry['instances']:
                if inst['split'] == split:
                    video_id = inst['video_id']
                    # verify file exists
                    path = os.path.join(video_dir, f"{video_id}.mp4")
                    if os.path.exists(path):
                        self.samples.append({
                            'video_path': path,
                            'gloss': gloss
                        })

        # create vocabulary
        self.glosses = sorted(list(set(s['gloss'] for s in self.samples)))
        self.gloss_to_id = {g: i for i, g in enumerate(self.glosses)}
        
        # transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        cap = cv2.VideoCapture(item['video_path'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # handle broken video
        if total_frames < 1:
            return torch.zeros((self.frames_per_clip, 3, 224, 224)), self.gloss_to_id[item['gloss']]

        # uniform sampling
        indices = np.linspace(0, total_frames - 1, self.frames_per_clip).astype(int)
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
        return torch.stack(frames), self.gloss_to_id[item['gloss']]