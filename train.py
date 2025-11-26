import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ASLResNetLSTM
import time

# --- CONFIGURATION ---
# Set DEBUG = True for fake training on mac without data
# Set DEBUG = False when running on Windows with real data
DEBUG = True 

# Paths (Only used if DEBUG=False)
JSON_PATH = "WLASL_v0.3.json"
VIDEO_DIR = "videos"

# hyperparams
BATCH_SIZE = 8 if not DEBUG else 2
LR = 1e-4
EPOCHS = 10 if not DEBUG else 1

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # mac metal acceleration
    else:
        return torch.device("cpu")

def get_dataloader(split):
    if DEBUG:
        print(f"⚠️ DEBUG MODE: Generatng FAKE {split} data (No files needed)")
        # create 10 fake samples: (Batch, Frames, Channel, H, W)
        fake_data = torch.randn(10, 32, 3, 224, 224)
        fake_labels = torch.randint(0, 10, (10,))
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        # mock the glosses attribute so code doesn't break
        dataset.glosses = [str(i) for i in range(10)]
        return DataLoader(dataset, batch_size=BATCH_SIZE)
    else:
        from dataset import WLASLDataset
        ds = WLASLDataset(JSON_PATH, VIDEO_DIR, split=split)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

def train():
    device = get_device()
    print(f"🚀 Running on {device}. DEBUG={DEBUG}")

    train_loader = get_dataloader('train')
    
    # initialize model
    # In Debug, we assume 10 classes. In Real, we calculate from dataset.
    num_classes = 10 if DEBUG else len(train_loader.dataset.glosses)
    
    model = ASLResNetLSTM(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

    print("✅ Finished Successfully.")
    if not DEBUG:
        torch.save(model.state_dict(), f"model_final.pth")

if __name__ == "__main__":
    train()