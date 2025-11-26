import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ASLResNetLSTM
import argparse

# Debug mode: when True, the model will use fake data and not save the model
# Real mode: when False, the model will use the real data and save the model

# default config
JSON_PATH = "WLASL_v0.3.json"
VIDEO_DIR = "videos"
LR = 1e-4

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # Mac Metal Acceleration
    else:
        return torch.device("cpu")

def get_dataloader(split, debug_mode, batch_size):
    if debug_mode:
        print(f"⚠️ DEBUG MODE: Generating FAKE {split} data (No files needed)")
        # create 10 fake samples: (Batch, Frames, Channel, H, W)
        fake_data = torch.randn(10, 32, 3, 224, 224)
        fake_labels = torch.randint(0, 10, (10,))
        dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
        # mock the glosses attribute so code doesn't break
        dataset.glosses = [str(i) for i in range(10)]
        return DataLoader(dataset, batch_size=batch_size)
    else:
        # import dataset only when not in debug mode to avoid errors on Mac
        from dataset import WLASLDataset
        ds = WLASLDataset(JSON_PATH, VIDEO_DIR, split=split)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

def train(args):
    device = get_device()
    debug_mode = args.debug
    
    # adjust hyperparameters based on mode
    batch_size = 2 if debug_mode else 8
    epochs = 1 if debug_mode else 10

    print(f"🚀 Running on {device}. Debug Mode: {debug_mode}")

    train_loader = get_dataloader('train', debug_mode, batch_size)
    
    # initialize model
    # In Debug, we assume 10 classes. In Real, we calculate from dataset.
    num_classes = 10 if debug_mode else len(train_loader.dataset.glosses)
    
    model = ASLResNetLSTM(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting Training for {epochs} epoch(s)...")
    
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}")

    print("✅ Finished Successfully.")
    
    # save model only if NOT in debug mode
    if not debug_mode:
        save_path = "model_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with fake data')
    
    args = parser.parse_args()
    
    train(args)