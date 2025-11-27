"""
Training script for the ASL recognition model.

This script handles:
1. Setting up the training environment (device selection: CPU/GPU/MPS)
2. Loading and preparing data (with debug mode for testing)
3. Initializing the model, optimizer, and loss function
4. Running the training loop (forward pass, backward pass, parameter updates)
5. Saving the trained model

Usage:
    python train.py              # Train with real data
    python train.py --debug      # Train with fake data (for testing without video files)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ASLResNetLSTM
import argparse
from mock import NUM_MOCK_VIDEOS
import json


# ========== Configuration ==========
# Debug mode: when True, uses fake data and doesn't save the model (for testing shapes)
# Real mode: when False, uses real video data and saves the trained model

# Default paths and hyperparameters
JSON_PATH = "data/WLASL_v0.3.json"  # Path to dataset metadata JSON file
VIDEO_DIR = "data/videos"           # Directory containing video files
HISTORY_PATH = "results/history.json"        # Path to save history
MODEL_DIR = "results/models"              # Path to save models
LR = 1e-4                           # Learning rate

# GPU config (tuned for my setup - 4070 Ti)
BATCH_SIZE = 64
DEBUG_BATCH_SIZE = 2
EPOCHS = 15
DEBUG_EPOCHS = 1

# CPU config (tuned for my setup - 5800X3D)
NUM_WORKERS = 8

def get_device() -> torch.device:
    """
    Automatically select the best available device for training.
    
    Priority order:
    1. CUDA (NVIDIA GPU) - fastest, if available
    2. MPS (Apple Silicon GPU) - for Macs with M1/M2/M3 chips
    3. CPU - fallback if no GPU available
    
    Returns:
        torch.device: The selected device object
    """
    if torch.cuda.is_available():
        # NVIDIA GPU
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU (M1/M2/M3 Macs) - Metal Performance Shaders
        return torch.device("mps")
    else:
        # fallback to CPU (slower but always available)
        return torch.device("cpu")

def get_dataloader(split: str, debug_mode: bool, batch_size: int) -> DataLoader:
    """
    Create a DataLoader for training or validation.
    
    DataLoader is a PyTorch utility that:
    - batches samples together
    - shuffles data (for training)
    - loads data in parallel using multiple workers
    - handles mem management efficiently
    
    Args:
        split (str): Dataset split - 'train', 'val', or 'test'
        debug_mode (bool): if True, use fake data instead of real videos
        batch_size (int): number of samples per batch
    
    Returns:
        DataLoader: PyTorch DataLoader object that yields batches of (inputs, labels) - shape (batch_size, frames, channels, height, width) and (batch_size,)
    """
     # debug mode: create fake data for testing without needing actual video files
    if debug_mode:
        print(f"⚠️ DEBUG MODE: Generating FAKE {split} data (No files needed)")
        # create dataloader with mock data
        from mock import mock_data
        return DataLoader(mock_data(), batch_size=batch_size)
    
    # real mode: load actual video dataset
    # Import here to avoid errors if OpenCV/video libraries aren't available in debug mode
    from dataset import WLASLDataset
    
    # create dataset instance
    ds = WLASLDataset(JSON_PATH, VIDEO_DIR, split=split)
    
    # create DataLoader with real data
    # shuffle=True: randomize order of samples (important for training)
    # num_workers=4: use 4 parallel processes to load data
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def train(args):
    """
    Main training function.
    
    This function orchestrates the entire training process:
    1. Sets up device and hyperparameters
    2. Loads data
    3. Initializes model, optimizer, and loss function
    4. Runs training loop (forward pass, backward pass, update weights)
    5. Saves the trained model
    
    Args:
        args: Command-line arguments (contains debug flag)
    """
    # get the best available device
    device = get_device()
    debug_mode = args.debug
    
    # adjust hyperparameters based on mode
    # debug mode uses smaller values for faster testing
    batch_size = DEBUG_BATCH_SIZE if debug_mode else BATCH_SIZE
    epochs = DEBUG_EPOCHS if debug_mode else EPOCHS

    print(f"🚀 Running on {device}. Debug Mode: {debug_mode}")

    # load training data
    train_loader = get_dataloader('train', debug_mode, batch_size)

    # load validation data (for monitoring loss during training)
    val_loader = get_dataloader('val', debug_mode, batch_size)
    
    # determine number of classes (sign language words)
    # debug mode: assume 10 classes (matches fake data)
    # real mode: count unique glosses in the dataset
    num_classes = NUM_MOCK_VIDEOS if debug_mode else len(train_loader.dataset.glosses)
    
    # initialize model (model and data must be on the same device)
    model = ASLResNetLSTM(num_classes=num_classes).to(device)
    
    # Initialize Adam (adaptive learning rate optimizer)
    # Combines momentum and RMSprop updates
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # CrossEntropyLoss since we are doing multi-class classification
    criterion = nn.CrossEntropyLoss()

    
    # history tracking for plotting later
    history = {
        'train_loss': [],
        'val_acc_top1': [],
        'val_acc_top5': [],
    }

    print(f"Starting Training for {epochs} epoch(s)...")

    # ========== Training Loop ==========
    # epoch = one complete pass through the entire dataset
    for epoch in range(epochs):
        # set model to training mode to enable dropout and batch norm
        model.train()

        # used to compute average loss per epoch
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            # move data to the same device as the model
            # inputs: video tensors - (batch_size, frames, channels, height, width)
            # labels: integer class IDs - (batch_size,)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero out grads from previous iteration
            # pytorch accumulates gradients by default, so need to reset them
            optimizer.zero_grad()
            
            # forward pass: run data through the model
            # internally calls model.forward()
            # outputs: raw logits - (batch_size, num_classes)
            outputs = model(inputs)
            
            # get cross entropy for outputs vs labels
            loss = criterion(outputs, labels)
            
            # backprop to compute gradients
            loss.backward()
            
            # update model params (weights)
            # uses computed gradients to adjust weights in the direction that reduces loss
            # the optimizer (Adam) determines the step size based on learning rate and gradients
            optimizer.step()

            running_loss += loss.item()
            
            # print progress every 10 batches
            # loss.item() converts the tensor to a Python float
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}")
            
            validate_and_checkpoint(model, val_loader, device, epoch, history, debug_mode)

    print("✅ Finished Successfully.")
    
    # save model weights only if NOT in debug mode
    # state_dict() contains all the learned parameters (weights and biases)
    # so we can load the trained model later without retraining
    if not debug_mode:
        save_path = "model_final.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Model saved to {save_path}")

def validate_and_checkpoint(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    history: dict,
    debug_mode: bool = False
) -> None:
    """
    Validate the model and save checkpoints.
    """
    model.eval()
    top1_correct, top5_correct, total_samples = 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # compute top1 and top5 accuracy

            # maxk: (batch_size, 5) - indices of the top 5 predictions
            _, maxk = torch.topk(outputs, 5, dim=1)

            # labels_resize: (batch_size, 1) - true labels
            labels_resize = labels.view(-1, 1)

            # correct_matrix: (batch_size, 5) - correct_matrix[i, j] = 1 if the j-th prediction of the i-th sample is correct
            correct_matrix = maxk == labels_resize

            # total top1 correct is just the sum of the first column of the correct_matrix
            top1_correct += correct_matrix[:, 0].sum().item()
            # total top5 correct is the sum of all the correct predictions
            top5_correct += correct_matrix.sum().item()
            # total samples is the number of samples in the validation set
            total_samples += labels.size(0)

    val_acc1 = 100 * top1_correct / total_samples
    val_acc5 = 100 * top5_correct / total_samples

    history['val_acc_top1'].append(val_acc1)
    history['val_acc_top5'].append(val_acc5)

    print(f"-------------------------------- Epoch {epoch+1} Summary --------------------------------")
    print(f"Train Loss: {history['train_loss'][-1]:.4f} | Val Top-1: {val_acc1:.2f}% | Val Top-5: {val_acc5:.2f}%")

    # save model checkpoint
    if not debug_mode:
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        with open("history.json", "w") as f:
            json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    
    # add --debug flag
    # action='store_true' means: if flag is present, set to True; otherwise False
    # Example: python train.py --debug  (debug_mode = True)
    #          python train.py          (debug_mode = False)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with fake data')
    
    args = parser.parse_args()
    
    # start training with parsed args
    train(args)