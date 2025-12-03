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
import os
from utils import get_accuracy_counts
from device import get_device
import config

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

    if split == 'train':
        json_path = config.TRAIN_JSON_PATH
    elif split == 'val':
        json_path = config.VAL_JSON_PATH
    elif split == 'test':
        json_path = config.TEST_JSON_PATH
    
    # create dataset instance
    ds = WLASLDataset(json_path, config.VIDEO_DIR, split=split, use_cached_features=config.USE_CACHED_FEATURES, num_classes=None)
    
    # create DataLoader with real data
    # shuffle=True: randomize order of samples (important for training)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

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
    batch_size = config.DEBUG_BATCH_SIZE if debug_mode else config.BATCH_SIZE
    epochs = config.DEBUG_EPOCHS if debug_mode else config.EPOCHS

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
    model = ASLResNetLSTM(num_classes=num_classes, frozenCNN=config.FROZEN_CNN, expect_features=config.USE_CACHED_FEATURES).to(device)
    
    # Initialize Adam (adaptive learning rate optimizer)
    # Combines momentum and RMSprop updates
    optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    
    # CrossEntropyLoss since we are doing multi-class classification
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    
    # history tracking for plotting later
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc_top1': [],
        'val_acc_top5': [],
    }

    # ensure output dirs exist
    if not debug_mode:
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.HISTORY_PATH), exist_ok=True)

    print(f"Starting Training for {epochs} epoch(s)...")

    # ========== Training Loop ==========
    # epoch = one complete pass through the entire dataset
    for epoch in range(epochs):
        # set model to training mode to enable dropout and batch norm
        model.train()

        # used to compute average loss per epoch
        running_loss = 0.0

        train_correct = train_total = 0

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

            with torch.no_grad():
                # assuming get_accuracy_counts returns (top1_count, top5_count)
                batch_acc_1, _ = get_accuracy_counts(outputs, labels) 
                train_correct += batch_acc_1
                train_total += labels.size(0)
            
            # print progress every 10 batches
            # loss.item() converts the tensor to a Python float
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}], Loss: {loss.item():.4f}")
            
        # calculate and store average train loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        
        validate_and_checkpoint(model, val_loader, device, epoch, history, debug_mode)

    print("✅ Finished Successfully.")
    
    # save model weights only if NOT in debug mode
    # state_dict() contains all the learned parameters (weights and biases)
    # so we can load the trained model later without retraining
    if not debug_mode:
        save_path = os.path.join(config.MODEL_DIR, "model_final.pth")
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

            # compute top1 and top5 accuracy counts for this batch
            # uses utility function to avoid code duplication
            batch_top1, batch_top5 = get_accuracy_counts(outputs, labels)
            
            # accumulate counts across all batches
            top1_correct += batch_top1
            top5_correct += batch_top5
            
            # track total number of samples processed
            total_samples += labels.size(0)

    val_acc1 = 100 * top1_correct / total_samples
    val_acc5 = 100 * top5_correct / total_samples

    history['val_acc_top1'].append(val_acc1)
    history['val_acc_top5'].append(val_acc5)

    print(f"-------------------------------- Epoch {epoch+1} Summary --------------------------------")
    print(f"Train Loss: {history['train_loss'][-1]:.4f} | Val Top-1: {val_acc1:.2f}% | Val Top-5: {val_acc5:.2f}%")

    if not debug_mode:
        # save model checkpoint and history
        torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"model_epoch_{epoch+1}.pth"))
        with open(config.HISTORY_PATH, "w") as f:
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