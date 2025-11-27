"""
Evaluation script for the ASL recognition model.

This script handles:
1. Loading a trained model checkpoint from disk
2. Running inference on the test dataset
3. Computing Top-1 and Top-5 accuracy metrics
4. Reporting final test results for evaluation

Usage:
    python evaluate.py
    
Note: This script evaluates on the TEST split only, which should be reserved
for final evaluation and not used during training/validation.
"""

import torch
from torch.utils.data import DataLoader
from model import ASLResNetLSTM
from dataset import WLASLDataset
import os
from train import JSON_PATH, VIDEO_DIR, MODEL_DIR, BATCH_SIZE

# ========== Configuration ==========
# change this number to match the best epoch you see in your training logs
BEST_EPOCH = 15 
MODEL_FILENAME = f"model_epoch_{BEST_EPOCH}.pth"

def evaluate() -> None:
    """
    Main evaluation function.
    
    This function orchestrates the evaluation process:
    1. Loads test dataset
    2. Initializes model architecture
    3. Loads trained weights from checkpoint
    4. Runs inference on test set
    5. Computes Top-1 and Top-5 accuracy metrics
    6. Reports final results
    
    The test split should only be used for final evaluation and not during
    training or hyperparameter tuning to ensure unbiased performance estimates.
    """
    # select device (CUDA GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")

    # load TEST split (strictly for final report - not used during training)
    test_dataset = WLASLDataset(JSON_PATH, VIDEO_DIR, split='test')
    # shuffle=False: maintain consistent order for reproducibility
    # num_workers=4: use 4 parallel processes to load data
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # determine number of classes from dataset vocabulary
    num_classes = len(test_dataset.glosses)
    
    # initialize model architecture (must match training configuration)
    model = ASLResNetLSTM(num_classes=num_classes).to(device)
    
    # construct full path to model checkpoint
    # example: results/models/model_epoch_15.pth
    weight_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    
    # load trained weights from checkpoint
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}...")
        try:
            # load_state_dict loads the saved weights into the model
            # map_location=device ensures weights are loaded to the correct device
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print("✅ Weights loaded successfully.")
        except RuntimeError as e:
            print(f"❌ Error loading weights: {e}")
            print("Tip: Ensure the model architecture in model.py matches the saved checkpoint.")
            return
    else:
        print(f"❌ Error: Could not find {weight_path}")
        print(f"Available models in {MODEL_DIR}:")
        if os.path.exists(MODEL_DIR):
            print(os.listdir(MODEL_DIR))
        return

    # set model to evaluation mode
    # disables dropout and batch normalization updates during inference
    model.eval()
    
    # initialize accuracy counters
    top1_correct = 0  # count of samples where top prediction matches label
    top5_correct = 0  # count of samples where true label is in top 5 predictions
    total_samples = 0  # total number of samples processed
    
    print("Starting inference on TEST set...")
    # torch.no_grad() disables gradient computation for efficiency during inference
    # we don't need gradients since we're not training
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            # move data to the same device as the model
            # inputs: video tensors - shape (batch_size, frames, channels, height, width)
            # labels: integer class IDs - shape (batch_size,)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # forward pass: get model predictions
            # outputs: raw logits - shape (batch_size, num_classes)
            outputs = model(inputs)
            
            # calculate Top-1 and Top-5 predictions
            # torch.topk returns the top k values and their indices
            # _: top k values (we don't need these, only the indices)
            # top5_preds: indices of top 5 predictions - shape (batch_size, 5)
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            
            # reshape labels to compare with predictions
            # labels_reshaped: shape (batch_size, 1) for broadcasting
            labels_reshaped = labels.view(-1, 1)
            
            # create boolean matrix: True where prediction matches label
            # correct_matrix: shape (batch_size, 5)
            # correct_matrix[i, j] = True if the j-th top prediction for sample i matches the true label
            correct_matrix = top5_preds == labels_reshaped
            
            # Top-1 accuracy: count matches in first column (top prediction)
            # correct_matrix[:, 0] gives first column, sum() counts True values
            top1_correct += correct_matrix[:, 0].sum().item()
            
            # Top-5 accuracy: count matches in any column (any of top 5 predictions)
            # sum() counts all True values across all columns
            top5_correct += correct_matrix.sum().item()
            
            # track total number of samples processed
            total_samples += labels.size(0)
            
            # print progress every 10 batches
            if i % 10 == 0:
                print(f"Processed batch {i}/{len(test_loader)}...")

    # calculate accuracy percentages
    acc1 = 100 * top1_correct / total_samples
    acc5 = 100 * top5_correct / total_samples
    
    # print final results
    print("\n" + "="*30)
    print("FINAL TEST RESULTS (For Report)")
    print("="*30)
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
    print("="*30)

if __name__ == "__main__":
    """
    Entry point when script is run directly.
    Starts the evaluation process.
    """
    evaluate()