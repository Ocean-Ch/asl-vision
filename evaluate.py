import torch
from torch.utils.data import DataLoader
from model import ASLResNetLSTM
from dataset import WLASLDataset
from train import HISTORY_PATH, JSON_PATH, VIDEO_DIR, MODEL_PATH, BATCH_SIZE

# CONFIG
MODEL_PATH = "model_epoch_15.pth" # should be the best epoch file from training

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on {device}")

    # Load TEST split (Strictly for final report)
    test_dataset = WLASLDataset(JSON_PATH, VIDEO_DIR, split='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    
    num_classes = len(test_dataset.glosses)
    
    # Load Model
    model = ASLResNetLSTM(num_classes=num_classes).to(device)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded weights from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Run training first!")
        return

    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    print("Starting inference...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Calculate Top-1 and Top-5
            # Get the top 5 class indices
            _, top5_preds = torch.topk(outputs, 5, dim=1)
            
            # Check matches
            labels_reshaped = labels.view(-1, 1)
            correct_matrix = top5_preds == labels_reshaped
            
            # Top-1 is just the first column
            top1_correct += correct_matrix[:, 0].sum().item()
            
            # Top-5 is if ANY column is True
            top5_correct += correct_matrix.sum().item()
            
            total_samples += labels.size(0)
            
            if i % 10 == 0:
                print(f"Processed batch {i}...")

    acc1 = 100 * top1_correct / total_samples
    acc5 = 100 * top5_correct / total_samples
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS")
    print("="*30)
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()