import torch
from torch.utils.data import DataLoader
from model import ASLResNetLSTM
from dataset import WLASLDataset
import os
import json
from utils import get_accuracy_counts
from device import get_device
import config

def evaluate() -> None:
    device = get_device()
    print(f"Running evaluation on {device}")

    # =========================================================
    # 1. FIX: Load the MASTER vocabulary first
    # =========================================================
    # We need to know exactly what classes the model was trained on, in what order.
    # The safest way is to load the JSON and sort the top N glosses exactly
    # as the training set did.
    
    print(f"Loading vocabulary from {config.TRAIN_JSON_PATH}...")
    with open(config.TRAIN_JSON_PATH, 'r') as f:
        raw_data = json.load(f)
        
    # Extract ALL glosses and sort them (matches Train logic)
    # If your train logic filtered by top_n, we must do that here too.
    # Assuming config.NUM_CLASSES controls the filter:
    
    # Simple counting to find top N (Must match training logic exactly)
    # If you have a saved 'vocab.json', load that instead. 
    # Otherwise, we reconstruct the Top N list:
    all_glosses = [entry['gloss'] for entry in raw_data]
    # Note: If your training set logic was complex (checking file availability),
    # this might slightly drift. 
    # BETTER OPTION: If you saved the model, you hopefully saved the class list.
    # If not, we assume standard sorting of the top N.
    
    # Quick dirty fix if you don't have a vocab file:
    # Load the TRAIN dataset just to get its gloss list
    print("⏳ Loading Dummy Train Dataset to extract vocabulary...")
    train_ref = WLASLDataset(
        config.TRAIN_JSON_PATH, 
        config.VIDEO_DIR, 
        split='train', 
        num_classes=config.NUM_CLASSES
    )
    correct_glosses = train_ref.glosses
    num_classes = len(correct_glosses)
    print(f"✅ Vocabulary fixed. {num_classes} classes detected.")
    
    # =========================================================

    # 2. Load TEST split with the FORCED vocabulary
    # You need to modify WLASLDataset __init__ to accept a 'gloss_list' argument
    # OR, we just hack it by manually setting it after init (if your class allows)
    
    test_dataset = WLASLDataset(
        config.TEST_JSON_PATH, 
        config.VIDEO_DIR, 
        split='test',
        use_cached_features=config.USE_CACHED_FEATURES,
        num_classes=config.NUM_CLASSES 
    )
    
    # ⚠️ OVERWRITE the test dataset's internal vocabulary with the correct one
    test_dataset.glosses = correct_glosses
    test_dataset.gloss_to_id = {g: i for i, g in enumerate(correct_glosses)}
    
    # ---------------------------------------------------------

    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    model = ASLResNetLSTM(
        num_classes=num_classes,
        frozenCNN=config.FROZEN_CNN,
        expect_features=config.USE_CACHED_FEATURES
    ).to(device)
    
    weight_path = os.path.join(config.MODEL_DIR, config.MODEL_FILENAME)
    
    if os.path.exists(weight_path):
        print(f"Loading weights from {weight_path}...")
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print(f"❌ Error: Could not find {weight_path}")
        return

    model.eval()
    
    top1_correct = 0
    top5_correct = 0
    total_samples = 0
    
    print("Starting inference on TEST set...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            batch_top1, batch_top5 = get_accuracy_counts(outputs, labels)
            
            top1_correct += batch_top1
            top5_correct += batch_top5
            total_samples += labels.size(0)
            
            if i % 10 == 0:
                print(f"Processed batch {i}/{len(test_loader)}...")

    acc1 = 100 * top1_correct / total_samples
    acc5 = 100 * top5_correct / total_samples
    
    print("\n" + "="*30)
    print("FINAL TEST RESULTS (For Report)")
    print("="*30)
    print(f"Top-1 Accuracy: {acc1:.2f}%")
    print(f"Top-5 Accuracy: {acc5:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()