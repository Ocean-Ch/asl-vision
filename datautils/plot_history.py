import json
import matplotlib.pyplot as plt
import argparse
import os
import sys
import config

def plot_history(file_path: str = config.HISTORY_PATH):
    """
    Loads a JSON history file and plots Loss and Accuracy.
    Ignores keys that don't exist or don't fit the 'loss'/'accuracy' naming convention.
    """
    
    # 1. Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # 2. Load the data
    try:
        with open(file_path, 'r') as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        sys.exit(1)

    # 3. Separate metrics based on keywords
    loss_metrics = {}
    acc_metrics = {}
    other_metrics = {}

    num_epochs = 0

    for key, values in history.items():
        # Update epoch count (based on the longest list found)
        if len(values) > num_epochs:
            num_epochs = len(values)

        key_lower = key.lower()
        
        if 'loss' in key_lower:
            loss_metrics[key] = values
        elif 'acc' in key_lower or 'top' in key_lower:
            acc_metrics[key] = values
        else:
            other_metrics[key] = values

    if num_epochs == 0:
        print("No data found in the history file.")
        return

    # 4. Create Subplots
    # We create 2 plots stacked vertically. 
    # If we only have loss or only accuracy, we handles that gracefully.
    has_loss = len(loss_metrics) > 0
    has_acc = len(acc_metrics) > 0

    if not has_loss and not has_acc:
        print("No standard 'loss' or 'accuracy' keys found to plot.")
        print(f"Found keys: {list(history.keys())}")
        return

    # Dynamic figure size based on what we are plotting
    rows = int(has_loss) + int(has_acc)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 5 * rows))
    
    # If there is only 1 row, axes is not a list, so we wrap it
    if rows == 1:
        axes = [axes]
    
    epochs_range = range(1, num_epochs + 1)
    current_ax_idx = 0

    # 5. Plot Loss
    if has_loss:
        ax = axes[current_ax_idx]
        for name, values in loss_metrics.items():
            # Handle cases where some lists might be shorter than others (crashed runs)
            x_vals = range(1, len(values) + 1)
            ax.plot(x_vals, values, marker='o', label=name)
        
        ax.set_title("Training & Validation Loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        current_ax_idx += 1

    # 6. Plot Accuracy
    if has_acc:
        ax = axes[current_ax_idx]
        for name, values in acc_metrics.items():
            x_vals = range(1, len(values) + 1)
            ax.plot(x_vals, values, marker='s', label=name)

        ax.set_title("Validation Accuracy (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Optional: Set y-axis to log scale if values are tiny (like 0.1%)
        # ax.set_yscale('log') 

    plt.tight_layout()
    
    # Save or Show
    output_img = config.HISTORY_GRAPH_PATH
    plt.savefig(output_img)
    print(f"✅ Plot saved to '{output_img}'")
    plt.show()

if __name__ == "__main__":
    # You can run this via command line: python plot_history.py history.json
    # Or just default to 'history.json' if no arg provided
    
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Create a dummy file if it doesn't exist for testing purposes
        target_file = config.HISTORY_PATH
        if not os.path.exists(target_file):
            print(f"No file specified. creating dummy '{target_file}' based on your data...")
            dummy_data = {
                "train_loss": [7.607, 7.586, 7.530, 7.445, 7.324, 7.167, 6.975, 6.777, 6.593],
                "val_acc_top1": [0.133, 0.0, 0.0, 0.044, 0.044, 0.0, 0.044, 0.044, 0.133],
                "val_acc_top5": [0.221, 0.177, 0.088, 0.088, 0.177, 0.443, 0.266, 0.355, 0.310]
            }
            with open(target_file, 'w') as f:
                json.dump(dummy_data, f)
    
    print(f"Plotting data from: {target_file}")
    plot_history(target_file)