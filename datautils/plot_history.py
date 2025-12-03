import json
import matplotlib.pyplot as plt
import os
import sys
import config

def plot_history(file_path: str = config.HISTORY_PATH):
    """
    Loads a JSON history file and plots Accuracy metrics.
    Ignores keys that don't fit the 'accuracy' naming convention.
    """
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    try:
        with open(file_path, 'r') as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")
        sys.exit(1)

    acc_metrics = {}

    for key, values in history.items():
        key_lower = key.lower()
        if 'acc' in key_lower or 'top' in key_lower:
            acc_metrics[key] = values

    if len(acc_metrics) == 0:
        print("No accuracy keys found to plot.")
        print(f"Found keys: {list(history.keys())}")
        return

    _, ax = plt.subplots(figsize=(10, 5))

    # 5. Plot Accuracy
    for name, values in acc_metrics.items():
        x_vals = range(1, len(values) + 1)
        ax.plot(x_vals, values, marker='s', label=name)

    ax.set_title("Training vs. Validation Accuracy (Top-1 and Top-5) (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Epochs")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

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