JSON_PATH = "data/WLASL_v0.3.json"
VIDEO_DIR = "data/videos"
FEATURE_FILE = "data/all_features.pt"

BEST_EPOCH = 15 
MODEL_FILENAME = f"model_epoch_{BEST_EPOCH}.pth"

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
BATCH_SIZE = 32
DEBUG_BATCH_SIZE = 2
EPOCHS = 15
DEBUG_EPOCHS = 1

# CPU config (tuned for my setup - 5800X3D)
NUM_WORKERS = 0

# Memory Config
PIN_MEMORY = False