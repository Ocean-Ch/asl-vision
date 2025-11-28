from mock import NUM_MOCK_CHANNELS


JSON_PATH = "data/WLASL_v0.3.json"
VIDEO_DIR = "data/videos"
FEATURE_FILE = "data/all_features.pt"

BEST_EPOCH = 15 
MODEL_FILENAME = f"model_epoch_{BEST_EPOCH}.pth"

# ========== Configuration ==========
# Debug mode: when True, uses fake data and doesn't save the model (for testing shapes)
# Real mode: when False, uses real video data and saves the trained model

# only take the top N most frequent classes
NUM_CLASSES = 100
FRAMES_PER_CLIP = 32


# Default paths
JSON_PATH = "data/WLASL_v0.3.json"  # Path to dataset metadata JSON file
VIDEO_DIR = "data/videos"           # Directory containing video files
HISTORY_PATH = "results/history.json"        # Path to save history
MODEL_DIR = "results/models"              # Path to save models
HISTORY_GRAPH_PATH = "results/history.png"
LR = 1e-4                           # Learning rate

# GPU config (tuned for my setup - 4070 Ti)
BATCH_SIZE = 8
DEBUG_BATCH_SIZE = 2
EPOCHS = 100
DEBUG_EPOCHS = 1

# CPU config (tuned for my setup - 5800X3D)
NUM_WORKERS = 0

# Memory Config
PIN_MEMORY = False

FROZEN_CNN = False

USE_CACHED_FEATURES = False

# MS-ASL Dataset Configuration
MSASL_TRAIN_JSON = "data/MSASL_train.json"  # Path to MS-ASL train JSON file
MSASL_OUTPUT_DIR = "data/msasl_100_videos"   # Output directory for downloaded MS-ASL videos