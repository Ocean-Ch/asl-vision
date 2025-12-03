from mock import NUM_MOCK_CHANNELS

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
TRAIN_JSON_PATH = "data/resplit/train_top_100.json"  # Path to dataset metadata JSON file
VAL_JSON_PATH = "data/resplit/val_top_100.json"  # Path to dataset metadata JSON file
TEST_JSON_PATH = "data/resplit/test_top_100.json"  # Path to dataset metadata JSON file
VIDEO_DIR = "data/videos"           # Directory containing video files
HISTORY_PATH = "results/history.json"        # Path to save history
MODEL_DIR = "results/models"              # Path to save models
HISTORY_GRAPH_PATH = "results/history.png"
LR = 1e-4                           # Learning rate
WEIGHT_DECAY = 1e-4
LSTM_DROPOUT = 0.5

# GPU config (tuned for my setup - 4070 Ti)
BATCH_SIZE = 64
DEBUG_BATCH_SIZE = 2
EPOCHS = 150
DEBUG_EPOCHS = 1

# CPU config (tuned for my setup - 5800X3D)
NUM_WORKERS = 0

# Memory Config
PIN_MEMORY = False

FROZEN_CNN = True

USE_CACHED_FEATURES = True

AUGMENT_ENABLED = True

# MS-ASL Dataset Configuration
MSASL_TRAIN_JSON = "data/MSASL_train.json"  # Path to MS-ASL train JSON file
MSASL_OUTPUT_DIR = "data/msasl_100_videos"   # Output directory for downloaded MS-ASL videos

# Dataset Resplitting Configuration
RESPLIT_OUTPUT_DIR = "data/resplit"  # Directory to save resplit JSON files
TRAIN_SPLIT_PERCENT = 0.7  # Percentage of videos for training set
VAL_SPLIT_PERCENT = 0.15   # Percentage of videos for validation set
TEST_SPLIT_PERCENT = 0.15  # Percentage of videos for test set