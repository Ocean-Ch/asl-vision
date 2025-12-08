from mock import NUM_MOCK_CHANNELS

# ========== Model Architecture Configuration ==========
NUM_CLASSES = 100  # Number of top-N most frequent classes to use
FRAMES_PER_CLIP = 32  # Number of frames to extract per video
LSTM_HIDDEN = 256  # Size of LSTM hidden state
NUM_LSTM_LAYERS = 2  # Number of stacked LSTM layers
LSTM_DROPOUT = 0.3  # Dropout rate for LSTM layers
FROZEN_CNN = True  # Whether to freeze the MobileNetV2 backbone

# ========== Training Hyperparameters ==========
LR = 1e-4  # Learning rate
WEIGHT_DECAY = 1e-4  # L2 regularization weight decay
LABEL_SMOOTHING = 0.05  # Label smoothing factor for CrossEntropyLoss
EPOCHS = 150  # Number of training epochs
DEBUG_EPOCHS = 1  # Number of epochs in debug mode

# ========== Data Loading Configuration ==========
BATCH_SIZE = 32  # Batch size for training (tuned for RTX 4070 Ti)
DEBUG_BATCH_SIZE = 2  # Batch size in debug mode
NUM_WORKERS = 0  # Number of data loading workers (0 for Windows/OpenCV stability)
PIN_MEMORY = False  # Whether to pin memory for faster GPU transfer

# ========== Feature Caching Configuration ==========
USE_CACHED_FEATURES = True  # Whether to use pre-extracted features from disk
AUGMENT_ENABLED = True  # Whether to enable data augmentation (speed jitter + noise)

# ========== Dataset Paths ==========
FULL_DATA_PATH = "data/WLASL_v0.3.json"  # Full WLASL dataset JSON
TRAIN_JSON_PATH = "data/resplit/train_top_100.json"  # Training split JSON
VAL_JSON_PATH = "data/resplit/val_top_100.json"  # Validation split JSON
TEST_JSON_PATH = "data/resplit/test_top_100.json"  # Test split JSON
VIDEO_DIR = "data/videos"  # Directory containing video files (.mp4)
FEATURE_FILE = "data/all_features.pt"  # Path to cached feature file

# ========== Output Paths ==========
MODEL_DIR = "results/models"  # Directory to save model checkpoints
HISTORY_PATH = "results/history.json"  # Path to save training history JSON
HISTORY_GRAPH_PATH = "results/history.png"  # Path to save training history plot

# ========== Evaluation Configuration ==========
BEST_EPOCH = 116  # Best epoch number for evaluation
MODEL_FILENAME = f"model_epoch_{BEST_EPOCH}.pth"  # Filename of best model checkpoint

# ========== Dataset Resplitting Configuration ==========
RESPLIT_OUTPUT_DIR = "data/resplit"  # Directory to save resplit JSON files
TRAIN_SPLIT_PERCENT = 0.7  # Percentage of videos for training set
VAL_SPLIT_PERCENT = 0.18  # Percentage of videos for validation set
TEST_SPLIT_PERCENT = 0.12  # Percentage of videos for test set

# ========== MS-ASL Dataset Configuration (Optional) ==========
MSASL_TRAIN_JSON = "data/MSASL_train.json"  # Path to MS-ASL train JSON file
MSASL_OUTPUT_DIR = "data/msasl_100_videos"  # Output directory for downloaded MS-ASL videos
