import torch
from torch.utils.data import Dataset

NUM_MOCK_VIDEOS = 10
NUM_MOCK_FRAMES = 32
NUM_MOCK_CHANNELS = 3
NUM_MOCK_HEIGHT = 224
NUM_MOCK_WIDTH = 224

def mock_data() -> Dataset:
    """
    Create a mock dataset for testing.
    
    Returns:
        Dataset: A mock dataset with 10 fake video samples and labels
    """
    # create 10 fake video samples
    # torch.randn generates random numbers from a normal distribution
    # shape: (10, 32, 3, 224, 224) = 10 videos, 32 frames each, RGB, 224x224
    fake_data = torch.randn(NUM_VIDEOS, NUM_FRAMES, NUM_CHANNELS, HEIGHT, WIDTH)
    
    # create 10 fake labels (random integers from 0 to 9)
    fake_labels = torch.randint(0, NUM_VIDEOS, (NUM_VIDEOS,))
    
    # tensorDataset wraps tensors into a dataset to pair each data sample with its corresponding label
    dataset = torch.utils.data.TensorDataset(fake_data, fake_labels)
    
    # mock the glosses attribute (list of class names)
    dataset.glosses = [str(i) for i in range(NUM_VIDEOS)]

    return dataset
