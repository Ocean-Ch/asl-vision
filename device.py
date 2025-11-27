import torch

def get_device() -> torch.device:
    """
    Automatically select the best available device for training.
    
    Priority order:
    1. CUDA (NVIDIA GPU) - fastest, if available
    2. MPS (Apple Silicon GPU) - for Macs with M1/M2/M3 chips
    3. CPU - fallback if no GPU available
    
    Returns:
        torch.device: The selected device object
    """
    if torch.cuda.is_available():
        # NVIDIA GPU
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Apple Silicon GPU (M1/M2/M3 Macs) - Metal Performance Shaders
        return torch.device("mps")
    else:
        # fallback to CPU (slower but always available)
        return torch.device("cpu")