import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Ensures full reproducibility across runs.
    Safe for both CPU and CUDA hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed}")