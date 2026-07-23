import os
from torchvision.datasets import CIFAR10, STL10, ImageFolder

def get_dataset(name="cifar10", data_dir="./data", transform=None, train=True, is_pretraining=True):
    """
    Factory function to fetch datasets cleanly for SSL pre-training and linear evaluation.
    Supports CIFAR10, STL10, and Custom ImageFolder datasets.
    """
    name = name.lower()
    
    if name == "cifar10":
        return CIFAR10(root=data_dir, train=train, transform=transform, download=True)
        
    elif name == "stl10":
        if is_pretraining:
            # SimCLR pre-training uses the 100,000 unlabeled images
            split = "unlabeled" if train else "test"
        else:
            # Linear evaluation uses the labeled 'train' (5k) and 'test' (8k) splits
            split = "train" if train else "test"
            
        return STL10(root=data_dir, split=split, transform=transform, download=True)
        
    elif name == "custom":
        # Check if train/test subdirectories exist inside data_dir
        sub_dir = "train" if train else "test"
        target_path = os.path.join(data_dir, sub_dir)
        
        if os.path.exists(target_path):
            return ImageFolder(root=target_path, transform=transform)
        else:
            # Fallback to base data_dir if no train/test subfolders exist
            return ImageFolder(root=data_dir, transform=transform)
            
    else:
        raise ValueError(f"Unsupported dataset: '{name}'. Choose from 'cifar10', 'stl10', or 'custom'.")