from torchvision.datasets import CIFAR10, STL10, ImageFolder

def get_dataset(name="cifar10", data_dir="./data", transform=None, train=True, stl10_split="unlabeled"):
    """
    Factory function to fetch datasets cleanly.
    Supports CIFAR10, STL10 (including the SSL 'unlabeled' split), and Custom image folders.
    """
    name = name.lower()
    if name == "cifar10":
        return CIFAR10(root=data_dir, train=train, transform=transform, download=True)
    elif name == "stl10":
        # SSL pre-training typically utilizes the 100k 'unlabeled' split on STL-10
        split = stl10_split if train else "test"
        return STL10(root=data_dir, split=split, transform=transform, download=True)
    elif name == "custom":
        return ImageFolder(root=data_dir, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: '{name}'. Choose from 'cifar10', 'stl10', or 'custom'.")