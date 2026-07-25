import random
import torch
import torchvision.transforms as transforms
from PIL import ImageFilter


class GaussianBlur(object):
    """
    SimCLR Gaussian Blur implementation with a 50% probability of being applied[cite: 40].
    """
    def __init__(self, sigma=(0.1, 2.0)):  
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < 0.5:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


# Dataset Normalization Statistics
DATASET_STATS = {
    "cifar10": {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2470, 0.2435, 0.2616],
    },
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


class SimCLRDataTransform(object):
    """
    Generates two stochastic augmented views (x_i, x_j) for contrastive pre-training[cite: 40].
    """
    def __init__(self, input_shape=32, s=1.0, dataset_name="cifar10"):
        stats = DATASET_STATS.get(dataset_name.lower(), DATASET_STATS["imagenet"])

        # s is the color jitter strength parameter (default 1.0 as specified in paper)
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        transform_list = [
            # 1. Random Resized Crop
            transforms.RandomResizedCrop(size=input_shape, scale=(0.08, 1.0)),
            # 2. Horizontal Flip
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. Color Jitter (80% probability)
            transforms.RandomApply([color_jitter], p=0.8),
            # 4. Grayscale (20% probability)
            transforms.RandomGrayscale(p=0.2),
        ]

        # 5. Gaussian Blur (Applied conditionally for larger resolutions >= 64x64)
        if input_shape >= 64:
            transform_list.append(GaussianBlur())

        # Convert to PyTorch Tensor & Normalize using dataset-specific stats
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=stats["mean"], 
                std=stats["std"]
            )
        ])

        self.transform = transforms.Compose(transform_list)

    def __call__(self, x):
        # Applies the exact same transform pipeline twice independently
        x_i = self.transform(x)
        x_j = self.transform(x)
        return x_i, x_j


class LinearEvalDataTransform(object):
    """
    Standard supervised evaluation transforms for linear probing.
    Returns a single image tensor.
    """
    def __init__(self, input_shape=32, is_train=True, dataset_name="cifar10"):
        stats = DATASET_STATS.get(dataset_name.lower(), DATASET_STATS["imagenet"])

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=input_shape, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=stats["mean"], std=stats["std"]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_shape, input_shape)),
                transforms.ToTensor(),
                transforms.Normalize(mean=stats["mean"], std=stats["std"]),
            ])

    def __call__(self, x):
        return self.transform(x)


if __name__ == "__main__":
    from PIL import Image

    # Create a dummy RGB image (32x32)
    dummy_img = Image.new('RGB', (32, 32), color='red')
    
    # Initialize pre-training transform pipeline
    pretrain_transform = SimCLRDataTransform(input_shape=32, dataset_name="cifar10")
    view_1, view_2 = pretrain_transform(dummy_img)

    # Initialize linear evaluation transform pipeline
    eval_transform = LinearEvalDataTransform(input_shape=32, is_train=True, dataset_name="cifar10")
    eval_view = eval_transform(dummy_img)
    
    # Assertions and sanity checks
    assert view_1.shape == (3, 32, 32), f"Unexpected View 1 shape: {view_1.shape}"
    assert view_2.shape == (3, 32, 32), f"Unexpected View 2 shape: {view_2.shape}"
    assert eval_view.shape == (3, 32, 32), f"Unexpected Eval View shape: {eval_view.shape}"
    
    print(f"Pre-training View 1 Tensor Shape: {view_1.shape}")
    print(f"Pre-training View 2 Tensor Shape: {view_2.shape}")
    print(f"Evaluation View Tensor Shape:    {eval_view.shape}")
    print(f"Pre-training views are identical: {torch.equal(view_1, view_2)} (Expected: False)")
    print("Module 1 (augmentations.py) is verified and ready for production!")