import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import random

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter


class GaussianBlur(object):
    """
    SimCLR Gaussian Blur implementation with a 50% probability of being applied.
    """
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        if random.random() < 0.5:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class SimCLRDataTransform(object):
    """
    Generates two stochastic augmented views of a single image.
    """
    def __init__(self, input_shape=224, s=1.0):
        color_jitter = transforms.ColorJitter(
            0.8 * s,
            0.8 * s,
            0.8 * s,
            0.2 * s
        )

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=input_shape,
                scale=(0.08, 1.0)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, x):
        x_i = self.transform(x)
        x_j = self.transform(x)
        return x_i, x_j


if __name__ == "__main__":

    # ======================================================
    # Change this to your own image
    # ======================================================
    IMAGE_PATH = "cat.jpg"

    # Load image
    image = Image.open(IMAGE_PATH).convert("RGB")

    # Create augmentation pipeline
    transform = SimCLRDataTransform(input_shape=224)

    # Generate two augmented views
    view_1, view_2 = transform(image)

    # Verify shapes
    assert view_1.shape == (3, 224, 224)
    assert view_2.shape == (3, 224, 224)

    print("View 1 Shape:", view_1.shape)
    print("View 2 Shape:", view_2.shape)
    print("Views are identical:", torch.equal(view_1, view_2))

    # Undo normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    view1 = (view_1 * std + mean).clamp(0, 1)
    view2 = (view_2 * std + mean).clamp(0, 1)

    view1 = view1.permute(1, 2, 0)
    view2 = view2.permute(1, 2, 0)

    # Display images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(view1)
    plt.title("Augmented View 1")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(view2)
    plt.title("Augmented View 2")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\n✅ Module 1 verified successfully!")