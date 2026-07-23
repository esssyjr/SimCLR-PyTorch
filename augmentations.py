import torchvision.transforms as transforms
from PIL import ImageFilter
import random
import torch

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
        # s is the color jitter strength parameter (default 1.0 as specified in paper)
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        
        self.transform = transforms.Compose([
            # 1. Random Resized Crop
            transforms.RandomResizedCrop(size=input_shape, scale=(0.08, 1.0)),
            # 2. Horizontal Flip
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. Color Jitter (80% probability)
            transforms.RandomApply([color_jitter], p=0.8),
            # 4. Grayscale (20% probability)
            transforms.RandomGrayscale(p=0.2),
            # 5. Gaussian Blur
            GaussianBlur(),
            # Convert to PyTorch Tensor & Normalize
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, x):
        # Applies the exact same transform pipeline twice independently
        x_i = self.transform(x)
        x_j = self.transform(x)
        return x_i, x_j


if __name__ == "__main__":
    from PIL import Image

    # Create a dummy RGB image (224x224)
    dummy_img = Image.new('RGB', (224, 224), color='red')
    
    # Initialize transform
    transform_pipeline = SimCLRDataTransform(input_shape=224)
    
    # Get positive pair
    view_1, view_2 = transform_pipeline(dummy_img)
    
    # Assertions and sanity checks
    assert view_1.shape == (3, 224, 224), f"Unexpected View 1 shape: {view_1.shape}"
    assert view_2.shape == (3, 224, 224), f"Unexpected View 2 shape: {view_2.shape}"
    
    print(f"View 1 Tensor Shape: {view_1.shape}")
    print(f"View 2 Tensor Shape: {view_2.shape}")
    print(f"Views are identical: {torch.equal(view_1, view_2)} (Expected: False)")
    print("Module 1 (augmentations.py) is verified and ready for GitHub!")
