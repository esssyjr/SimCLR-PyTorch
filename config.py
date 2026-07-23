import torch

class SimCLRConfig:
    """
    Centralized configuration management for SimCLR pre-training and linear evaluation.
    Allows easy toggling between paper-exact settings and Colab-friendly defaults.
    """
    # ----------------------------
    # 1. Dataset & Paths
    # ----------------------------
    data_dir: str = "./data"
    save_dir: str = "./saved_models"
    dataset_name: str = "cifar10"  # Supported: "cifar10", "imagenet", "custom"
    
    # ----------------------------
    # 2. Architecture Settings
    # ----------------------------
    # Choices: "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
    arch: str = "resnet18"        # Default lightweight encoder; set to "resnet50" for exact paper match
    out_dim: int = 128            # Projection head feature dimension (z) [Paper: 128]
    cifar_stem: bool = True       # Modify initial conv/pool layers for 32x32 images (CIFAR-10)

    # ----------------------------
    # 3. Augmentation Settings
    # ----------------------------
    img_size: int = 32            # Resolution (32 for CIFAR-10, 224 for ImageNet/Paper)
    blur_prob: float = 0.5        # Gaussian blur probability [Paper: 0.5]
    color_jitter_s: float = 1.0   # Color jitter strength factor [Paper: 1.0]

    # ----------------------------
    # 4. Loss Function (NT-Xent)
    # ----------------------------
    temperature: float = 0.1      # NT-Xent loss scaling parameter [Paper: 0.1]

    # ----------------------------
    # 5. Pre-training Settings (train.py)
    # ----------------------------
    batch_size: int = 128         # Batch size [Paper: 4096; Colab default: 128]
    epochs: int = 50              # Pre-training epochs [Paper: 100-1000; Colab default: 50]
    optimizer: str = "adam"       # Choices: "adam", "lars", "sgd" [Paper: "lars"]
    lr: float = 3e-4              # Base learning rate [Adam: 3e-4; Paper LARS: 0.3 * (BatchSize / 256)]
    weight_decay: float = 1e-4    # Weight decay [Paper: 1e-6; Adam default: 1e-4]
    use_amp: bool = True          # Automatic Mixed Precision for fast GPU training

    # ----------------------------
    # 6. Linear Evaluation Settings (evaluate.py)
    # ----------------------------
    eval_batch_size: int = 256    # Batch size during linear probe training
    eval_epochs: int = 20         # Linear evaluation epochs [Paper: 90; Colab default: 20]
    eval_optimizer: str = "adam"  # Choices: "adam", "sgd" [Paper: "sgd"]
    eval_lr: float = 1e-3         # Learning rate for linear classifier [Adam: 1e-3; Paper SGD: 0.1]
    num_classes: int = 10         # CIFAR-10 classes

    # ----------------------------
    # 7. Hardware & System
    # ----------------------------
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 2
    pin_memory: bool = True


if __name__ == "__main__":
    cfg = SimCLRConfig()
    print(f"Loaded SimCLRConfig successfully!")
    print(f"Target Device: {cfg.device}")
    print(f"Pre-training Architecture: {cfg.arch} (CIFAR Stem: {cfg.cifar_stem})")
    print(f"Batch Size: {cfg.batch_size} | Epochs: {cfg.epochs} | Temperature: {cfg.temperature}")