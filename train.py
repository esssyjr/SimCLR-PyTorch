import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from augmentations import SimCLRDataTransform
from model import SimCLRModel
from loss import NTXentLoss

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    for step, ((x_i, x_j), _) in enumerate(dataloader):
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Modern PyTorch Automatic Mixed Precision (AMP) API
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=(device.type == "cuda")):
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = criterion(z_i, z_j)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def adapt_resnet_for_cifar(model):
    """
    Replaces initial 7x7 conv (stride 2) and maxpool with a 3x3 conv (stride 1)
    to prevent spatial dimension collapse on 32x32 images.
    """
    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.backbone.maxpool = nn.Identity()
    return model


def main():
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Pre-training")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for raw dataset")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="resnet18", 
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Backbone architecture")
    parser.add_argument("--cifar_stem", action="store_true", help="Adapt ResNet first conv/pool layer for 32x32 images")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=20, help="Number of pre-training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="NT-Xent temperature parameter")
    parser.add_argument("--img_size", type=int, default=32, help="Input image dimension (32 for CIFAR10, 224 for ImageNet)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    transform = SimCLRDataTransform(input_shape=args.img_size)
    dataset = CIFAR10(root=args.data_dir, train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    # 2. Model, Loss, Optimizer, Scaler, Scheduler
    model = SimCLRModel(base_arch=args.arch, out_dim=128)
    if args.cifar_stem:
        model = adapt_resnet_for_cifar(model)
    model = model.to(device)

    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Modern PyTorch GradScaler API
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # 3. Pre-training Loop
    print(f"Starting pre-training for {args.epochs} epochs on {args.arch}...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(model, dataloader, optimizer, criterion, scaler, device)
        scheduler.step()
        
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {epoch_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save structured checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"simclr_{args.arch}_epoch_{epoch}.pth")
            checkpoint = {
                "epoch": epoch,
                "arch": args.arch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()