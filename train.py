import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset import get_dataset
from augmentations import SimCLRDataTransform
from model import SimCLRModel
from loss import NTXentLoss
from utils.seed import set_seed


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    for step, ((x_i, x_j), _) in enumerate(dataloader):
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        optimizer.zero_grad()

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


def adapt_resnet_stem(model, img_size):
    """
    Automatically adapts the initial conv/pool stem based on input image resolution.
    - Low resolution (<= 64x64): Replaces 7x7 stride-2 conv and maxpool with a 3x3 stride-1 conv.
    - High resolution (> 64x64): Keeps the standard ImageNet stem.
    """
    if img_size <= 64:
        print(f"-> Low resolution detected ({img_size}x{img_size}). Adapting CIFAR stem (3x3 conv, stride 1, no maxpool)...")
        model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.backbone.maxpool = nn.Identity()
    else:
        print(f"-> High resolution detected ({img_size}x{img_size}). Using standard ImageNet stem (7x7 conv, stride 2, maxpool)...")
    return model


def main():
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Pre-training Engine")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (cifar10, stl10, custom, etc.)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for raw dataset")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="resnet18", 
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Backbone architecture")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of pre-training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.1, help="NT-Xent temperature parameter")
    parser.add_argument("--img_size", type=int, default=32, help="Input image dimension (32 for CIFAR10, 224 for ImageNet/Custom)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pth) to resume training from")
    
    args = parser.parse_args()

    # 0. Set seed for complete reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Dataset & DataLoader (Modularized)
    transform = SimCLRDataTransform(input_shape=args.img_size)
    dataset = get_dataset(name=args.dataset, data_dir=args.data_dir, transform=transform, train=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    # 2. Model, Loss, Optimizer, Scaler, Scheduler
    model = SimCLRModel(base_arch=args.arch, out_dim=128)
    model = adapt_resnet_stem(model, img_size=args.img_size)
    model = model.to(device)

    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # 3. Resume Checkpoint Logic
    start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from '{args.resume}'...")
            checkpoint = torch.load(args.resume, map_location=device)
            
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            if "scaler_state_dict" in checkpoint and device.type == "cuda":
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                
            start_epoch = checkpoint["epoch"] + 1
            print(f"Successfully resumed! Continuing training from Epoch {start_epoch}/{args.epochs}")
        else:
            print(f"No checkpoint found at '{args.resume}'! Starting from scratch.")

    # 4. Pre-training Loop
    print(f"Starting pre-training from Epoch {start_epoch} to {args.epochs} on {args.arch} with {args.dataset}...")
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss = train_one_epoch(model, dataloader, optimizer, criterion, scaler, device)
        scheduler.step()
        
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {epoch_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save structured checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.save_dir, f"simclr_{args.arch}_{args.dataset}_epoch_{epoch}.pth")
            checkpoint = {
                "epoch": epoch,
                "config": vars(args),  # Complete argument configuration dictionary
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "loss": epoch_loss,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()