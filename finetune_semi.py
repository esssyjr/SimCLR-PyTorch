import argparse
import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import SimCLRModel
from utils.seed import set_seed


class FineTuneClassifier(nn.Module):
    """
    End-to-End Fine-Tuning Architecture.
    Attaches a linear classifier directly to feature vector h, keeping the backbone unfrozen.
    """
    def __init__(self, base_arch, num_classes, pretrain_img_size=32):
        super(FineTuneClassifier, self).__init__()
        self.encoder = SimCLRModel(base_arch=base_arch, out_dim=128)
        
        # Adapt ResNet stem to match pre-trained checkpoint
        if pretrain_img_size <= 64:
            self.encoder.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.encoder.backbone.maxpool = nn.Identity()

        # Determine feature dimension h
        resnet_ftrs = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}
        feature_dim = getattr(self.encoder, "num_ftrs", resnet_ftrs.get(base_arch.lower(), 512))

        # Downstream linear classifier head
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # Extract representation h directly from encoder backbone
        h, _ = self.encoder(x)
        logits = self.fc(h)
        return logits


def get_fine_tune_transforms(img_size):
    """
    Supervised data transforms with light augmentation for fine-tuning.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Fine-Tuning Network] Total Parameters: {total:,} | Trainable Parameters: {trainable:,}")
    return total, trainable


def run_epoch(model, dataloader, criterion, optimizer, device, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            outputs = model(x)
            loss = criterion(outputs, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    acc = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    return avg_loss, acc


def log_results_to_csv(csv_path, fieldnames, row_dict):
    file_exists = os.path.exists(csv_path)
    if os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    parser = argparse.ArgumentParser(description="Publication-Grade End-to-End Fine-Tuning Protocol")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pre-trained SimCLR checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing Train/, val/ (optional), and Test/")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tuning epochs")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer choice")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for end-to-end fine-tuning (Recommended: 1e-4 for Adam, 1e-2 for SGD)")
    parser.add_argument("--img_size", type=int, default=224, help="Target image resolution")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./experiment_results", help="Directory to save checkpoints and logs")
    parser.add_argument("--exp_name", type=str, default="naira_finetune", help="Unique identifier for experiment")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Inspect Pre-trained Checkpoint Config
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.ckpt_path}")

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    arch = config.get("arch", "resnet18")
    pretrain_img_size = config.get("img_size", 32)
    img_size = args.img_size

    # 2. Check Dataset Directories
    train_path = os.path.join(args.data_dir, "Train") if os.path.exists(os.path.join(args.data_dir, "Train")) else os.path.join(args.data_dir, "train")
    val_path = os.path.join(args.data_dir, "val") if os.path.exists(os.path.join(args.data_dir, "val")) else os.path.join(args.data_dir, "Val")
    test_path = os.path.join(args.data_dir, "Test") if os.path.exists(os.path.join(args.data_dir, "Test")) else os.path.join(args.data_dir, "test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset MUST contain 'train/' and 'test/' directories! Checked: {args.data_dir}")
    
    has_val = os.path.exists(val_path)

    # 3. Setup DataLoaders
    train_transform, eval_transform = get_fine_tune_transforms(img_size)

    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=eval_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=eval_transform) if has_val else None

    classes = train_dataset.classes
    num_classes = len(classes)

    print(f"\n================ Target Dataset Metadata ================")
    print(f"Dataset: {args.data_dir} | Detected Classes ({num_classes}): {classes}")
    print(f"Input Resolution: {img_size}x{img_size}")
    print(f"Samples -> Train: {len(train_dataset)} | " + (f"Val: {len(val_dataset)} | " if has_val else "") + f"Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True) if has_val else None

    # 4. Instantiate Fine-Tuning Model & Load Pre-trained Weights
    model = FineTuneClassifier(base_arch=arch, num_classes=num_classes, pretrain_img_size=pretrain_img_size)

    # Load encoder pre-trained weights
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.encoder.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.encoder.load_state_dict(checkpoint)

    # UNFREEZE ALL PARAMETERS FOR END-TO-END FINE-TUNING
    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    print(f"\n================ Parameter Distribution ================")
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()

    # Setup Optimizer & Scheduler
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5. Fine-Tuning Loop
    print(f"\nStarting End-to-End Fine-Tuning on '{args.exp_name}' using {args.optimizer.upper()} (LR={args.lr})...")
    best_val_acc = 0.0
    best_epoch = 0

    checkpoint_save_path = os.path.join(args.save_dir, f"{args.exp_name}_best_model.pth")
    csv_log_path = os.path.join(args.save_dir, f"{args.exp_name}_training_log.csv")

    csv_fieldnames = ["epoch", "lr", "train_loss", "train_acc", "eval_loss", "eval_acc"]

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        
        eval_loader = val_loader if has_val else test_loader
        eval_name = "Val" if has_val else "Test"
        eval_loss, eval_acc = run_epoch(model, eval_loader, criterion, optimizer, device, is_train=False)

        scheduler.step()

        # Log metrics to CSV
        log_results_to_csv(csv_log_path, csv_fieldnames, {
            "epoch": epoch,
            "lr": f"{current_lr:.6f}",
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.2f}",
            "eval_loss": f"{eval_loss:.4f}",
            "eval_acc": f"{eval_acc:.2f}"
        })

        # Save Best Full Model Checkpoint
        if eval_acc > best_val_acc:
            best_val_acc = eval_acc
            best_epoch = epoch

            meta_checkpoint = {
                "epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "model_state_dict": model.state_dict(),
                "experiment_metadata": {
                    "dataset": args.data_dir,
                    "exp_name": args.exp_name,
                    "num_classes": num_classes,
                    "class_names": classes,
                    "pretrained_ckpt": args.ckpt_path,
                    "optimizer": args.optimizer,
                    "learning_rate": args.lr,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "arch": arch
                }
            }
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(meta_checkpoint, checkpoint_save_path)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | {eval_name} Loss: {eval_loss:.4f} Acc: {eval_acc:.2f}% (Best {eval_name}: {best_val_acc:.2f}%)")

    # 6. Final Benchmark Evaluation on Unseen Test Set
    if has_val:
        print(f"\n================ Final Benchmark Evaluation ================")
        best_ckpt = torch.load(checkpoint_save_path)
        model.load_state_dict(best_ckpt["model_state_dict"])
        
        final_test_loss, final_test_acc = run_epoch(model, test_loader, criterion, optimizer, device, is_train=False)
        print(f"Loaded Best Model from Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%)")
        print(f"🔥 Final Unseen Test Set Accuracy (Fine-Tuned): {final_test_acc:.2f}%")
        
        summary_csv_path = os.path.join(args.save_dir, "benchmark_summary.csv")
        summary_fieldnames = ["experiment", "arch", "dataset", "best_val_acc", "final_test_acc", "best_epoch", "seed"]
        log_results_to_csv(summary_csv_path, summary_fieldnames, {
            "experiment": args.exp_name,
            "arch": arch,
            "dataset": args.data_dir,
            "best_val_acc": f"{best_val_acc:.2f}",
            "final_test_acc": f"{final_test_acc:.2f}",
            "best_epoch": best_epoch,
            "seed": args.seed
        })

    print(f"\nExperiment Complete! Results logged to: {csv_log_path}")
    print(f"Best fine-tuned checkpoint saved to: {checkpoint_save_path}")


if __name__ == "__main__":
    main()