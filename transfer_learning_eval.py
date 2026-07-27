import argparse
import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import SimCLRModel
from utils.seed import set_seed


class TransferClassifier(nn.Module):
    """
    Linear classifier attached to frozen pre-trained encoder representations (h).
    Strictly linear: W * h + b (No non-linear activations).
    """
    def __init__(self, feature_dim, num_classes):
        super(TransferClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def adapt_resnet_stem(model, img_size):
    if img_size <= 64:
        print(f"-> Low resolution detected ({img_size}x{img_size}). Adapting CIFAR stem (3x3 conv, stride 1, no maxpool)...")
        model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.backbone.maxpool = nn.Identity()
    else:
        print(f"-> High resolution detected ({img_size}x{img_size}). Using standard ImageNet stem (7x7 conv, stride 2, maxpool)...")
    return model


def get_deterministic_transforms(img_size):
    """
    Paper-compliant evaluation transforms: strictly deterministic (Resize + Normalize).
    No training augmentations during initial linear probe benchmarks.
    """
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return eval_transform


def count_parameters(model, name="Model"):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Total Parameters: {total:,} | Trainable Parameters: {trainable:,}")
    return total, trainable


def run_epoch(model, classifier, dataloader, criterion, optimizer, device, is_train=True):
    if is_train:
        classifier.train()
    else:
        classifier.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Extract frozen backbone representation h
            with torch.no_grad():
                h, _ = model(x)

            outputs = classifier(h)
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
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def main():
    parser = argparse.ArgumentParser(description="Publication-Grade Linear Transfer Evaluation")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pre-trained SimCLR checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory containing train/, val/, and test/")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to train linear classifier")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer choice")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=None, help="Image dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="./experiment_results", help="Directory to save checkpoints and logs")
    parser.add_argument("--exp_name", type=str, default="naira_transfer", help="Unique identifier for experiment")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Pre-trained Checkpoint
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.ckpt_path}")

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    arch = config.get("arch", "resnet18")
    img_size = args.img_size or config.get("img_size", 32)

    # 2. Check Directories & Protocol Splits
    train_path = os.path.join(args.data_dir, "Train")
    val_path = os.path.join(args.data_dir, "val")
    test_path = os.path.join(args.data_dir, "Test")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Dataset MUST contain 'train/' and 'test/' directories! Checked: {args.data_dir}")
    
    has_val = os.path.exists(val_path)
    if not has_val:
        print("⚠️ WARNING: No 'val/' directory detected! Protocol will fallback to selecting best epoch on test set (Non-ideal for publications).")

    # 3. Load Datasets with Deterministic Evaluation Transforms
    eval_transform = get_deterministic_transforms(img_size)

    train_dataset = datasets.ImageFolder(root=train_path, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=eval_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=eval_transform) if has_val else None

    classes = train_dataset.classes
    num_classes = len(classes)

    print(f"\n================ Target Dataset Metadata ================")
    print(f"Dataset: {args.data_dir} | Detected Classes ({num_classes}): {classes}")
    print(f"Samples -> Train: {len(train_dataset)} | " + (f"Val: {len(val_dataset)} | " if has_val else "") + f"Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True) if has_val else None

    # 4. Build Pre-trained SimCLR Backbone
    model = SimCLRModel(base_arch=arch, out_dim=128)
    model = adapt_resnet_stem(model, img_size=img_size)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # Freeze Encoder Backbone
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Calculate Feature Dimension h
    if hasattr(model, "num_ftrs"):
        feature_dim = model.num_ftrs
    else:
        resnet_ftrs = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}
        feature_dim = resnet_ftrs.get(arch.lower(), 512)

    classifier = TransferClassifier(feature_dim=feature_dim, num_classes=num_classes).to(device)

    print(f"\n================ Parameter Distribution ================")
    count_parameters(model, name="Frozen ResNet-18 Backbone")
    count_parameters(classifier, name="Trainable Linear Head")

    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-6)

    # 5. Training Loop & Validation Tracking
    print(f"\nStarting Linear Evaluation on '{args.exp_name}' using {args.optimizer.upper()}...")
    best_val_acc = 0.0
    best_epoch = 0

    checkpoint_save_path = os.path.join(args.save_dir, f"{args.exp_name}_best_classifier.pth")
    csv_log_path = os.path.join(args.save_dir, f"{args.exp_name}_training_log.csv")

    csv_fieldnames = ["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc"]

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, classifier, train_loader, criterion, optimizer, device, is_train=True)
        
        # Evaluate on Val if available, else Test
        eval_loader = val_loader if has_val else test_loader
        eval_name = "Val" if has_val else "Test"
        eval_loss, eval_acc = run_epoch(model, classifier, eval_loader, criterion, optimizer, device, is_train=False)

        # Log metrics to CSV
        log_results_to_csv(csv_log_path, csv_fieldnames, {
            "epoch": epoch,
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.2f}",
            "eval_loss": f"{eval_loss:.4f}",
            "eval_acc": f"{eval_acc:.2f}"
        })

        # Save Best Model Based on Validation Performance
        if eval_acc > best_val_acc:
            best_val_acc = eval_acc
            best_epoch = epoch

            meta_checkpoint = {
                "epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "classifier_state_dict": classifier.state_dict(),
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

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | {eval_name} Loss: {eval_loss:.4f} Acc: {eval_acc:.2f}% (Best {eval_name}: {best_val_acc:.2f}%)")

    # 6. Final Unseen Test Set Evaluation (If Val set was used)
    if has_val:
        print(f"\n================ Final Benchmark Evaluation ================")
        best_ckpt = torch.load(checkpoint_save_path)
        classifier.load_state_dict(best_ckpt["classifier_state_dict"])
        
        final_test_loss, final_test_acc = run_epoch(model, classifier, test_loader, criterion, optimizer, device, is_train=False)
        print(f"Loaded Best Classifier from Epoch {best_epoch} (Val Acc: {best_val_acc:.2f}%)")
        print(f"🔥 Final Unseen Test Set Accuracy: {final_test_acc:.2f}%")
        
        # Summary row in CSV
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
    print(f"Best classifier checkpoint saved to: {checkpoint_save_path}")


if __name__ == "__main__":
    main()