import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset import get_dataset
from augmentations import LinearEvalDataTransform
from model import SimCLRModel
from utils.seed import set_seed


class LinearClassifier(nn.Module):
    """
    Simple linear layer attached directly to frozen pre-trained encoder representations (h or z).
    """
    def __init__(self, feature_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
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


def run_epoch(model, classifier, dataloader, criterion, optimizer, device, eval_feat="h", is_train=True):
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

            # Extract frozen representations
            with torch.no_grad():
                h, z = model(x)
                # Dynamically choose between backbone features (h) or projection head features (z)
                feat = z if eval_feat.lower() == "z" else h

            # Forward pass through trainable linear classifier
            outputs = classifier(feat)
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


def main():
    parser = argparse.ArgumentParser(description="SimCLR Linear Evaluation Protocol (Paper Compliant)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pre-trained model checkpoint")
    parser.add_argument("--eval_feat", type=str, default="h", choices=["h", "z"], 
                        help="Feature representation to evaluate on: 'h' (backbone output) or 'z' (projection head output)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name (cifar10, stl10, custom). If None, inferred from config.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for dataset")
    parser.add_argument("--arch", type=str, default=None, choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Backbone architecture. If None, inferred from checkpoint config.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for linear evaluation")
    parser.add_argument("--epochs", type=int, default=50, help="Linear evaluation epochs (Paper uses 90 for ImageNet)")
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer (Paper uses SGD)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (0.1 for SGD, 1e-3 for Adam)")
    parser.add_argument("--img_size", type=int, default=None, help="Input image dimension. If None, inferred from config.")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of target classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_classifier", type=str, default="./saved_models/linear_classifier.pth", help="Path to save trained classifier")
    
    args = parser.parse_args()

    # Set seed for exact reproducibility
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Checkpoint & Infer Config
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.ckpt_path}")
        
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    arch = args.arch or config.get("arch", "resnet18")
    img_size = args.img_size or config.get("img_size", 32)
    dataset_name = args.dataset or config.get("dataset", "cifar10")

    print(f"Checkpoint Metadata -> Arch: {arch} | Img Size: {img_size}x{img_size} | Dataset: {dataset_name}")

    # Modularized Datasets with proper evaluation transforms
    train_transform = LinearEvalDataTransform(input_shape=img_size, is_train=True, dataset_name=dataset_name)
    test_transform = LinearEvalDataTransform(input_shape=img_size, is_train=False, dataset_name=dataset_name)

    train_dataset = get_dataset(name=dataset_name, data_dir=args.data_dir, transform=train_transform, train=True, is_pretraining=False)
    test_dataset = get_dataset(name=dataset_name, data_dir=args.data_dir, transform=test_transform, train=False, is_pretraining=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Build Model & Adapt Stem
    model = SimCLRModel(base_arch=arch, out_dim=128)
    model = adapt_resnet_stem(model, img_size=img_size)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # Freeze Backbone Encoder
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Safe feature dimension calculation based on chosen feature (h vs z)
    if args.eval_feat.lower() == "z":
        feature_dim = getattr(model, "out_dim", 128)
    else:
        if hasattr(model, "num_ftrs"):
            feature_dim = model.num_ftrs
        else:
            resnet_ftrs = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048, "resnet152": 2048}
            feature_dim = resnet_ftrs.get(arch.lower(), 512)

    print(f"Selected Feature Space: '{args.eval_feat.upper()}' | Detected Dimension: {feature_dim}")

    # Attach Linear Classifier directly to selected representation space
    classifier = LinearClassifier(feature_dim=feature_dim, num_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()

    # Paper Protocol Optimizer
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=1e-6)

    # Linear Evaluation Loop
    print(f"Starting Linear Evaluation on representation '{args.eval_feat.upper()}' of frozen {arch} using {args.optimizer.upper()} for {args.epochs} epochs...")
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, classifier, train_loader, criterion, optimizer, device, eval_feat=args.eval_feat, is_train=True)
        test_loss, test_acc = run_epoch(model, classifier, test_loader, criterion, optimizer, device, eval_feat=args.eval_feat, is_train=False)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # Save best classifier state dict
            if os.path.dirname(args.save_classifier):
                os.makedirs(os.path.dirname(args.save_classifier), exist_ok=True)
            torch.save(classifier.state_dict(), args.save_classifier)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")

    print(f"\nLinear Evaluation Completed for representation '{args.eval_feat.upper()}'! Peak Test Accuracy: {best_test_acc:.2f}%")
    print(f"Linear classifier state dict saved to: {args.save_classifier}")


if __name__ == "__main__":
    main()