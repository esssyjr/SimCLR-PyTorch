import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from model import SimCLRModel


class LinearClassifier(nn.Module):
    """
    Simple linear layer attached directly to frozen pre-trained encoder features.
    """
    def __init__(self, feature_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def adapt_resnet_for_cifar(model):
    """
    Adapts ResNet stem for 32x32 images (matches the training setup if cifar_stem was used).
    """
    model.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.backbone.maxpool = nn.Identity()
    return model


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
            x, y = x.to(device), y.to(device)

            # Extract representation h from frozen encoder
            with torch.no_grad():
                h, _ = model(x)

            # Forward pass through linear classifier
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


def main():
    parser = argparse.ArgumentParser(description="SimCLR Linear Evaluation Protocol")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pre-trained model checkpoint")
    parser.add_argument("--arch", type=str, default="resnet18", 
                        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                        help="Backbone architecture")
    parser.add_argument("--cifar_stem", action="store_true", help="Adapt ResNet first conv/pool layer for 32x32 images")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for linear evaluation")
    parser.add_argument("--epochs", type=int, default=20, help="Linear evaluation epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for linear classifier")
    parser.add_argument("--img_size", type=int, default=32, help="Input image dimension")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard testing transformations (No heavy augmentations during evaluation)
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CIFAR10(root="./data", train=True, transform=test_transform, download=True)
    test_dataset = CIFAR10(root="./data", train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 1. Instantiate Base Model
    model = SimCLRModel(base_arch=args.arch, out_dim=128)
    if args.cifar_stem:
        model = adapt_resnet_for_cifar(model)

    # 2. Load Checkpoint (Robust to both dict checkpoints and raw state_dicts)
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {args.ckpt_path}")
        
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)

    # 3. Freeze Backbone Encoder Parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 4. Attach Linear Classifier
    feature_dim = 512 if args.arch in ["resnet18", "resnet34"] else 2048
    classifier = LinearClassifier(feature_dim=feature_dim, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # 5. Run Linear Evaluation
    print(f"Starting Linear Evaluation on frozen {args.arch} backbone for {args.epochs} epochs...")
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, classifier, train_loader, criterion, optimizer, device, is_train=True)
        test_loss, test_acc = run_epoch(model, classifier, test_loader, criterion, optimizer, device, is_train=False)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    print(f"\nLinear Evaluation Completed! Peak Test Accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()