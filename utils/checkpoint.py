import os
import torch

def save_checkpoint(model, optimizer, epoch, loss, save_dir, filename="checkpoint.pth", config=None):
    """
    Saves structured model checkpoints along with metadata and configurations.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "arch": model.__class__.__name__,
        "config": config if config else {}
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved successfully to {path}")

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """
    Loads checkpoints and restores model/optimizer training states.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at '{path}'")
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
    print(f"Loaded checkpoint from '{path}' (Epoch {checkpoint.get('epoch', 'N/A')})")
    return checkpoint.get("epoch", 0), checkpoint.get("loss", None)