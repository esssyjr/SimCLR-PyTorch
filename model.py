import torch
import torch.nn as nn
import torchvision.models as models

class SimCLRModel(nn.Module):
    """
    SimCLR Architecture consisting of a base ResNet encoder backbone 
    and a 2-layer non-linear MLP projection head.
    Supports: resnet18, resnet34, resnet50, resnet101, resnet152.
    """
    def __init__(self, base_arch='resnet18', out_dim=128):
        super(SimCLRModel, self).__init__()
        
        # 1. Dynamically load any ResNet architecture from torchvision
        if not hasattr(models, base_arch):
            raise ValueError(f"Unsupported architecture: '{base_arch}'. Choose from torchvision.models (e.g., resnet18, resnet50).")
            
        resnet_fn = getattr(models, base_arch)
        self.backbone = resnet_fn(weights=None)
        
        # 2. Extract feature dimension dynamically (512 for R18/R34, 2048 for R50/R101/R152)
        num_ftrs = self.backbone.fc.in_features
        
        # 3. Replace classification head with Identity
        self.backbone.fc = nn.Identity()
        
        # 4. Add 2-layer MLP projection head: g(h) = W2 * relu(W1 * h)
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(inplace=True),
            nn.Linear(num_ftrs, out_dim)
        )

    def forward(self, x):
        # h: Representation vector before projection head [N, num_ftrs]
        h = self.backbone(x)
        
        # z: Unnormalized projected vector [N, out_dim]
        # (L2 normalization is handled inside loss.py)
        z = self.projection_head(h)
        
        return h, z


if __name__ == "__main__":
    # Simulate a mini-batch of 4 RGB images
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # Test with lightweight ResNet-18
    model_r18 = SimCLRModel(base_arch='resnet18', out_dim=128)
    h18, z18 = model_r18(dummy_input)
    assert h18.shape == (4, 512), f"Unexpected R18 h shape: {h18.shape}"
    assert z18.shape == (4, 128), f"Unexpected R18 z shape: {z18.shape}"
    print("ResNet-18 forward pass verified!")

    # Test with paper-standard ResNet-50
    model_r50 = SimCLRModel(base_arch='resnet50', out_dim=128)
    h50, z50 = model_r50(dummy_input)
    assert h50.shape == (4, 2048), f"Unexpected R50 h shape: {h50.shape}"
    assert z50.shape == (4, 128), f"Unexpected R50 z shape: {z50.shape}"
    print("ResNet-50 forward pass verified!")
    
    # Test backward pass
    dummy_loss = z50.mean()
    dummy_loss.backward()
    print("Backward pass successful!")
    
    print("Module 2 (model.py) is verified and ready for GitHub!")