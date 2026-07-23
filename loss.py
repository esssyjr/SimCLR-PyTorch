import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for SimCLR.
    Computes pairwise cosine similarities across a batch of N original images 
    (2N augmented views total).
    """
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: [N, D] projected representations for View 1
            z_j: [N, D] projected representations for View 2
        Returns:
            Scalar loss tensor
        """
        device = z_i.device
        N = z_i.shape[0]

        # 1. Concatenate projections to shape [2N, D]
        z = torch.cat([z_i, z_j], dim=0)

        # 2. L2 Normalization along feature dimension (Critical for Cosine Similarity)
        z = F.normalize(z, p=2, dim=1)

        # 3. Compute 2N x 2N Cosine Similarity Matrix: S = (z @ z.T) / temperature
        similarity_matrix = torch.matmul(z, z.T) / self.temperature

        # 4. Create Mask to discard self-similarity (diagonal entries i == j)
        # Using -1e4 instead of -9e15 to prevent FP16/AMP underflow & overflow errors
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        similarity_matrix.masked_fill_(mask, -1e4)

        # 5. Define Targets for Positive Pairs:
        # For item k (where 0 <= k < N), its positive twin is at k + N.
        # For item k (where N <= k < 2N), its positive twin is at k - N.
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),
            torch.arange(0, N, device=device)
        ], dim=0)

        # 6. Cross Entropy Loss over the Softmax distribution
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


if __name__ == "__main__":
    # Simulate mini-batch of N=4 images (2N = 8 total views)
    # Output projection dimension D = 128
    N = 4
    D = 128
    
    # Generate dummy unnormalized projection tensors
    z_i = torch.randn(N, D, requires_grad=True)
    z_j = torch.randn(N, D, requires_grad=True)

    # Initialize loss with standard temperature scale
    criterion = NTXentLoss(temperature=0.1)
    
    # Forward pass
    loss = criterion(z_i, z_j)
    print(f"Calculated NT-Xent Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive!"
    
    # Test backward pass to verify gradient flow through loss
    loss.backward()
    assert z_i.grad is not None, "Gradients failed to flow back to z_i!"
    assert z_j.grad is not None, "Gradients failed to flow back to z_j!"
    print("Backward pass through NTXentLoss successful!")
    
    print("Module 3 (loss.py) is verified and ready for GitHub!")