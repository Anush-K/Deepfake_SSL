import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.1):

    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = torch.mm(z, z.t()) #/ temperature

    # mask self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    # Apply temperature after masking
    sim = sim / temperature
    # create positive labels
    targets = torch.arange(B, 2 * B, device=z.device)
    targets = torch.cat([targets, torch.arange(0, B, device=z.device)])

    loss = F.cross_entropy(sim, targets)

    return loss