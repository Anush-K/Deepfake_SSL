import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.1):
    """
    z1, z2: normalized embeddings [B, D]
    """

    B = z1.size(0)

    z = torch.cat([z1, z2], dim=0)  # 2B x D
    sim = torch.mm(z, z.t()) / temperature

    # mask self similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # positive pairs
    positives = torch.cat([
        torch.arange(B, 2*B),
        torch.arange(0, B)
    ]).to(z.device)

    loss = F.cross_entropy(sim, positives)
    return loss