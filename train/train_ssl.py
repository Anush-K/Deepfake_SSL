import os
import glob
import torch
from torch.utils.data import DataLoader
from ssl_simclr.ssl_dataset import SSLDataset
from ssl_simclr.ssl_model import SSLModel
from ssl_simclr.contrastive_loss import nt_xent_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def train_ssl(csv_files, device="cuda", epochs=50, batch_size=64):

    dataset = SSLDataset(csv_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    model = SSLModel().to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    total_steps = epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, total_steps)

    scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint if exists
    start_epoch = 0
    checkpoint_dir = "/content/drive/MyDrive/DF_Datasets/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Find latest checkpoint
    checkpoints = sorted(glob.glob(f"{checkpoint_dir}/ssl_epoch_*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Resuming from {latest}")
        ckpt = torch.load(latest)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"Resumed from epoch {start_epoch}")


    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss = 0

        for v1, v2 in tqdm(loader):

            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        current_lr = optimizer.param_groups[0]["lr"]
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        #print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, f"/content/drive/MyDrive/DF_Datasets/checkpoints/ssl_epoch_{epoch+1}.pt")

    torch.save(
        model.state_dict(),
        "/content/drive/MyDrive/DF_Datasets/ssl_final.pth"
    )