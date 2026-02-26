import torch
from torch.utils.data import DataLoader
from ssl.ssl_dataset import SSLDataset
from ssl.ssl_model import SSLModel
from ssl.contrastive_loss import nt_xent_loss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


def train_ssl(csv_files, device="cuda", epochs=50, batch_size=64):

    dataset = SSLDataset(csv_files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    model = SSLModel().to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for v1, v2 in tqdm(loader):

            v1, v2 = v1.to(device), v2.to(device)

            z1 = model(v1)
            z2 = model(v2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "ssl_pretrained.pth")