import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import argparse


class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if generator is None:
            eps = torch.randn_like(self.std, dtype=torch.float32)
        else:
            eps = torch.randn_like(self.std, dtype=torch.float32, generator=generator)
        return self.mean + self.std * eps

    def mode(self) -> torch.Tensor:
        return self.mean


def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE model on a specific GPU")
    parser.add_argument("--gpu_id", type=int, default=0, help="ID of the GPU to use (0,1,...)")
    parser.add_argument("--csv_folder", type=str,
                        default="C:/Users/admin/Documents/GPBL/diffusers/Data_Dipole_Model_Normalized",
                        help="Path to the folder containing CSV files")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use_kl", type=bool, default=True, help="Use KL divergence in loss")
    parser.add_argument("--beta_kl", type=float, default=1e-4, help="KL divergence weight in VAE loss")
    parser.add_argument("--visualize", action="store_true", help="Run latent visualization after training")
    return parser.parse_args()


class CSVDataset2D(Dataset):
    def __init__(self, csv_folder: str):
        self.csv_files = sorted([os.path.join(csv_folder, f)
                                 for f in os.listdir(csv_folder) if f.endswith(".csv")])

    def __len__(self) -> int:
        return len(self.csv_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = np.loadtxt(self.csv_files[idx], delimiter=",", dtype=np.float32)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        return img, 0


class VAE_model(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        use_kl: bool = True,
        sample_size: int = 32,
        beta_kl: float = 1e-4,
    ):
        super().__init__()
        self.config = {
            "in_channels": in_channels,
            "latent_channels": latent_channels,
            "use_kl": use_kl,
            "sample_size": sample_size,
            "beta_kl": beta_kl,
        }
        self.use_kl = use_kl
        self.beta_kl = beta_kl
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.conv_mu = nn.Conv2d(64, latent_channels * 2, kernel_size=3, stride=1, padding=1)
        self.decoder_input = nn.Conv2d(latent_channels, 64, kernel_size=3, stride=1, padding=1)
        self.decoder = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
        )

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        h = self.conv_mu(h)
        return DiagonalGaussianDistribution(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.decoder_input(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x: torch.Tensor, sample_posterior: bool = False,
                generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior = self.encode(x)
        z = posterior.sample(generator) if sample_posterior else posterior.mode()
        recon_x = self.decode(z)
        mu, logvar = posterior.mean, posterior.logvar
        return recon_x, mu, logvar

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f)

    @classmethod
    def from_pretrained(cls, load_directory: str, map_location=None):
        with open(os.path.join(load_directory, "config.json"), "r") as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(
            torch.load(os.path.join(load_directory, "pytorch_model.bin"),
                       map_location=map_location or "cpu")
        )
        return model.to(dtype=torch.float32)


def vae_loss(recon_x, x, mu, logvar, beta_kl=1e-4):
    mse = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta_kl * kld


def save_reconstruction(original, reconstructed, epoch, save_dir="vae_step_r_results"):
    os.makedirs(save_dir, exist_ok=True)
    num_images = min(len(original), 8)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4), squeeze=False)
    for i in range(num_images):
        axes[0, i].imshow(original[i, 0].cpu().numpy(), cmap="jet")
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i, 0].detach().cpu().numpy(), cmap="jet")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reconstruction_epoch{epoch}.png"))
    plt.close()


def visualize_latent(csv_folder, model_path="vae_step_r_results/models", max_samples=3000, gpu_id=0):
    import seaborn as sns
    from sklearn.decomposition import PCA

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    dataset = CSVDataset2D(csv_folder)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = VAE_model.from_pretrained(model_path, map_location=device)
    model = model.to(device, dtype=torch.float32)
    model.eval()

    all_latents = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device, dtype=torch.float32)
            posterior = model.encode(imgs)
            z = posterior.mode()
            z = z.view(z.size(0), -1)
            all_latents.append(z.cpu())
            if len(all_latents) * imgs.size(0) >= max_samples:
                break

    all_latents = torch.cat(all_latents, dim=0)[:max_samples]
    lat_np = all_latents.numpy()
    plt.figure(figsize=(6, 4))
    sns.histplot(lat_np[:, 0], bins=50, kde=True)
    plt.savefig("vae_step_r_results/latent_hist.png")
    plt.close()

    pca = PCA(n_components=2)
    lat_2d = pca.fit_transform(lat_np)
    plt.figure(figsize=(6, 6))
    plt.scatter(lat_2d[:, 0], lat_2d[:, 1], s=5, alpha=0.5)
    plt.savefig("vae_step_r_results/latent_pca.png")
    plt.close()


def train_vae(csv_folder, epochs=500, batch_size=128, lr=1e-3, beta_kl=1e-4, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    dataset = CSVDataset2D(csv_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    model = VAE_model(in_channels=1, latent_channels=4,
                      use_kl=use_kl, beta_kl=beta_kl)
    model = model.to(device, dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            imgs = imgs.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                recon, mu, logvar = model(imgs, sample_posterior=use_kl)
                loss = vae_loss(recon, imgs, mu, logvar,
                                use_kl=use_kl, beta_kl=beta_kl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}: Loss={avg_loss:.6f}")
        model.eval()
        with torch.no_grad():
            sample_batch, _ = next(iter(dataloader))
            sample_batch = sample_batch.to(device, dtype=torch.float32)
            reconstructed, _, _ = model(sample_batch, sample_posterior=use_kl)
            save_reconstruction(sample_batch.cpu(), reconstructed.cpu(), epoch)
    model.save_pretrained("vae_step_r_results/models")


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    train_vae(
        csv_folder=args.csv_folder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_kl=args.use_kl,
        beta_kl=args.beta_kl,
        gpu_id=args.gpu_id
    )
    if args.visualize:
        visualize_latent(args.csv_folder, model_path="vae_step_r_results/models", max_samples=3000, gpu_id=args.gpu_id)
