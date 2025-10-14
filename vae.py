#!/usr/bin/env python3
# train_vae.py
"""
Full VAE training + save/load (pytorch + safetensors if available) + evaluation + plotting.
Modified: spatial latent maps (B, latent_channels, Hz, Wz) suitable for UNet input (e.g. 4x4x4).
"""

import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

from tqdm import tqdm
from numpy.fft import fft2

try:
    from scipy.stats import pearsonr
except Exception:
    def pearsonr(a, b):
        return (np.nan, np.nan)

try:
    from safetensors.torch import save_file as safetensors_save
    from safetensors.torch import load_file as safetensors_load
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False

def try_import_dtw():
    try:
        import dtaidistance.dtw as dtaid_tw
        return lambda x, y: dtaid_tw.distance_fast(x, y, use_pruning=True)
    except Exception:
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            return lambda x, y: fastdtw(x, y, dist=euclidean)[0]
        except Exception:
            def naive_dtw(a, b):
                n, m = len(a), len(b)
                dtw = np.full((n+1, m+1), np.inf)
                dtw[0, 0] = 0.0
                for i in range(1, n+1):
                    for j in range(1, m+1):
                        cost = abs(a[i-1] - b[j-1])
                        dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
                return float(dtw[n,m])
            return naive_dtw

dtw_distance = try_import_dtw()

class SelfAttention2D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)

        # reshape multi-head
        q = q.view(b, self.num_heads, c // self.num_heads, h * w)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w)

        attn = torch.einsum("bhcn,bhcm->bhnm", q, k) / math.sqrt(c // self.num_heads)
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum("bhnm,bhcm->bhcn", attn, v)
        out = out.reshape(b, c, h, w)
        return self.proj(out) + x
class ResBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.block(x) + x

class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),
            nn.SiLU(),
            ResBlock(32),
            ResBlock(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            ResBlock(64),
            ResBlock(64),
            SelfAttention2D(64),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            ResBlock(128),
            ResBlock(128),
            SelfAttention2D(128)
        )
        self.conv_mu = nn.Conv2d(128, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(128, latent_channels, 1)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        mu = self.conv_mu(h)
        logvar = torch.clamp(self.conv_logvar(h), -20, 10)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_channels=4):
        super().__init__()
        self.proj = nn.Conv2d(latent_channels, 128, 1)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            ResBlock(64),
            ResBlock(64),
            SelfAttention2D(64),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            ResBlock(32),
            ResBlock(32),
        )
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1)

    def forward(self, z):
        h = self.proj(z)
        h = self.deconv1(h)
        h = self.deconv2(h)
        out = self.deconv3(h)
        return out

# ======================
# VAE wrapper
# ======================
class VAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4, sample_size=32, beta=4.0):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(in_channels, latent_channels)
        self.latent_channels = latent_channels
        self.sample_size = sample_size
        self.latent_spatial = sample_size // 8
        self.beta = beta
        self.config = {
            "in_channels": in_channels,
            "latent_channels": latent_channels,
            "sample_size": sample_size
        }
    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar, normalize_latent=True):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Bỏ normalization latent space
        return z

    def decode(self, z, apply_sigmoid=False):
        out = self.decoder(z)
        return torch.sigmoid(out) if apply_sigmoid else out

    def forward(self, x, apply_sigmoid=False, normalize_latent=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, normalize_latent)
        x_recon = self.decode(z, apply_sigmoid)
        return x_recon, mu, logvar

    def get_latent(self, x, deterministic=False, normalize_latent=True):
        mu, logvar = self.encode(x)
        return mu if deterministic else self.reparameterize(mu, logvar, normalize_latent)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        state_dict = self.state_dict()
        torch.save({k: v.cpu() for k, v in state_dict.items()}, os.path.join(save_directory, "pytorch_model.bin"))
        if HAS_SAFETENSORS:
            safe_path = os.path.join(save_directory, "diffusion_pytorch_model.safetensors")
            safe_dict = {k: v.cpu() for k, v in state_dict.items()}
            safetensors_save(safe_dict, safe_path)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory, map_location=None, strict_load=False):
        cfg_path = os.path.join(load_directory, "config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"config.json not found in {load_directory}")
        with open(cfg_path, "r") as f:
            config = json.load(f)
        try:
            sig = inspect.signature(cls.__init__)
            valid_params = set([p.name for p in list(sig.parameters.values())[1:]])
            filtered = {k: v for k, v in config.items() if k in valid_params}
        except Exception:
            filtered = {k: v for k, v in config.items() if not k.startswith("_")}
        model = cls(**filtered)
        map_loc = map_location or "cpu"
        safe_path = os.path.join(load_directory, "diffusion_pytorch_model.safetensors")
        bin_path = os.path.join(load_directory, "pytorch_model.bin")
        loaded_state = None
        if HAS_SAFETENSORS and os.path.isfile(safe_path):
            loaded = safetensors_load(safe_path, device=None)
            loaded_state = {}
            for k, v in loaded.items():
                if isinstance(v, np.ndarray):
                    t = torch.from_numpy(v)
                else:
                    t = v
                loaded_state[k] = t.to(map_loc)
        elif os.path.isfile(bin_path):
            loaded = torch.load(bin_path, map_location=map_loc)
            if isinstance(loaded, dict) and ("state_dict" in loaded or "model_state_dict" in loaded):
                candidate = loaded.get("state_dict", loaded.get("model_state_dict"))
                loaded_state = {k: v.to(map_loc) for k, v in candidate.items()}
            else:
                loaded_state = {k: v.to(map_loc) for k, v in loaded.items()}
        else:
            raise FileNotFoundError("No model file found in " + load_directory)
        try:
            model.load_state_dict(loaded_state, strict=strict_load)
        except RuntimeError as e:
            print("Warning: strict load failed:", e)
            print("Retrying with strict=False")
            model.load_state_dict(loaded_state, strict=False)
        return model

class CSVDataset2D(Dataset):
    def __init__(self, csv_folder: str, input_shape=(32, 32)):
        self.label_names = ["ellipse", "rectangular", "step_r", "step_t", "triangular"]
        self.label2idx = {name: i for i, name in enumerate(self.label_names)}
        self.csv_files = []
        self.labels = []
        for label_name in self.label_names:
            label_dir = os.path.join(csv_folder, label_name)
            if not os.path.isdir(label_dir):
                continue
            for f in os.listdir(label_dir):
                if f.endswith(".csv"):
                    self.csv_files.append(os.path.join(label_dir, f))
                    self.labels.append(self.label2idx[label_name])
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {csv_folder}")
        self.input_shape = input_shape
        sample_img = np.loadtxt(self.csv_files[0], delimiter=",", dtype=np.float32)
        if sample_img.shape != input_shape:
            raise ValueError(f"Data shape {sample_img.shape} does not match expected shape {input_shape}")

    def __len__(self):
        return len(self.csv_files)

    def __getitem__(self, idx):
        img = np.loadtxt(self.csv_files[idx], delimiter=",", dtype=np.float32)
        img = img.reshape(self.input_shape)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        label = self.labels[idx]
        return img_tensor, label

# ======================
# Loss & beta schedule
# ======================
# def vae_loss_function(recon_x, x, mu, logvar, beta):
#     recon_loss = F.mse_loss(recon_x, x, reduction='mean')
#     kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#     # sửa view -> reshape
#     kl_per_image = kl_elementwise.reshape(kl_elementwise.size(0), -1).sum(dim=1)  # (B,)
#     kl_div = kl_per_image.mean()
#     total_loss = recon_loss + beta * kl_div
#     return total_loss, recon_loss, kl_div
# def vae_loss_function(recon_x, x, mu, logvar, beta, free_bits=0.05):
#     # reconstruction loss
#     recon_loss = F.mse_loss(recon_x, x, reduction='mean')

#     # KL divergence per element
#     kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D,...)

#     # gộp các chiều lại per image
#     kl_per_image = kl_elementwise.reshape(kl_elementwise.size(0), -1).sum(dim=1)  # (B,)

#     # KL Free-bits: ép mỗi chiều KL >= free_bits
#     kl_per_dim = kl_elementwise.mean(0)  # trung bình theo batch cho từng chiều
#     kl_free = torch.clamp(kl_per_dim, min=free_bits).sum()

#     # chọn dùng KL đã free-bits thay cho KL trung bình
#     kl_div = kl_free

#     # tổng loss
#     total_loss = recon_loss + beta * kl_div
#     return total_loss, recon_loss, kl_div
# def vae_loss_function(recon_x, x, mu, logvar, beta=1.0, free_bits=0.1):
#     recon_loss = F.mse_loss(recon_x, x, reduction='mean')

#     # KL Divergence
#     kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
#     kl = torch.mean(torch.clamp(kl, min=free_bits))  # free-bits
#     loss = recon_loss + beta * kl
#     return loss, recon_loss, kl

def vae_loss_function(recon_x, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_div = kl_elementwise.mean()
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss, kl_div

def get_beta(epoch, beta_start=0.1, beta_end=0.2, warmup_epochs=40):
    scale = beta_end - beta_start
    beta = scale / (1 + math.exp(-0.3 * (epoch - warmup_epochs / 2))) + beta_start
    return beta
# def get_beta(epoch, beta_start=0.0, beta_end=1.0, hold_epochs=10, warmup_epochs=20):
#     if epoch < hold_epochs:
#         return 0.0
#     progress = (epoch - hold_epochs) / max(1, warmup_epochs)
#     progress = min(progress, 1.0)
#     # sigmoid schedule (tăng chậm lúc đầu, nhanh dần sau)
#     return beta_start + (beta_end - beta_start) * (1 / (1 + np.exp(-10 * (progress - 0.5))))


def compute_psnr(original, reconstructed, max_val=1.0):
    mse = ((original - reconstructed) ** 2).mean()
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr

def compute_fft_mse(x, y):
    x_fft = np.abs(fft2(x))
    y_fft = np.abs(fft2(y))
    return np.mean((x_fft - y_fft) ** 2)

def compute_pearson(x, y):
    try:
        r, _ = pearsonr(x.flatten(), y.flatten())
        if np.isnan(r):
            return None
        return r
    except Exception:
        return None

def compute_dtw(x, y):
    try:
        xa = np.array(x.flatten(), dtype=np.double)
        ya = np.array(y.flatten(), dtype=np.double)
        return float(dtw_distance(xa, ya))
    except Exception:
        return float('nan')

def save_reconstruction(original, reconstructed, epoch, save_dir="figures_2d"):
    os.makedirs(save_dir, exist_ok=True)
    num_images = min(original.shape[0], 8)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4), squeeze=False)
    for i in range(num_images):
        orig_img = original[i, 0].cpu().numpy()
        recon_img = reconstructed[i, 0].detach().cpu().numpy()
        axes[0, i].imshow(orig_img, cmap="jet")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon_img, cmap="jet")
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"reconstruction_epoch{epoch}.png"))
    plt.close()

def main():
    save_dir = "./vae12"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures_2d"), exist_ok=True)

    csv_folder = "/home/dat.lt19010205/CongDuc/diffusers/newdata/all"  
    input_shape = (32, 32)
    batch_size = 32
    num_epochs = 50

    # Dataset
    full_dataset = CSVDataset2D(csv_folder, input_shape=input_shape)
    total_size = len(full_dataset)
    test_size = int(0.2 * total_size)
    train_size = total_size - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model + Optimizer + Scheduler
    model = VAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, threshold=1e-3
    )

    history = {"total": [], "recon": [], "kl": [], "beta": [], 
               "val_total": [], "val_recon": [], "val_kl": []}

    best_val = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = total_recon = total_kl = 0.0
        beta = get_beta(epoch, beta_start=0.0, beta_end=1.0, warmup_epochs=30)

        # -------- Training --------
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if isinstance(batch, (list, tuple)):
                inputs, _ = batch
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs, mu, logvar = model(inputs, apply_sigmoid=False)
            loss, recon_loss, kl_div = vae_loss_function(outputs, inputs, mu, logvar, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = inputs.size(0)
            total_loss += loss.item() * bs
            total_recon += recon_loss.item() * bs
            total_kl += kl_div.item() * bs

        avg_loss = total_loss / len(train_dataloader.dataset)
        avg_recon = total_recon / len(train_dataloader.dataset)
        avg_kl = total_kl / len(train_dataloader.dataset)

        # -------- Validation --------
        model.eval()
        val_total = val_recon = val_kl = 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                if isinstance(batch, (list, tuple)):
                    inputs, _ = batch
                else:
                    inputs = batch
                inputs = inputs.to(device)
                outputs, mu, logvar = model(inputs, apply_sigmoid=False)
                loss, recon_loss, kl_div = vae_loss_function(outputs, inputs, mu, logvar, beta=beta)
                bs = inputs.size(0)
                val_total += loss.item() * bs
                val_recon += recon_loss.item() * bs
                val_kl += kl_div.item() * bs

        val_avg = val_total / len(test_dataloader.dataset)
        val_recon_avg = val_recon / len(test_dataloader.dataset)
        val_kl_avg = val_kl / len(test_dataloader.dataset)

        # Step scheduler with validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_avg)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"⚠️ LR reduced from {old_lr:.6f} → {new_lr:.6f}")

        # Save history
        history["total"].append(avg_loss)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)
        history["beta"].append(beta)
        history["val_total"].append(val_avg)
        history["val_recon"].append(val_recon_avg)
        history["val_kl"].append(val_kl_avg)

        # Logging
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Beta: {beta:.4f} | Train Total: {avg_loss:.6f} | Recon: {avg_recon:.6f} | KL: {avg_kl:.6f} "
              f"|| Val Total: {val_avg:.6f} | Val Recon: {val_recon_avg:.6f} | Val KL: {val_kl_avg:.6f}")

        # Save best model
        if val_avg < best_val:
            best_val = val_avg
            torch.save(model.state_dict(), os.path.join(save_dir, "best_vae.pth"))
            print(f"✅ Model saved at epoch {epoch+1} with val_loss={val_avg:.6f}")

        # Save one reconstruction figure
        with torch.no_grad():
            sample_batch = next(iter(test_dataloader))
            if isinstance(sample_batch, (list, tuple)):
                sample_inputs, _ = sample_batch
            else:
                sample_inputs = sample_batch
            sample_inputs = sample_inputs.to(device)
            outs, _, _ = model(sample_inputs, apply_sigmoid=False)
            save_reconstruction(sample_inputs, outs, epoch+1, save_dir=os.path.join(save_dir, "figures_2d"))

    # save model
    model.save_pretrained(save_dir)
    print("Model saved to", save_dir)

    # plot losses
    try:
        plt.figure(figsize=(6,4))
        plt.plot(history["total"], label="train_total_loss")
        plt.plot(history["val_total"], label="val_total_loss")
        plt.plot(history["recon"], label="train_recon_loss")
        plt.plot(history["kl"], label="train_kl_div")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "loss_history.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    model.eval()
    all_mse = []
    all_psnr = []
    all_latents = []
    all_pearson = []
    all_dtw = []
    all_fft_mse = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Evaluating on test set")):
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch
            else:
                inputs = batch
                labels = None
            inputs = inputs.to(device)
            outputs, mu, logvar = model(inputs, apply_sigmoid=False)
            latent = model.get_latent(inputs, deterministic=False)
            inputs_np = inputs.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            latents_np = latent.cpu().numpy()
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels_np = labels.cpu().numpy()
                else:
                    labels_np = np.array(labels)
            else:
                labels_np = np.zeros(inputs_np.shape[0])
            for i in range(inputs_np.shape[0]):
                original = inputs_np[i, 0]
                reconstructed = outputs_np[i, 0]
                mse = mean_squared_error(original.flatten(), reconstructed.flatten())
                psnr = compute_psnr(original, reconstructed)
                pear = compute_pearson(original, reconstructed)
                dtw_dist = compute_dtw(original, reconstructed)
                fft_mse = compute_fft_mse(original, reconstructed)
                all_mse.append(mse)          
                all_psnr.append(psnr)
                all_pearson.append(pear)
                all_dtw.append(dtw_dist)
                all_fft_mse.append(fft_mse)
                all_latents.append(latents_np[i].flatten())
                all_labels.append(labels_np[i])

    avg_mse = np.mean(all_mse) if all_mse else float("nan")
    avg_psnr = np.mean(all_psnr) if all_psnr else float("nan")
    valid_pearson = [p for p in all_pearson if p is not None]
    avg_pearson = np.mean(valid_pearson) if valid_pearson else float("nan")
    avg_dtw = np.nanmean(all_dtw) if all_dtw else float("nan")
    avg_fft_mse = np.mean(all_fft_mse) if all_fft_mse else float("nan")

    print("\n📊 Evaluation Metrics on Test Set")
    print(f"🧮 Mean MSE:           {avg_mse:.6f}")
    print(f"📈 Average PSNR:       {avg_psnr:.2f} dB")
    print(f"🔗 Average Pearson r:  {avg_pearson:.4f}")
    print(f"📏 Average DTW:        {avg_dtw:.4f}")
    print(f"🎚️ Average FFT MSE:    {avg_fft_mse:.6f}")

    all_latents_arr = np.array(all_latents) if all_latents else np.zeros((0, model.latent_channels * model.latent_spatial * model.latent_spatial))

    if all_latents_arr.size > 0:
        try:
            pca = PCA(n_components=2)
            z_pca = pca.fit_transform(all_latents_arr)
            all_labels_arr = np.array(all_labels)
            label_names = ["ellipse", "rectangular", "step_r", "step_t", "triangular"]
            plt.figure(figsize=(6,5))
            for i, name in enumerate(label_names):
                idxs = all_labels_arr == i
                plt.scatter(z_pca[idxs,0], z_pca[idxs,1], s=12, alpha=0.7, label=name)
            plt.title("Latent Space PCA by Crack Type")
            plt.xlabel("PC1"); plt.ylabel("PC2")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "latent_space_pca2_by_label.png"), dpi=300)
            plt.close()
        except Exception as e:
            print("PCA plotting failed:", e)

        latent_means = all_latents_arr.mean(axis=1)
        plt.figure(figsize=(5,4))
        plt.hist(latent_means, bins=50)
        plt.title("Distribution of Mean Latent Values")
        plt.xlabel("Mean latent value"); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "latent_mean_distribution2.png"), dpi=300)
        plt.close()
    else:
        print("No latent data for PCA/histogram.")

    metrics = {
        "MSE": avg_mse,
        "DTW": avg_dtw,
        "FFT MSE": avg_fft_mse,
        "PSNR (dB)": avg_psnr,
        "Pearson r": avg_pearson
    }

    group1 = {"MSE": metrics["MSE"]}
    group2 = {k: metrics[k] for k in ["DTW", "FFT MSE"]}
    group3 = {k: metrics[k] for k in ["PSNR (dB)", "Pearson r"]}

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(5,8))
    def plot_bar(ax, group, title):
        names = list(group.keys()); values = list(group.values())
        bars = ax.bar(names, values, width=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, (bar.get_height() if not np.isnan(val) else 0)*1.01,
                    f"{val:.4f}" if (not np.isnan(val) and val < 1) else (f"{val:.2f}" if not np.isnan(val) else "nan"),
                    ha='center', va='bottom', fontsize=10)
        ax.set_title(title); ax.set_ylabel("Value"); ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        finite_vals = [v for v in values if not (isinstance(v,float) and (math.isnan(v) or math.isinf(v)))]
        if finite_vals:
            ax.set_ylim(0, max(finite_vals)*1.25)

    plot_bar(ax1, group1, "MSE")
    plot_bar(ax2, group2, "DTW & FFT MSE")
    plot_bar(ax3, group3, "PSNR & Pearson r")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(os.path.join(save_dir, "vae_metrics_bar_chart.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
