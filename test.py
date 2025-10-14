import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, DDPMScheduler
import matplotlib.pyplot as plt
from vae import VAE 

device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "/home/dat.lt19010205/CongDuc/diffusers/neww_1"  

# Load models
vae = VAE.from_pretrained("/home/dat.lt19010205/CongDuc/diffusers/neww_1/vae").to(device)
unet = UNet2DConditionModel.from_pretrained(f"{output_dir}/unet").to(device)
text_encoder = CLIPTextModel.from_pretrained(f"{output_dir}/text_encoder").to(device)
tokenizer = CLIPTokenizer.from_pretrained(f"{output_dir}/tokenizer")
scheduler = DDPMScheduler.from_pretrained(f"{output_dir}/scheduler")

vae.eval()
unet.eval()
text_encoder.eval()

# Prompt để test
prompt = "an ellipse shape with width 0.3, length 13.0, and depth 2.27"
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)
input_ids = text_inputs.input_ids.to(device)

with torch.no_grad():
    text_embeddings = text_encoder(input_ids)[0]

# Sinh latent noise
latent_channels = vae.config["latent_channels"]
resolution = vae.config["sample_size"]
weight_dtype = torch.float32  # thêm dòng này để định nghĩa dtype

latents = torch.randn(
    (1, latent_channels, 4, 4),
    device=device,
    dtype=weight_dtype
)
scheduler.set_timesteps(100)
latents = latents * scheduler.init_noise_sigma

# Diffusion sampling loop
for t in scheduler.timesteps:
    latent_model_input = torch.cat([latents] * 2)
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=torch.cat([text_embeddings, torch.zeros_like(text_embeddings)]),
            return_dict=False
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 5.0 * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

# Decode latent thành ảnh
with torch.no_grad():
    reconstructed = vae.decode(latents)

img = reconstructed[0, 0].detach().cpu().numpy()
plt.figure(figsize=(4, 4))
plt.imshow(img, cmap="jet", origin="upper")
plt.colorbar()
plt.axis("on")
plt.tight_layout()
plt.savefig(f"{output_dir}/test_sample.png")
plt.show()
print(f"✅ Saved test image to {output_dir}/test_sample.png")