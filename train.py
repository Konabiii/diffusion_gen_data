import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import json
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms
import accelerate
from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
import torchvision.transforms.functional as TF
from functools import partial
import diffusers
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vae import VAE

if is_wandb_available():
    import wandb

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

check_min_version("0.35.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    vae.eval()
    text_encoder.eval()
    unet.eval()

    validation_prompt = args.validation_prompts[0] if args.validation_prompts else "shape: Ellipse, width: 0.3, length: 2.0, depth: 0.1, angle: 0, x_offset: 0, y_offset: 0"

    text_inputs = tokenizer(
        validation_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = text_inputs.input_ids.to(accelerator.device)

    with torch.no_grad():
        text_embeddings = text_encoder(input_ids)[0]

    latents = torch.randn(
        (1, vae.config['latent_channels'], 4, 4),
        device=accelerator.device,
        dtype=weight_dtype
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(50)
    latents = latents * noise_scheduler.init_noise_sigma

    for t in noise_scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2)
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([text_embeddings, torch.zeros_like(text_embeddings)]),
                return_dict=False
            )[0]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    with torch.no_grad():
        reconstructed = vae.decode(latents)

    if reconstructed.shape[2:] != (args.resolution, args.resolution):
        logger.warning(f"Validation image size {reconstructed.shape[2:]} does not match expected ({args.resolution}, {args.resolution})")

    if accelerator.is_main_process:
        reconstructed = reconstructed.cpu()
        plt.figure(figsize=(4, 4))
        plt.imshow(reconstructed[0, 0].detach().numpy(), cmap="jet", origin="upper")
        plt.colorbar()
        plt.axis("off")
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f"validation_epoch_{epoch}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved validation image to {save_path}")

    return []

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Stable Diffusion with custom VAE for 32x32 images.")
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files, e.g., fp16",
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        help="Revision of pretrained non-ema model identifier.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from HuggingFace hub) or a local path.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--train_jsonl_file",
        type=str,
        default=None,
        help="Path to the JSONL file containing file_name (CSV paths) and text (captions).",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="Base directory for CSV files listed in the JSONL file.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image path or array.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging, truncate the number of training examples.",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help="Prompts evaluated every `--validation_epochs` and logged. Set to None to skip validation.",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Skip cropping images during preprocessing.",
    )
    parser.add_argument(
        "--validation_data_dir",
        type=str,
        default=None,
        help="A folder containing the validation CSVs (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory for model predictions and checkpoints.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory for downloaded models and datasets.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="The integration to report results and logs to.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help="The project_name for Accelerator.init_trackers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="The resolution for input images (must be 32 for custom VAE).",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop the input images.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="Whether to randomly flip images horizontally.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for training.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale learning rate by GPUs, gradient accumulation, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma for loss rebalancing.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help="Use DREAM training method.",
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use 8-bit Adam optimizer.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model.",
    )
    parser.add_argument(
        "--offload_ema",
        action="store_true",
        help="Offload EMA model to CPU during training.",
    )
    parser.add_argument(
        "--foreach_ema",
        action="store_true",
        help="Use faster foreach implementation of EMAModel.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type for training: 'epsilon' or 'v_prediction'.",
    )
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=0,
        help="The scale of input perturbation.",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0,
        help="The scale of noise offset.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with output_dir.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help="Save a checkpoint every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from a previous checkpoint.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Use xformers for memory-efficient attention.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Use mixed precision: fp16 or bf16.",
    )
    parser.add_argument(
        "--image_interpolation_mode",
        type=str,
        default="lanczos",
        choices=[f.lower() for f in dir(transforms.InterpolationMode) if not f.startswith("__") and not f.endswith("__")],
        help="The image interpolation method for resizing.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None and args.train_jsonl_file is None:
        raise ValueError("Need either a dataset name, a training folder, or a JSONL file with CSV directory.")
    if args.train_jsonl_file is not None and args.csv_dir is None:
        raise ValueError("Must provide --csv_dir when using --train_jsonl_file.")
    
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    if args.resolution != 32:
        raise ValueError("Resolution must be 32 to match custom VAE configuration.")

    return args

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, args.logging_dir)),
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Cannot use both --report_to=wandb and --hub_token.")
    
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        logging.getLogger("accelerate").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("diffusers").setLevel(logging.INFO)
    else:
        logging.getLogger("accelerate").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("diffusers").setLevel(logging.ERROR)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        return [] if deepspeed_plugin is None else [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = VAE.from_pretrained("/home/dat.lt19010205/CongDuc/diffusers/vae12")
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )

    if unet.config.sample_size != args.resolution:
        logger.warning(f"UNet sample_size ({unet.config.sample_size}) does not match resolution ({args.resolution}). Adjusting UNet configuration.")
        unet.register_to_config(sample_size=args.resolution)

    if vae.config['sample_size'] != args.resolution:
        raise ValueError(f"VAE sample_size ({vae.config['sample_size']}) does not match resolution ({args.resolution}).")
    if vae.config['latent_channels'] != unet.config.in_channels:
        raise ValueError(f"VAE latent_channels ({vae.config['latent_channels']}) does not match UNet in_channels ({unet.config.in_channels}).")

    vae.requires_grad_(False)    
    text_encoder.requires_grad_(False)
    unet.train()

    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=args.foreach_ema,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may cause issues. Update to at least 0.0.17.")
            unet.enable_xformers_memory_efficient_attention()
        else:
            logger.warning("xformers is not available. Install it with `pip install xformers` to enable memory-efficient attention.")

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, foreach=args.foreach_ema
                )
                ema_unet.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_unet.pin_memory()
                else:
                    ema_unet.to(accelerator.device)
                del load_model
            for _ in range(len(models)):
                model = models.pop()
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        if bnb is None:
            raise ImportError("bitsandbytes is required for --use_8bit_adam. Install with `pip install bitsandbytes`.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    def load_csv_as_tensor(csv_path, expected_resolution=32):
        arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        if arr.shape != (expected_resolution, expected_resolution):
            raise ValueError(f"CSV file {csv_path} has shape {arr.shape}, expected ({expected_resolution}, {expected_resolution})")
        tensor = torch.from_numpy(arr).unsqueeze(0) 
        return tensor

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            data_dir=args.train_data_dir,
        )
    elif args.train_jsonl_file is not None:
        jsonl_data = pd.read_json(args.train_jsonl_file, lines=True)
        data = []
        for _, row in jsonl_data.iterrows():
            csv_path = os.path.join(args.csv_dir, row["file_name"])
            if os.path.exists(csv_path):
                data.append({"image": csv_path, "text": row["text"]})
        train_dataset = Dataset.from_list(data)
        dataset = DatasetDict({"train": train_dataset})
        if args.validation_data_dir is not None:
            logger.warning("Validation data dir provided but no JSONL for validation; ignoring.")
    else:
        data_files = {}
        if args.train_data_dir is not None:
            train_csvs = glob.glob(os.path.join(args.train_data_dir, "*.csv"))
            if len(train_csvs) == 0:
                raise ValueError("No CSV files found in train_data_dir")
            first_df = pd.read_csv(train_csvs[0], header=None)
            num_cols = first_df.shape[1]
            if num_cols != args.resolution:
                raise ValueError(f"CSV files in train_data_dir have {num_cols} columns, expected {args.resolution}")
            column_names = [f"feature_{i}" for i in range(num_cols)]
            data_files["train"] = train_csvs
        if args.validation_data_dir is not None:
            val_csvs = glob.glob(os.path.join(args.validation_data_dir, "*.csv"))
            data_files["validation"] = val_csvs
        features = Features({name: Value("float64") for name in column_names})
        dataset = load_dataset(
            "csv",
            data_files=data_files,
            column_names=column_names,
            features=features,
            cache_dir=args.cache_dir,
        )

    column_names = dataset["train"].column_names
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)

    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs.input_ids

    interpolation = getattr(transforms.InterpolationMode, args.image_interpolation_mode.upper())
    train_transforms = transforms.Compose([
    transforms.Resize(args.resolution, interpolation=interpolation),
    transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
    transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
    ])

    def preprocess_train(examples):
        csv_paths = examples[image_column]
        captions = examples[caption_column]
        images = []

        for i, path in enumerate(csv_paths):
            current_caption = captions[i]
            
            # if i < 3 and accelerator.is_main_process: 
            #     logger.info(f"DEBUG DATA PAIRING | CSV File: '{os.path.basename(path)}', Text: '{current_caption}'")

            img_tensor = load_csv_as_tensor(path, args.resolution)
            transformed_img = train_transforms(img_tensor)
            images.append(transformed_img)

        examples["pixel_values"] = images
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples, weight_dtype=torch.float32):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=partial(collate_fn, weight_dtype=weight_dtype),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # START: DEBUG CODE BLOCK
            # if step == 0 and accelerator.is_main_process:
            #     try:
            #         first_input_ids = batch["input_ids"][0]
            #         decoded_text = tokenizer.decode(first_input_ids, skip_special_tokens=True)
            #         logger.info("="*60)
            #         logger.info("DEBUG: VERIFYING TEXT INPUT TO CLIP TEXT ENCODER (Epoch {})".format(epoch))
            #         logger.info(f"  -> Input IDs (first sample): {first_input_ids.tolist()}")
            #         logger.info(f"  -> Decoded Text from IDs: '{decoded_text.strip()}'")
            #         logger.info("="*60)
            #     except Exception as e:
            #         logger.warning(f"DEBUG: Could not decode input_ids for verification. Error: {e}")
            # END: DEBUG CODE BLOCK

            with accelerator.accumulate(unet):
                latent_mean, latent_logvar = vae.encode(batch["pixel_values"].to(weight_dtype))
                latents = vae.reparameterize(latent_mean, latent_logvar)

                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                else:
                    new_noise = noise
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)

                with torch.autocast(device_type='cuda', dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                    encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device), return_dict=False)[0]

                if args.prediction_type is not None:
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.dream_training:
                    noisy_latents, target = compute_dream_and_update_latents(
                        unet,
                        noise_scheduler,
                        timesteps,
                        noise,
                        noisy_latents,
                        target,
                        encoder_hidden_states,
                        args.dream_detail_preservation,
                    )

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema_unet.to(device="cuda", non_blocking=True)
                    ema_unet.step(unet.parameters())
                    if args.offload_ema:
                        ema_unet.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 or global_step >= args.max_train_steps:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

        if accelerator.is_main_process and args.validation_prompts is not None and (epoch % args.validation_epochs == 0 or epoch == args.num_train_epochs - 1):
            if args.use_ema:
                ema_unet.store(unet.parameters())
                ema_unet.copy_to(unet.parameters())
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                unet,
                args,
                accelerator,
                weight_dtype,
                epoch,
            )
            if args.use_ema:
                ema_unet.restore(unet.parameters())

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        os.makedirs(args.output_dir, exist_ok=True)
        
        unet.save_pretrained(os.path.join(args.output_dir, "unet"))
        text_encoder.save_pretrained(os.path.join(args.output_dir, "text_encoder"))
        vae.save_pretrained(os.path.join(args.output_dir, "vae"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
        noise_scheduler.save_pretrained(os.path.join(args.output_dir, "scheduler"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

if __name__ == "__main__":
    main()