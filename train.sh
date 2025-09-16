CUDA_VISIBLE_DEVICES=6 python3 train.py \
  --pretrained_model_name_or_path="/home/dat.lt19010205/CongDuc/diffusers/pretrained_models/stable-diffusion-v1-5" \
  --train_jsonl_file="/home/dat.lt19010205/CongDuc/diffusers/step_t.jsonl" \
  --csv_dir="/home/dat.lt19010205/CongDuc/diffusers/data/normalized_data" \
  --output_dir="/home/dat.lt19010205/CongDuc/diffusers/step_t" \
  --resolution=32 \
  --train_batch_size=256 \
  --num_train_epochs=10000 \
  --learning_rate=1e-5 \
  # --validation_prompts "shape: Ellipse, width: 0.3, length: 2.0, depth: 0.1, angle: 0, x_offset: 0, y_offset: 0" \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=13000 
  # --mixed_precision="fp16" \
  # --gradient_accumulation_steps=4 \
  # --gradient_checkpointing \
  # --use_8bit_adam \
  # --image_column="image" \
  # --caption_column="text"