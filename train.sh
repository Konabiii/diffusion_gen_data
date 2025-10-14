#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==== Base model (pretrained) ====
BASE_MODEL="/home/dat.lt19010205/CongDuc/diffusers/pretrained_models/stable-diffusion-v1-5"

# ==== Common hyperparameters ====
RESOLUTION=32
BATCH_SIZE=32
EPOCHS=1000
LR=1e-5
SEED=42

# ==== Dataset paths ====
JSONL_LIST=(
"/home/dat.lt19010205/CongDuc/diffusers/newdata2/data_shape_first.jsonl"
)

CSV_DIR_LIST=(
"/home/dat.lt19010205/CongDuc/diffusers/newdata2/data"
)

# ==== SỬA LỖI: Thêm một dấu gạch dưới để khớp với thư mục thực tế ====
OUTPUT_BASE="/home/dat.lt19010205/CongDuc/diffusers/Version__"

# ==== Train models ====
# Vòng lặp chỉ chạy 1 lần vì chỉ có 1 dataset
for i in {0..0}; do
  # Xác định thư mục output cho lần chạy hiện tại
  OUTPUT_DIR="${OUTPUT_BASE}$((i+1))"

  echo "===================================="
  
  # Lần chạy này sẽ resume từ checkpoint
  # SỬA LỖI: Đảm bảo đường dẫn này chính xác
  # CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-221000"
  
  echo "🚀 Resuming training run $((i+1)) from checkpoint"
  echo "Dataset: ${JSONL_LIST[$i]}"
  echo "CSV dir: ${CSV_DIR_LIST[$i]}"
  echo "Output: $OUTPUT_DIR"
  echo "Checkpoint: $CHECKPOINT_PATH"
  echo "===================================="

  python3 train6.py \
    --pretrained_model_name_or_path="$BASE_MODEL" \
    --train_jsonl_file="${JSONL_LIST[$i]}" \
    --csv_dir="${CSV_DIR_LIST[$i]}" \
    --output_dir="$OUTPUT_DIR" \
    --resolution=$RESOLUTION \
    --train_batch_size=$BATCH_SIZE \
    --num_train_epochs=$EPOCHS \
    --learning_rate=$LR \
    --image_column="image" \
    --caption_column="text" \
    --random_flip \
    --mixed_precision="fp16" \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=13000 \
    --checkpoints_total_limit=3 \
    # --resume_from_checkpoint="$CHECKPOINT_PATH"

  echo "✅ Finished training run $((i+1))"
done

echo "🎯 All training runs completed successfully!"