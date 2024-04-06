export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="./cat"

CUDA_VISIBLE_DEVICES="7" accelerate launch textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cat-toy>" \
  --initializer_token="toy" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=100 \
  --learning_rate=5.0e-03 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="results/teddy_100step_lr5e-3_endMSE_no_teddy" \
  --save_steps=10 \
  --checkpointing_steps=100 \
  --num_vectors 2