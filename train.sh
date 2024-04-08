export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="./results/local_prompt_male_nurse"

CUDA_VISIBLE_DEVICES="5" accelerate launch textual_inversion_real.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<nurse-person>" \
  --initializer_token="person" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=3000 \
  --learning_rate=5.0e-03 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="results/male_nurse" \
  --save_steps=500 \
  --checkpointing_steps=500 \
  --num_vectors 1
