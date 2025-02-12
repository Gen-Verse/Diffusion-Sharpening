export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATASET_NAME="your_dataset_name"

accelerate launch --main_process_port 10950 train_sft_diffusion_sharpen_sdxl.py  \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --enable_xformers_memory_efficient_attention \
  --resolution=512 --center_crop --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=10000 \
  --use_8bit_adam \
  --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --report_to="wandb" \
  --validation_prompt="a cute Sundar Pichai creature" --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="sdxl-naruto-model" 