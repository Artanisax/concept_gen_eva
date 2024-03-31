MY_CMD="python txt2img.py --prompt prompt_unwanted --output_name debug_not_full_model --seed 1"

# MY_CMD="python txt2img_common.py --prompt prompt_unwanted --output_name debug_base --seed 1"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='1' $MY_CMD # HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR
