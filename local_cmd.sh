# MY_CMD="python txt2img.py --prompt prompt_i2p --output_name i2p --seed 1"

# MY_CMD="python icl_concept_removal.py --prompt prompt_unwanted --output_name icl --icl_model gpt --seed 1"

# MY_CMD="python gpt_step1.py"

# OPENAI_API_KEY=`cat ../openai_key.txt`

# MY_CMD="python txt2img_common.py --prompt prompt_i2p --output_name i2p --seed 2"

MY_CMD="python -u llama2_step1.py --output_name giphy --template llama2_templatev0_3"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='7' OPENAI_API_KEY=`cat ../openai_key.txt` $MY_CMD # HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR
