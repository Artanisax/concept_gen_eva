MY_CMD="python txt2img.py --prompt prompt_unwanted --output_name test_male --seed 1 --counter_exit 100"

# MY_CMD="python icl_concept_removal.py --prompt prompt_unwanted --output_name icl --icl_model gpt --seed 1"

# MY_CMD="python gpt_step1.py"

# OPENAI_API_KEY=`cat ../openai_key.txt`

# MY_CMD="python txt2img_common.py --prompt prompt_unwanted --output_name empty --seed 2"

# MY_CMD="python -u llama2_step1.py --output_name giphy --template llama2_templatev0_3"

# MY_CMD="python dalle_openai.py --prompt prompt_engineer --output_name dalle3"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
CUDA_VISIBLE_DEVICES='5' OPENAI_API_KEY=`cat ../openai_key.txt` $MY_CMD # HF_HOME=$HF_CACHE_DIR TRANSFORMERS_CACHE=$HF_CACHE_DIR
