from openai import OpenAI
import os
from transformers import AutoTokenizer, pipeline
import transformers 
import torch
from icl_template import *

def llm_concept_detect_removal(prompt):

    # print(templatev0_1.format(prompt))

    # openai.api_key = os.environ["OPENAI_API_KEY"]

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=os.environ["OPENAI_API_KEY"],
    )

    response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": llama2_templatev0_3.format(prompt)}
            ]
        )

    return response.choices[0].message.content.strip()

class TinyLlama:

    def __init__(self, model_path):
        model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
        self.tokenizer = AutoTokenizer.from_pretrained(model, cache_dir="/localscratch/renjie/cache")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            # cache_dir="/localscratch/renjie/cache",
        )

    def inference(self, prompt):

        prompt = "two dogs playing on the grass"

        sequences = self.pipeline(
            templatev0_3.format(prompt),
            do_sample=False,
            # top_k=10,
            # num_return_sequences=4,
            repetition_penalty=1.5,
            eos_token_id=self.tokenizer.eos_token_id,
            # max_length=2048,
            max_new_tokens=100,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
            import pdb ; pdb.set_trace()

    # return response.choices[0].message.content.strip()
            

class TinyLlamaChat:

    def __init__(self, model_path):
        self.pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

    def inference(self, prompt):

        prompt = "two dogs playing on the grass"

        messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": templatev0_3.format(prompt)},
            ]
        
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        print(outputs[0]["generated_text"])
        
        import pdb ; pdb.set_trace()

    # return response.choices[0].message.content.strip()
