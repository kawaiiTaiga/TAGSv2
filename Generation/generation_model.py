import random
import asyncio
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from transformers import AutoTokenizer, LlamaForCausalLM,LlamaTokenizer
from datasets import load_dataset

class Generation_model():
    def __init__(self, device) -> None:
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model =  LlamaForCausalLM.from_pretrained("chavinlo/alpaca-native",low_cpu_mem_usage=True,torch_dtype='auto').to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")
        self.device_number = device
    def generate(self,input_samples):
        candidate_list = []
        placeholders  = "'{}',"* 1
        formatted_prompt = f"prompt promp" + placeholders + "PROMPT PROMPT {}".format(input_samples) 
        print(input_samples)
        for input_sample in input_samples:
            prompt = formatted_prompt.format(*input_sample[1])
            inputs = self.tokenizer.encode(formatted_prompt,return_tensors="pt").to(self.device)
            candidate_senetnces = self.model.generate(inputs,max_new_tokens = 30) #make to candidate format
            text = self.tokenizer.decode(*candidate_senetnces)
            candidate_list.append([input_sample[0],text])

        return candidate_list
        