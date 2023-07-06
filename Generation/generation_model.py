import random
import asyncio
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from transformers import AutoTokenizer, LlamaForCausalLM,LlamaTokenizer
from datasets import load_dataset
import re


def extract_sentences(text):
    sentences = re.split(r'sentence [A-Za-z0-9]+:', text, flags=re.IGNORECASE)
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


class Generation_model():
    def __init__(self, device) -> None:
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model =  LlamaForCausalLM.from_pretrained("chavinlo/alpaca-native",low_cpu_mem_usage=True,torch_dtype='auto').to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained("chavinlo/alpaca-native")
        self.device_number = device
    def generate(self,input_samples):
        candidate_list = []
        
        
        for input_sample in input_samples:
            instruction = f"Input sentences are part of a '{input_sample[0]}' in different medical papers. Please create five senetences in '{input_sample[0]}' part put new knowledge irrelevant with input sentences. Separate each sentence like 'sentence a:'"
            input = ', '.join(["'" + sentence + "'" for sentence in input_sample[1]])
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            inputs = self.tokenizer.encode(prompt,return_tensors="pt").to(self.device)
            candidate_senetnces = self.model.generate(inputs,max_new_tokens = 300) #make to candidate format
            text = self.tokenizer.decode(*candidate_senetnces)
            candidate_list.append([input_sample[0],extract_sentences(re.split('### Response:',text)[1])])

        return candidate_list
        