# Helper functions for querying OpenAI API models
import openai
import tiktoken
import os

from huggingface_hub import login as hf_login

def tok2id(tok, tokenizer_model="p50k_base"):
    encoding = tiktoken.get_encoding(tokenizer_model)
    return encoding.encode(tok)

def set_key_from_file(key_file):
    with open(key_file, "r") as fp:
        key = fp.read()
    openai.api_key = key

# HuggingFace token setup for gated models like Llama 2
def set_hf_token_from_file(token_file):
    with open(token_file, "r") as fp:
        token = fp.read().strip()
    os.environ["HF_TOKEN"] = token
    hf_login(token=token)
