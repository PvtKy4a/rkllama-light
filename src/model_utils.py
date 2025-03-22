import json
import os
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

def get_available_models():
    file = open(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    models_cfg = json.load(file)

    file.close()

    return models_cfg.keys()

def get_model_cfg(model_name):
    file = open(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    models_cfg = json.load(file)

    file.close()

    if not model_name in models_cfg:
        print("Incorrect model name", flush=True)
        return None

    return models_cfg[model_name]

def download_model(models_path, model_name):
    model_cfg = get_model_cfg(model_name)

    if not model_cfg:
        return False

    try:
        if not os.path.exists(models_path + "/" + model_cfg["repo_id"]):
            hf_hub_download(repo_id=model_cfg["repo_id"], filename=model_cfg["filename"], local_dir=models_path, cache_dir=models_path)
    except:
        print("Failed to download model. Check model filename.", flush=True)
        return False

    return True

def download_tokenizer(tokenizers_path, model_name):
    model_cfg = get_model_cfg(model_name)

    if not model_cfg:
        return False

    try:
        tokenizer_path = tokenizers_path + "/" + model_cfg["repo_id"].replace("/","-")

        if not os.path.exists(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(model_cfg["repo_id"], trust_remote_code=True, cache_dir=tokenizers_path)
            os.mkdir(tokenizer_path)
            tokenizer.save_pretrained(tokenizer_path)
    except:
        print("Failed to download tokenizer. Check model repo_id.", flush=True)
        return False

    return True

def get_tokenizer(tokenizers_path, model_name):
    model_cfg = get_model_cfg(model_name)

    tokenizer_path = tokenizers_path + "/" + model_cfg["repo_id"].replace("/","-")

    return AutoTokenizer.from_pretrained(tokenizer_path, cache_dir=tokenizers_path)