import json
import os
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

def get_available_models():
    file = open('models_cfg.json')

    models_cfg = json.load(file)

    file.close()
    
    return models_cfg.keys()

def download_model(models_path, model_name):
    file = open('models_cfg.json')
    ret = True

    models_cfg = json.load(file)

    file.close()

    try:
        hf_hub_download(repo_id=models_cfg[model_name]["repo_id"], filename=models_cfg[model_name]["filename"], local_dir=models_path)
    except:
        ret = False

    return ret

def get_model(name):
    file = open('models_cfg.json')

    models_cfg = json.load(file)

    file.close()

    return models_cfg[name]

def download_tokenizer(models_path, model_name):
    file = open('models_cfg.json')
    ret = True

    models_cfg = json.load(file)

    file.close()

    tokenizer_path = models_path + "/" + models_cfg[model_name]["repo_id"].replace("/","-")

    try:
        if not os.path.exists(tokenizer_path):
            os.mkdir(tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(models_cfg[model_name]["repo_id"], trust_remote_code=True)
            tokenizer.save_pretrained(tokenizer_path)
    except:
        ret = False

    return ret

def get_tokenizer(models_path, model):
    tokenizer_path = models_path + "/" + model["repo_id"].replace("/","-")

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)