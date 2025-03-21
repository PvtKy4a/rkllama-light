import json
import os
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

def get_available_models():
    file = open(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    models_cfg = json.load(file)

    file.close()

    return models_cfg.keys()

def download_model(models_path, model_name):
    ret = True
    file = open(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    models_cfg = json.load(file)

    file.close()

    if not os.path.exists(models_path + "/" + models_cfg[model_name]["repo_id"]):
        try:
            hf_hub_download(repo_id=models_cfg[model_name]["repo_id"], filename=models_cfg[model_name]["filename"], local_dir=models_path)
        except:
            ret = False

    return ret

def get_model_cfg(model_name):
    file = open(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    models_cfg = json.load(file)

    file.close()

    return models_cfg[model_name]

def download_tokenizer(models_path, model_name):
    ret = True
    file = open(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    models_cfg = json.load(file)

    file.close()

    tokenizer_path = models_path + "/" + models_cfg[model_name]["repo_id"].replace("/","-")

    if not os.path.exists(tokenizer_path):
        try:
            os.mkdir(tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(models_cfg[model_name]["repo_id"], trust_remote_code=True)
            tokenizer.save_pretrained(tokenizer_path)
        except:
            ret = False

    return ret

def get_tokenizer(models_path, model_name):
    file = open(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    models_cfg = json.load(file)

    file.close()

    tokenizer_path = models_path + "/" + models_cfg[model_name]["repo_id"].replace("/","-")

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)