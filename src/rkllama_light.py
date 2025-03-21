import sys
import os
import json
import argparse
from model_run import model_run
from model_list import model_list

def create_models_template(path):
    cfg_template = {"Qwen2.5-Coder-1.5B-Instruct": {
        "repo_id": "c01zaut/Qwen2.5-Coder-1.5B-Instruct-rk3588-1.1.4",
        "filename": "Qwen2.5-Coder-1.5B-Instruct-rk3588-w8a8_g128-opt-0-hybrid-ratio-0.0.rkllm",
        "system_prompt": "",
        "max_context_len": 8192,
        "max_new_tokens": 2048,
        "top_k": 20,
        "top_p": 0.9,
        "temperature": 0.6,
        "repeat_penalty": 1.2,
        "frequency_penalty": 0.7,
        "presence_penalty": 0.4,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1
    }}

    with open(path, "w") as outfile:
        json.dump(cfg_template, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='rkllama-light')
    parser.add_argument('--list', action='store_true', help='Show a list of available models')
    parser.add_argument('--run', type=str, metavar='<model_name>', help='Run a model from the list of available models')
    args = parser.parse_args()

    if not os.path.exists(os.path.expanduser("~") + '/.config/rkllama_light_models.json'):
        create_models_template(os.path.expanduser("~") + '/.config/rkllama_light_models.json')

    if not os.path.exists(os.path.expanduser("~") + "/.rkllama-light"):
        os.mkdir(os.path.expanduser("~") + "/.rkllama-light")

    if args.list:
        model_list()
    elif args.run:
        model_run(args.run)
    else:
        parser.print_help()

    sys.stdout.flush()
    sys.exit(0)
