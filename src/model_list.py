import sys
from model_utils import get_available_models

def model_list():
    models = get_available_models()

    if len(models) < 1:
        print("There are no models available", flush=True)
    else:
        print("Available models:", flush=True)
        for model in models:
            print(model)
        sys.stdout.flush()
