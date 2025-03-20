import sys
import resource
import argparse
import signal
import readline
from rkllm import RKLLMModel
from models_utils import *

# Set resource limit
resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

def print_help():
    print("/regenerate      Regenerate last response", flush=True)
    print("/clear           Clear chat history", flush=True)
    print("/set history     Enable chat history (enabled by default)", flush=True)
    print("/unset history   Disable chat history", flush=True)
    print("/save <file>     Save chat history to specified file", flush=True)
    print("/load <file>     Load chat history from specified file", flush=True)
    print("/bye             Exit", flush=True)
    print(flush=True)
    print("Ctrl + l         Clear the screen", flush=True)
    print("Ctrl + c         Stop the model from responding", flush=True)
    print("Ctrl + d         Exit (/bye)", flush=True)
    print(flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-list', action='store_true', help='List of avilable models')
    parser.add_argument('-run', type=str, help='Model from avilable models list to run')
    args = parser.parse_args()

    LIB_PATH = "./lib/librkllmrt.so"

    MODELS_PATH = "./models"

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    if args.list:
        models = get_available_models()

        if len(models) < 1:
            print("There are no models available", flush=True)
        else:
            print("Available models:", flush=True)
            for model in models:
                print(model)

        sys.stdout.flush()
        sys.exit(0)

    if args.run:
        model_name = args.run
        regenerate = False
        enable_history = True

        if not download_tokenizer(MODELS_PATH, model_name):
            print("\nIncorrect model repo_id", flush=True)
            sys.exit(0)

        if not download_model(MODELS_PATH, model_name):
            print("\nIncorrect model filename.", flush=True)
            sys.exit(0)

        rkllm_model = RKLLMModel(LIB_PATH, MODELS_PATH, model_name)

        def abort_handler(sig, frame):
            rkllm_model.set_abort()
            rkllm_model.rkllm_abort(rkllm_model.handle)

        default_sigint_handler = signal.getsignal(signal.SIGINT)

        readline.set_auto_history(True)

        sys.stdout.flush()

        print("Enter \"/?\" or \"/help\" for help", flush=True)

        while True:
            user_input = ""

            try:
                user_input = input("\nYou: ")
            except EOFError:
                break
            except KeyboardInterrupt:
                if readline.get_line_buffer() == "":
                    print("\nUse Ctrl + d or /bye to exit.", flush=True, end="")
                continue

            if user_input == "/bye":
                break

            if user_input == "/?" or user_input == "/help":
                print_help()
                continue

            if user_input == "/regenerate":
                if rkllm_model.get_history_len() < 2:
                    print("There have been no requests yet.", flush=True, end="")
                    continue
                regenerate = True

            if user_input == "/set history":
                enable_history = True
                continue

            if user_input == "/unset history":
                enable_history = False
                continue

            if user_input == "/clear":
                rkllm_model.clear_history()
                continue

            signal.signal(signal.SIGINT, abort_handler)

            print("\nAI: ", flush=True, end="")

            rkllm_model.RKLLM_run(user_input, regenerate=regenerate, enable_history=enable_history)

            regenerate = False

            signal.signal(signal.SIGINT, default_sigint_handler)

        print("\nExiting...", flush=True)

        rkllm_model.rkllm_destroy(rkllm_model.handle)

        sys.exit(0)