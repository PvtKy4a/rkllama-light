import os
import sys
import signal
import readline
import rkllm
from model_utils import download_tokenizer, download_model

def print_help():
    print("/regenerate      Regenerate last response")
    print("/clear           Clear chat history")
    print("/set history     Enable chat history (enabled by default)")
    print("/unset history   Disable chat history")
    print("/set thinking    Enable model thinking")
    print("/unset thinking  Disable model thinking (disabled by default)")
    print("/save <file>     Save chat history to specified file")
    print("/load <file>     Load chat history from specified file")
    print("/bye             Exit")
    print()
    print("Ctrl + l         Clear the screen")
    print("Ctrl + c         Stop the model from responding")
    print("Ctrl + d         Exit (/bye)")
    sys.stdout.flush()

def model_run(model_name):
    LIB_PATH = "/usr/local/lib/librkllmrt.so"
    MODELS_PATH = os.path.expanduser("~") + "/.rkllama-light/models"
    TOKENIZERS_PATH = os.path.expanduser("~") + "/.rkllama-light/tokenizers"

    regenerate = False
    enable_history = True
    enable_thinking = False

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    if not os.path.exists(TOKENIZERS_PATH):
        os.mkdir(TOKENIZERS_PATH)

    if not download_tokenizer(TOKENIZERS_PATH, model_name):
        return

    if not download_model(MODELS_PATH, model_name):
        return

    rkllm_model = rkllm.model(LIB_PATH, TOKENIZERS_PATH, MODELS_PATH, model_name)

    def abort_handler(sig, frame):
        rkllm_model.set_abort()

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
        elif user_input == "/?" or user_input == "/help":
            print_help()
            continue
        elif user_input == "/regenerate":
            regenerate = True
        elif user_input == "/set history":
            enable_history = True
            continue
        elif user_input == "/unset history":
            enable_history = False
            continue
        elif user_input == "/set thinking":
            enable_thinking = True
            continue
        elif user_input == "/unset thinking":
            enable_thinking = False
            continue
        elif user_input == "/save":
            print("Not implemented yet.", flush=True, end="")
            continue
        elif user_input == "/load":
            print("Not implemented yet.", flush=True, end="")
            continue
        elif user_input == "/clear":
            rkllm_model.history_clear()
            continue

        signal.signal(signal.SIGINT, abort_handler)

        print("\nAI: ", flush=True, end="")

        rkllm_model.run(user_input, regenerate=regenerate, enable_history=enable_history, enable_thinking=enable_thinking)

        regenerate = False

        signal.signal(signal.SIGINT, default_sigint_handler)

    print(flush=True)

    rkllm_model.destroy()
