import ctypes
import rktypes
from model_utils import get_model, get_tokenizer

# Connect the callback function between the Python side and the C side
rkllm_cb_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(rktypes.RKLLMResult), ctypes.c_void_p, ctypes.c_int)

# Define the callback function
def rkllm_cb_imp(result, userdata, state):
    rkllm_model = ctypes.cast(userdata, ctypes.py_object).value

    if state == rktypes.LLMCallState.RKLLM_RUN_NORMAL:
        token = rkllm_model.tokenizer.decode(result.contents.token_id, skip_special_tokens=True)
        rkllm_model.response += token
        print(token, flush=True, end="")
    elif state == rktypes.LLMCallState.RKLLM_RUN_FINISH:
        print("", flush=True)
        if rkllm_model.abort:
            rkllm_model.abort = False
        else:
            rkllm_model.history.append({"role": "assistant", "content": rkllm_model.response})
    elif state == rktypes.LLMCallState.RKLLM_RUN_ERROR:
        print("RKLLM run error", flush=True)
    else:
        pass

class RKLLM:
    def __init__(self, lib_path, models_path, model_name):
        self.rkllm_lib = ctypes.CDLL(lib_path)
        self.model = get_model(model_name)
        self.tokenizer = get_tokenizer(models_path, self.model)
        self.rkllm_cb = rkllm_cb_type(rkllm_cb_imp)
        self.history = []
        self.response = ""
        self.abort = False

        self.rkllm_param = rktypes.RKLLMParam()
        self.rkllm_param.model_path = bytes(models_path + "/" + self.model["filename"], 'utf-8')
        self.rkllm_param.max_context_len = self.model["max_context_len"]
        self.rkllm_param.max_new_tokens = self.model["max_new_tokens"]
        self.rkllm_param.skip_special_token = True
        self.rkllm_param.top_k = self.model["top_k"]
        self.rkllm_param.top_p = self.model["top_p"]
        self.rkllm_param.temperature = self.model["temperature"]
        self.rkllm_param.repeat_penalty = self.model["repeat_penalty"]
        self.rkllm_param.frequency_penalty = self.model["frequency_penalty"]
        self.rkllm_param.presence_penalty = self.model["presence_penalty"]
        self.rkllm_param.mirostat = self.model["mirostat"]
        self.rkllm_param.mirostat_tau = self.model["mirostat_tau"]
        self.rkllm_param.mirostat_eta = self.model["mirostat_eta"]
        self.rkllm_param.is_async = False
        self.rkllm_param.img_start = "".encode('utf-8')
        self.rkllm_param.img_end = "".encode('utf-8')
        self.rkllm_param.img_content = "".encode('utf-8')
        self.rkllm_param.extend_param.base_domain_id = 0

        self.rkllm_handle = rktypes.RKLLM_Handle_t()
        self.rkllm_init = self.rkllm_lib.rkllm_init
        self.rkllm_init.restype = ctypes.c_int
        self.rkllm_init.argtypes = [ctypes.POINTER(rktypes.RKLLM_Handle_t), ctypes.POINTER(rktypes.RKLLMParam), rkllm_cb_type]
        self.rkllm_init(self.rkllm_handle, self.rkllm_param, self.rkllm_cb)

        self.rkllm_infer_params = rktypes.RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(rktypes.RKLLMInferParam))
        self.rkllm_infer_params.mode = rktypes.RKLLMInferMode.RKLLM_INFER_GENERATE

        self.rkllm_input = rktypes.RKLLMInput()
        self.rkllm_input.input_mode = rktypes.RKLLMInputMode.RKLLM_INPUT_TOKEN

        self.rkllm_run = self.rkllm_lib.rkllm_run
        self.rkllm_run.restype = ctypes.c_int
        self.rkllm_run.argtypes = [rktypes.RKLLM_Handle_t, ctypes.POINTER(rktypes.RKLLMInput), ctypes.POINTER(rktypes.RKLLMInferParam), ctypes.py_object]

        self.rkllm_abort = self.rkllm_lib.rkllm_abort
        self.rkllm_abort.restype = ctypes.c_int
        self.rkllm_abort.argtypes = [rktypes.RKLLM_Handle_t]

        self.rkllm_destroy = self.rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [rktypes.RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

    def list_to_ctype_array(self, tokens, ctype):
        # Converts a Python list to a ctype array.
        return (ctype * len(tokens))(*tokens)
    
    def get_history_len(self):
        return len(self.history)
    
    def clear_history(self):
        self.history.clear()
    
    def set_abort(self):
        self.abort = True

    # Retrieve the output from the RKLLM model and print it in a streaming manner
    def RKLLM_run(self, request, regenerate=False, enable_history=True):
        if not enable_history:
            self.clear_history()

        if not self.model["system_prompt"] == "":
            self.history.append({"role": "system", "content": self.model["system_prompt"]})

        if regenerate:
            self.history.pop()
        else:
            self.history.append({"role": "user", "content": request})

        tokens = self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True) # The tokenizer outputs as a Python list.

        self.response = ""

        self.rkllm_input.input_data.token_input.input_ids = self.list_to_ctype_array(tokens, ctypes.c_int)

        self.rkllm_input.input_data.token_input.n_tokens = ctypes.c_ulong(len(tokens))

        self.rkllm_run(self.rkllm_handle, ctypes.byref(self.rkllm_input), ctypes.byref(self.rkllm_infer_params), ctypes.py_object(self))
