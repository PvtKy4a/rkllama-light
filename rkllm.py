import ctypes
import rktypes
from model_utils import get_model, get_tokenizer

# Connect the callback function between the Python side and the C side
rkllm_cb_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(rktypes.RKLLMResult), ctypes.c_void_p, ctypes.c_int)

# Define the callback function
def rkllm_cb_imp(result, userdata, state):
    rkllm_model = ctypes.cast(userdata, ctypes.py_object).value

    if state == rktypes.LLMCallState.RKLLM_RUN_NORMAL:
        token:str = rkllm_model.decode(result.contents.token_id, skip_special_tokens=True)
        rkllm_model.response_append(token)
        print(token, flush=True, end="")
    elif state == rktypes.LLMCallState.RKLLM_RUN_FINISH:
        print("", flush=True)
        if rkllm_model.is_abort():
            rkllm_model.reset_abort()
        else:
            rkllm_model.finish()
    elif state == rktypes.LLMCallState.RKLLM_RUN_ERROR:
        print("RKLLM run error", flush=True)
    else:
        pass

class model:
    def __init__(self, lib_path, models_path, model_name):
        self.__rkllm_lib = ctypes.CDLL(lib_path)
        self.__model = get_model(model_name)
        self.__tokenizer = get_tokenizer(models_path, self.__model)
        self.__rkllm_cb = rkllm_cb_type(rkllm_cb_imp)
        self.__history = []
        self.__response = ""
        self.__abort_flag = False

        self.__rkllm_param = rktypes.RKLLMParam()
        self.__rkllm_param.model_path = bytes(models_path + "/" + self.__model["filename"], 'utf-8')
        self.__rkllm_param.max_context_len = self.__model["max_context_len"]
        self.__rkllm_param.max_new_tokens = self.__model["max_new_tokens"]
        self.__rkllm_param.skip_special_token = True
        self.__rkllm_param.top_k = self.__model["top_k"]
        self.__rkllm_param.top_p = self.__model["top_p"]
        self.__rkllm_param.temperature = self.__model["temperature"]
        self.__rkllm_param.repeat_penalty = self.__model["repeat_penalty"]
        self.__rkllm_param.frequency_penalty = self.__model["frequency_penalty"]
        self.__rkllm_param.presence_penalty = self.__model["presence_penalty"]
        self.__rkllm_param.mirostat = self.__model["mirostat"]
        self.__rkllm_param.mirostat_tau = self.__model["mirostat_tau"]
        self.__rkllm_param.mirostat_eta = self.__model["mirostat_eta"]
        self.__rkllm_param.is_async = False
        self.__rkllm_param.img_start = "".encode('utf-8')
        self.__rkllm_param.img_end = "".encode('utf-8')
        self.__rkllm_param.img_content = "".encode('utf-8')
        self.__rkllm_param.extend_param.base_domain_id = 0

        self.__rkllm_handle = rktypes.RKLLM_Handle_t()
        self.__rkllm_init = self.__rkllm_lib.rkllm_init
        self.__rkllm_init.restype = ctypes.c_int
        self.__rkllm_init.argtypes = [ctypes.POINTER(rktypes.RKLLM_Handle_t), ctypes.POINTER(rktypes.RKLLMParam), rkllm_cb_type]
        self.__rkllm_init(self.__rkllm_handle, self.__rkllm_param, self.__rkllm_cb)

        self.__rkllm_infer_params = rktypes.RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.__rkllm_infer_params), 0, ctypes.sizeof(rktypes.RKLLMInferParam))
        self.__rkllm_infer_params.mode = rktypes.RKLLMInferMode.RKLLM_INFER_GENERATE

        self.__rkllm_input = rktypes.RKLLMInput()
        self.__rkllm_input.input_mode = rktypes.RKLLMInputMode.RKLLM_INPUT_TOKEN

        self.__rkllm_run = self.__rkllm_lib.rkllm_run
        self.__rkllm_run.restype = ctypes.c_int
        self.__rkllm_run.argtypes = [rktypes.RKLLM_Handle_t, ctypes.POINTER(rktypes.RKLLMInput), ctypes.POINTER(rktypes.RKLLMInferParam), ctypes.py_object]

        self.__rkllm_abort = self.__rkllm_lib.rkllm_abort
        self.__rkllm_abort.restype = ctypes.c_int
        self.__rkllm_abort.argtypes = [rktypes.RKLLM_Handle_t]

        self.__rkllm_destroy = self.__rkllm_lib.rkllm_destroy
        self.__rkllm_destroy.argtypes = [rktypes.RKLLM_Handle_t]
        self.__rkllm_destroy.restype = ctypes.c_int

    def get_history_len(self):
        return len(self.__history)
    
    def history_append(self, data):
        self.__history.append(data)

    def history_pop(self, index=-1):
        return self.__history.pop(index)

    def history_clear(self):
        self.__history.clear()

    def response_append(self, data):
        self.__response += data

    def response_clear(self):
        self.__response = ""

    def set_abort(self):
        self.__abort_flag = True
        self.__rkllm_abort(self.__rkllm_handle)
        self.history_pop()

    def reset_abort(self):
        self.__abort_flag = False

    def is_abort(self):
        return self.__abort_flag

    def __list_to_ctype_array(self, tokens, ctype):
        # Converts a Python list to a ctype array.
        return (ctype * len(tokens))(*tokens)

    def run(self, request, regenerate=False, enable_history=True):
        if not enable_history:
            self.history_clear()

        if not self.__model["system_prompt"] == "":
            self.history_append({"role": "system", "content": self.__model["system_prompt"]})

        if regenerate:
            self.history_pop()
        else:
            self.history_append({"role": "user", "content": request})

        tokens = self.__tokenizer.apply_chat_template(self.__history, tokenize=True, add_generation_prompt=True) # The tokenizer outputs as a Python list.

        self.response_clear()

        self.__rkllm_input.input_data.token_input.input_ids = self.__list_to_ctype_array(tokens, ctypes.c_int)

        self.__rkllm_input.input_data.token_input.n_tokens = ctypes.c_ulong(len(tokens))

        self.__rkllm_run(self.__rkllm_handle, ctypes.byref(self.__rkllm_input), ctypes.byref(self.__rkllm_infer_params), ctypes.py_object(self))

    def finish(self):
        self.history_append({"role": "assistant", "content": self.__response})

    def decode(self, token_id, skip_special_tokens=False):
        return self.__tokenizer.decode(token_id, skip_special_tokens=skip_special_tokens)
    
    def destroy(self):
        self.__rkllm_destroy(self.__rkllm_handle)
