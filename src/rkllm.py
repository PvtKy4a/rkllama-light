import ctypes
import rktypes
from model_utils import get_model_cfg, get_tokenizer

def rkllm_cb(result, userdata, state):
    rkllm_model = ctypes.cast(userdata, ctypes.py_object).value

    if state == rktypes.RKLLMCallState.RKLLM_RUN_NORMAL:
        token = rkllm_model.decode(result.contents.token_id, skip_special_tokens=True)
        rkllm_model.response_append(token)
        print(token, flush=True, end="")
    elif state == rktypes.RKLLMCallState.RKLLM_RUN_FINISH:
        if rkllm_model.is_abort():
            rkllm_model.reset_abort()
        else:
            print("", flush=True)
            rkllm_model.finish()
    elif state == rktypes.RKLLMCallState.RKLLM_RUN_ERROR:
        print("RKLLM run error", flush=True)
    else:
        pass

class model:
    def __init__(self, lib_path, tokenizers_path, models_path, model_name):
        self.__cfg = get_model_cfg(model_name)
        self.__tokenizer = get_tokenizer(tokenizers_path, model_name)
        self.__history = []
        self.__response = ""
        self.__abort = False

        self.__rkllm_lib = ctypes.CDLL(lib_path)

        self.__rkllm_cb = rktypes.LLMResultCallback(rkllm_cb)

        self.__rkllm_param = rktypes.RKLLMParam()
        self.__rkllm_param.model_path = bytes(models_path + "/" + self.__cfg["filename"], 'utf-8')
        self.__rkllm_param.max_context_len = self.__cfg["max_context_len"]
        self.__rkllm_param.max_new_tokens = self.__cfg["max_new_tokens"]
        self.__rkllm_param.skip_special_token = True
        self.__rkllm_param.top_k = self.__cfg["top_k"]
        self.__rkllm_param.n_keep = -1
        self.__rkllm_param.top_p = self.__cfg["top_p"]
        self.__rkllm_param.temperature = self.__cfg["temperature"]
        self.__rkllm_param.repeat_penalty = self.__cfg["repeat_penalty"]
        self.__rkllm_param.frequency_penalty = self.__cfg["frequency_penalty"]
        self.__rkllm_param.presence_penalty = self.__cfg["presence_penalty"]
        self.__rkllm_param.mirostat = self.__cfg["mirostat"]
        self.__rkllm_param.mirostat_tau = self.__cfg["mirostat_tau"]
        self.__rkllm_param.mirostat_eta = self.__cfg["mirostat_eta"]
        self.__rkllm_param.is_async = False
        self.__rkllm_param.img_start = "".encode('utf-8')
        self.__rkllm_param.img_end = "".encode('utf-8')
        self.__rkllm_param.img_content = "".encode('utf-8')
        self.__rkllm_param.extend_param.base_domain_id = 0
        self.__rkllm_param.extend_param.enabled_cpus_num = 4
        self.__rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)

        self.__rkllm_handle = rktypes.RKLLM_Handle_t()
        self.__rkllm_init = self.__rkllm_lib.rkllm_init
        self.__rkllm_init.restype = ctypes.c_int
        self.__rkllm_init.argtypes = [ctypes.POINTER(rktypes.RKLLM_Handle_t), ctypes.POINTER(rktypes.RKLLMParam), rktypes.LLMResultCallback]
        self.__rkllm_init(self.__rkllm_handle, self.__rkllm_param, self.__rkllm_cb)

        self.__rkllm_infer_params = rktypes.RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.__rkllm_infer_params), 0, ctypes.sizeof(rktypes.RKLLMInferParam))
        self.__rkllm_infer_params.mode = rktypes.RKLLMInferMode.RKLLM_INFER_GENERATE
        self.__rkllm_infer_params.keep_history = 0

        self.__rkllm_input = rktypes.RKLLMInput()
        self.__rkllm_input.input_mode = rktypes.RKLLMInputMode.RKLLM_INPUT_TOKEN

        self.__rkllm_run = self.__rkllm_lib.rkllm_run
        self.__rkllm_run.restype = ctypes.c_int
        self.__rkllm_run.argtypes = [rktypes.RKLLM_Handle_t, ctypes.POINTER(rktypes.RKLLMInput), ctypes.POINTER(rktypes.RKLLMInferParam), ctypes.py_object]

        self.__rkllm_abort = self.__rkllm_lib.rkllm_abort
        self.__rkllm_abort.restype = ctypes.c_int
        self.__rkllm_abort.argtypes = [rktypes.RKLLM_Handle_t]

        self.__rkllm_destroy = self.__rkllm_lib.rkllm_destroy
        self.__rkllm_destroy.restype = ctypes.c_int
        self.__rkllm_destroy.argtypes = [rktypes.RKLLM_Handle_t]

    def history_append(self, data):
        self.__history.append(data)

    def history_clear(self):
        self.__history.clear()

    def response_append(self, data):
        self.__response += data

    def response_clear(self):
        self.__response = ""

    def set_abort(self):
        self.__abort = True
        self.__rkllm_abort(self.__rkllm_handle)
        self.__history.pop()

    def reset_abort(self):
        self.__abort = False

    def is_abort(self):
        return self.__abort

    def __list_to_ctype_array(self, tokens, ctype):
        return (ctype * len(tokens))(*tokens)

    def run(self, request, regenerate=False, enable_history=True, enable_thinking=False):
        if not enable_history:
            self.__history.clear()

        if not self.__cfg["system_prompt"] == "":
            self.__history.append({"role": "system", "content": self.__cfg["system_prompt"]})

        if regenerate:
            if len(self.__history) < 2:
                print("There have been no requests yet.", flush=True, end="")
                return
            else:
                self.__history.pop()
        else:
            self.__history.append({"role": "user", "content": request})

        tokens = self.__tokenizer.apply_chat_template(self.__history, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking)

        self.__response = ""

        self.__rkllm_input.input_data.token_input.input_ids = self.__list_to_ctype_array(tokens, ctypes.c_int)

        self.__rkllm_input.input_data.token_input.n_tokens = ctypes.c_size_t(len(tokens))

        self.__rkllm_run(self.__rkllm_handle, ctypes.byref(self.__rkllm_input), ctypes.byref(self.__rkllm_infer_params), ctypes.py_object(self))

    def finish(self):
        self.__history.append({"role": "assistant", "content": self.__response})

    def decode(self, token_id, skip_special_tokens=False):
        return self.__tokenizer.decode(token_id, skip_special_tokens=skip_special_tokens)

    def destroy(self):
        self.__rkllm_destroy(self.__rkllm_handle)
