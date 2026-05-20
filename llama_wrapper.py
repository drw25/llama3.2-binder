from llama_cpp import Llama
from gguf.gguf_reader import GGUFReader
import numpy as np

LLM_ARGS = {"n_threads":1,"n_threads_batch":1,"logits_all":True,"verbose":False,"flash_attn":True}

class LlamaWrapperEmptyLLM:
    def __getattr__(self,name):
        print("ERROR: LLM not loaded. To load the LLM, use the wrapper as a context manager i.e. as the subject of a 'with' statement.")
        return None

class LlamaWrapper:
    def __init__(self,model_path,tokenizer_only=False,**kwargs):
        self.model_path = model_path
        self._llm = LlamaWrapperEmptyLLM()
        self._kwargs = kwargs
        self.tokenizer_only = tokenizer_only
    def __repr__(self):
        return f"LLMWrapper(model_path={self.model_path},tokenizer_only={self.tokenizer_only})"
    def __enter__(self):
        if self.tokenizer_only:
            self._llm = Llama(model_path=self.model_path,vocab_only=False,**LLM_ARGS,**self._kwargs)
        else:
            self._llm = Llama(model_path=self.model_path,**LLM_ARGS,**self._kwargs)
        return self
    def __exit__(self,type,value,traceback):
        del(self._llm)
        self._llm = LlamaWrapperEmptyLLM()
    def __getattr__(self,name):
        if not name.startswith("__"):
            return getattr(self._llm,name)
        else:
            raise Exception("Private attribute does not exist.")
    def get_tensors(self):
        return GGUFReader(self.model_path,"r").tensors
    def get_tensor(self,tensor_name):
        tensor = [t for t in self.get_tensors() if t.name==tensor_name][0]
        return (np.array(tensor.data,copy=True),tensor.tensor_type)