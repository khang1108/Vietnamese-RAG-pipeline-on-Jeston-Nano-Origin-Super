from core.llm import BaseLLM
from core.logger import logger
from llama_cpp import Llama
from typing import List, Any

class QwenLLM(BaseLLM):
    def __init__(self, model_path: str, 
                n_ctx: int = 2048, 
                n_gpu_layers: int = -1, 
                n_batch: int = 512,
                chat_format: str = 'llama-2',
                verbose: bool = False
                ):
        '''
        To initialize Qwen2.5-3B-Instruct-GGUF.

        Parameters:
            model_path (str): Path to GGUF file of Qwen2.5-3B-Instruct-GGUF model
            n_ctx (int): Number of context tokens (Default = 2048, 0 = only from model)
            n_gpu_layers (int): Number of layers to offload to GPU. (-1 = all layers offloaded)
            n_batch (int): Prompt processing maximum batch size
            chat_format (str): String specifying the chat format to use when calling 'create_chat_completion'
            verbose (bool): Print verbose output to stderr
        '''
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False
            )
            logger.success(f"Successfully loaded QWen model")
        except Exception as e:
            logger.error(f"[Local Model] Cannot load model: {e}")

    def generate(self, 
                prompt: str, 
                temperature: float, 
                top_p: float, 
                top_k: float,
                max_tokens: int, 
                stop: List[str],
                response_format: Any,
                stream: bool = False,
                ):
        pass