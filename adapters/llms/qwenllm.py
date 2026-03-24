from typing import Any

from llama_cpp import Llama

from core.llm import BaseLLM
from core.logger import logger


class QwenLLM(BaseLLM):
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        chat_format: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize a local Qwen GGUF model through llama-cpp-python.

        Args:
            model_path: Path to the GGUF file.
            n_ctx: Runtime context window.
            n_gpu_layers: Number of layers offloaded to GPU. Use -1 to offload all.
            n_batch: Maximum prompt processing batch size.
            chat_format: Optional chat template override. Leave as None to let GGUF metadata decide.
            verbose: Whether llama.cpp should print verbose logs.
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.chat_format = chat_format
        self.verbose = verbose
        self.llm: Llama | None = None

        llama_kwargs = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "verbose": self.verbose,
        }
        if self.chat_format is not None:
            llama_kwargs["chat_format"] = self.chat_format

        try:
            self.llm = Llama(**llama_kwargs)
            logger.success(
                f"[Local Model] Successfully loaded Qwen model from {self.model_path}"
            )
        except Exception as exc:
            logger.error(f"[Local Model] Cannot load model: {exc}")
            raise

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        response_format: Any | None = None,
        stream: bool = False,
    ) -> Any:
        """
        Generate a chat completion from role-based messages.

        Args:
            messages: Chat messages in OpenAI-style format.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Limits token candidates at each step.
            max_tokens: Maximum number of generated tokens.
            stop: Optional stop strings.
            response_format: Optional response schema for structured outputs.
            stream: Whether to stream tokens.
        """
        if self.llm is None:
            raise RuntimeError("Qwen model is not initialized.")

        stop_sequences = stop or []

        request_kwargs = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "stop": stop_sequences,
            "stream": stream,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        return self.llm.create_chat_completion(**request_kwargs)
