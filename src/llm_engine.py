"""
LLM Engine — поддерживает два режима:
  ollama    → Mistral/любая модель через Ollama HTTP API
  llama_cpp → Qwen2.5 через llama-cpp-python (.gguf файл)
Переключается через config.LLM_MODE
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ══════════════════════════════════════════════════════════════
#  OLLAMA
# ══════════════════════════════════════════════════════════════
class _OllamaBackend:

    def __init__(self):
        import requests
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        self._requests = requests

    def generate(self, prompt: str, temperature: float,
                 max_tokens: int, stream: bool = False):
        payload = {
            "model":  config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature":    temperature,
                "top_p":          config.LLM_TOP_P,
                "num_predict":    max_tokens,
                "repeat_penalty": config.LLM_REPEAT_PENALTY,
                "stop":           ["<|im_end|>", "</s>", "ВОПРОС:", "КОНТЕКСТ:"],
            },
        }
        if stream:
            import json
            r = self._requests.post(config.OLLAMA_URL, json=payload,
                                    stream=True, timeout=180)
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
        else:
            r = self._requests.post(config.OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            yield r.json().get("response", "").strip()

    def is_available(self) -> bool:
        try:
            r = self._requests.get("http://localhost:11434/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════
#  LLAMA-CPP (Qwen2.5)
# ══════════════════════════════════════════════════════════════
class _LlamaCppBackend:

    def __init__(self):
        self._model = None

    def load(self):
        if self._model is not None:
            return
        if not os.path.exists(config.LLM_MODEL_PATH):
            raise FileNotFoundError(
                f"Модель не найдена: {config.LLM_MODEL_PATH}\n"
                "Скачайте модель через вкладку «Настройки» или командой:\n"
                "  pip install huggingface-hub\n"
                "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF "
                "qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models/\n"
                "  mv models/qwen2.5-7b-instruct-q4_k_m.gguf models/model.gguf"
            )
        from llama_cpp import Llama
        print(f"🔄 Загружаем модель: {os.path.basename(config.LLM_MODEL_PATH)}")
        kwargs = dict(
            model_path=config.LLM_MODEL_PATH,
            n_ctx=config.LLM_CONTEXT_SIZE,
            n_gpu_layers=config.LLM_GPU_LAYERS,
            n_batch=config.LLM_N_BATCH,
            verbose=False,
        )
        if config.LLM_N_THREADS:
            kwargs["n_threads"] = config.LLM_N_THREADS
        self._model = Llama(**kwargs)
        print("✅ Модель загружена")

    def generate(self, prompt: str, temperature: float,
                 max_tokens: int, stream: bool = False):
        self.load()
        kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=config.LLM_TOP_P,
            repeat_penalty=config.LLM_REPEAT_PENALTY,
            stop=["<|im_end|>", "<|end|>", "</s>", "<|eot_id|>"],
            echo=False,
            stream=stream,
        )
        if stream:
            for chunk in self._model(prompt, **kwargs):
                token = chunk["choices"][0]["text"]
                if token:
                    yield token
        else:
            result = self._model(prompt, **kwargs)
            yield result["choices"][0]["text"].strip()

    def is_available(self) -> bool:
        return os.path.exists(config.LLM_MODEL_PATH)


# ══════════════════════════════════════════════════════════════
#  ЕДИНЫЙ ИНТЕРФЕЙС
# ══════════════════════════════════════════════════════════════
class LLMEngine:
    """
    Единый интерфейс для обоих движков.
    Переключается через config.LLM_MODE = "ollama" | "llama_cpp"
    """

    def __init__(self):
        if config.LLM_MODE == "ollama":
            self._backend = _OllamaBackend()
            print(f"🤖 LLM: Ollama → {config.OLLAMA_MODEL}")
        else:
            self._backend = _LlamaCppBackend()
            print(f"🤖 LLM: llama-cpp → {os.path.basename(config.LLM_MODEL_PATH)}")

    def load(self):
        """Явная загрузка (нужна для llama-cpp)"""
        if hasattr(self._backend, "load"):
            self._backend.load()

    def call(self, prompt: str, temperature: float = None,
             max_tokens: int = None) -> str:
        """Синхронный вызов → строка"""
        temp   = temperature if temperature is not None else config.LLM_TEMPERATURE
        tokens = max_tokens  if max_tokens  is not None else config.LLM_MAX_TOKENS
        result = ""
        for token in self._backend.generate(prompt, temp, tokens, stream=False):
            result += token
        return result.strip()

    def stream(self, prompt: str, temperature: float = None,
               max_tokens: int = None):
        """Генератор токенов для стриминга в Gradio"""
        temp   = temperature if temperature is not None else config.LLM_TEMPERATURE
        tokens = max_tokens  if max_tokens  is not None else config.LLM_MAX_TOKENS
        yield from self._backend.generate(prompt, temp, tokens, stream=True)

    def generate_with_context(self, template: str, topic: str,
                               context: str, stream: bool = False):
        """Подставляет переменные в шаблон и вызывает модель"""
        prompt = template.format(
            system=config.SYSTEM_PROMPT,
            topic=topic,
            context=context,
        )
        if stream:
            return self.stream(prompt)
        return self.call(prompt)

    def is_available(self) -> bool:
        return self._backend.is_available()
