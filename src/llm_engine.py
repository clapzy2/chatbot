"""
LLM Engine v3.0 — три режима:
  ollama    → любая модель через Ollama HTTP API (локально)
  llama_cpp → Qwen3 через .gguf файл (локально)
  api       → Groq / OpenAI-совместимые API (облачно, параллельные запросы)

Переключается через config.LLM_MODE
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


# ══════════════════════════════════════════════════════════════
#  OLLAMA (локально)
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
#  LLAMA-CPP (локально)
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
                "Скачайте модель командой:\n"
                "  pip install huggingface-hub\n"
                "  huggingface-cli download Qwen/Qwen3-8B-GGUF "
                "qwen3-8b-q4_k_m.gguf --local-dir models/\n"
                "  mv models/qwen3-8b-q4_k_m.gguf models/model.gguf"
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
#  API (Groq / OpenAI-совместимые)
# ══════════════════════════════════════════════════════════════
class _ApiBackend:
    """
    Работает с любым OpenAI-совместимым API:
    - Groq (бесплатно, быстро)
    - Together AI
    - OpenRouter
    - OpenAI
    """

    def __init__(self):
        import requests
        self._requests = requests
        self._api_url = getattr(config, "API_URL", "https://api.groq.com/openai/v1/chat/completions")
        self._api_key = getattr(config, "API_KEY", "")
        self._api_model = getattr(config, "API_MODEL", "qwen/qwen3-32b")

        # Обходим прокси для API
        os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.groq.com,openrouter.ai"
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",api.groq.com,openrouter.ai"

        if not self._api_key:
            self._api_key = os.environ.get("GROQ_API_KEY", "")

        if not self._api_key:
            print("⚠️ API_KEY не задан! Установите в config.py или переменной GROQ_API_KEY")
        else:
            print(f"🔑 API ключ: {self._api_key[:10]}...")

    def generate(self, prompt: str, temperature: float,
                 max_tokens: int, stream: bool = False):
        print(f"🔍 API вызов: stream={stream}, model={self._api_model}, prompt_len={len(prompt)}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Разделяем на system и user сообщения
        system_prompt = getattr(config, "SYSTEM_PROMPT", "")
        if system_prompt and prompt.startswith(system_prompt):
            user_content = prompt[len(system_prompt):].strip()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "system", "content": "Ты — полезный ассистент. Отвечай на русском языке."},
                {"role": "user", "content": prompt},
            ]

        # Qwen3: отключаем thinking mode (иначе content=null, 50 сек задержка)
        if "qwen3" in self._api_model.lower():
            messages[-1]["content"] = "/no_think\n" + messages[-1]["content"]

        payload = {
            "model": self._api_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if stream:
            r = self._requests.post(
                self._api_url, headers=headers, json=payload,
                stream=True, timeout=120
            )
            if r.status_code != 200:
                error_text = r.text[:500] if hasattr(r, 'text') else str(r.status_code)
                raise RuntimeError(f"API ошибка {r.status_code}: {error_text}")
            for line in r.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content") or ""
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        else:
            r = self._requests.post(
                self._api_url, headers=headers, json=payload, timeout=120
            )
            if r.status_code != 200:
                error_text = r.text[:500] if hasattr(r, 'text') else str(r.status_code)
                raise RuntimeError(f"API ошибка {r.status_code}: {error_text}")
            data = r.json()
            raw = data["choices"][0]["message"].get("content") or ""
            text = raw.strip()
            if text:
                yield text

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
            }
            r = self._requests.get(
                self._api_url.replace("/chat/completions", "/models"),
                headers=headers, timeout=5
            )
            return r.status_code == 200
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════
#  ЕДИНЫЙ ИНТЕРФЕЙС
# ══════════════════════════════════════════════════════════════
class LLMEngine:
    """
    Единый интерфейс для всех движков.
    config.LLM_MODE = "ollama" | "llama_cpp" | "api"
    """

    def __init__(self):
        mode = config.LLM_MODE
        if mode == "ollama":
            self._backend = _OllamaBackend()
            print(f"🤖 LLM: Ollama → {config.OLLAMA_MODEL}")
        elif mode == "api":
            self._backend = _ApiBackend()
            model = getattr(config, "API_MODEL", "qwen/qwen3-32b")
            print(f"🤖 LLM: API → {model}")
        else:
            self._backend = _LlamaCppBackend()
            print(f"🤖 LLM: llama-cpp → {os.path.basename(config.LLM_MODEL_PATH)}")

    def load(self):
        if hasattr(self._backend, "load"):
            self._backend.load()

    def call(self, prompt: str, temperature: float = None,
             max_tokens: int = None) -> str:
        temp   = temperature if temperature is not None else config.LLM_TEMPERATURE
        tokens = max_tokens  if max_tokens  is not None else config.LLM_MAX_TOKENS
        result = ""
        for token in self._backend.generate(prompt, temp, tokens, stream=False):
            result += token
        return result.strip()

    def stream(self, prompt: str, temperature: float = None,
               max_tokens: int = None):
        temp   = temperature if temperature is not None else config.LLM_TEMPERATURE
        tokens = max_tokens  if max_tokens  is not None else config.LLM_MAX_TOKENS
        yield from self._backend.generate(prompt, temp, tokens, stream=True)

    def generate_with_context(self, template: str, topic: str,
                               context: str, stream: bool = False):
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
