"""
llm_engine.py — подключение к LLM (нейросети для генерации текста).
Поддерживает два режима: API (облако) и Ollama (локально).
"""
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Бэкенд для локальной работы через Ollama (без интернета)
class _OllamaBackend:

    def __init__(self):
        import requests
        # Отключаем прокси для локальных запросов
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        os.environ["no_proxy"] = "localhost,127.0.0.1"
        self._requests = requests

    def generate(self, prompt, temperature, max_tokens, stream=False):
        """Отправляет запрос к локальному серверу Ollama."""
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
            # Потоковый режим: читаем ответ по кусочкам (для стриминга в чате)
            r = self._requests.post(config.OLLAMA_URL, json=payload, stream=True, timeout=180)
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break
        else:
            # Обычный режим: ждём полный ответ
            r = self._requests.post(config.OLLAMA_URL, json=payload, timeout=180)
            r.raise_for_status()
            yield r.json().get("response", "").strip()

    def is_available(self):
        """Проверяет, запущен ли сервер Ollama."""
        try:
            r = self._requests.get("http://localhost:11434/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


# Бэкенд для облачной работы через API (OpenRouter)
class _ApiBackend:

    def __init__(self):
        import requests
        self._requests = requests
        self._api_url = getattr(config, "API_URL", "https://openrouter.ai/api/v1/chat/completions")
        self._api_key = getattr(config, "API_KEY", "")
        self._api_model = getattr(config, "API_MODEL", "qwen/qwen3-32b")

        # Отключаем прокси для API-серверов
        os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",openrouter.ai"
        os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",openrouter.ai"

        # Проверяем наличие ключа
        if not self._api_key:
            self._api_key = os.environ.get("API_KEY", "")
        if not self._api_key:
            print("⚠️ API_KEY не задан! Установите в config.py")

    def generate(self, prompt, temperature, max_tokens, stream=False):
        """Отправляет запрос к OpenRouter API."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Разделяем промпт на системное сообщение и вопрос пользователя
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

        # Для Qwen3: отключаем режим размышлений (ускоряет ответ)
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
            # Потоковый режим: ответ приходит по кусочкам (токенам)
            r = self._requests.post(self._api_url, headers=headers, json=payload, stream=True, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"API ошибка {r.status_code}: {r.text[:500]}")
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
                        content = chunk["choices"][0].get("delta", {}).get("content") or ""
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        else:
            # Обычный режим: ждём полный ответ
            r = self._requests.post(self._api_url, headers=headers, json=payload, timeout=120)
            if r.status_code != 200:
                raise RuntimeError(f"API ошибка {r.status_code}: {r.text[:500]}")
            data = r.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if text:
                yield text

    def is_available(self):
        """Проверяет, доступен ли API."""
        if not self._api_key:
            return False
        try:
            headers = {"Authorization": f"Bearer {self._api_key}"}
            r = self._requests.get(
                self._api_url.replace("/chat/completions", "/models"),
                headers=headers, timeout=5
            )
            return r.status_code == 200
        except Exception:
            return False


# Единый интерфейс для работы с LLM
class LLMEngine:
    """
    Выбирает нужный бэкенд (API или Ollama) и предоставляет
    единые методы call() и stream() для всего приложения.
    """

    def __init__(self):
        mode = config.LLM_MODE
        if mode == "ollama":
            self._backend = _OllamaBackend()
        else:
            self._backend = _ApiBackend()

    def call(self, prompt, temperature=None, max_tokens=None):
        """Отправить запрос и получить полный ответ (строкой)."""
        temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        result = ""
        for token in self._backend.generate(prompt, temp, tokens, stream=False):
            result += token
        return result.strip()

    def stream(self, prompt, temperature=None, max_tokens=None):
        """Отправить запрос и получать ответ по токенам (для стриминга в чате)."""
        temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        yield from self._backend.generate(prompt, temp, tokens, stream=True)

    def generate_with_context(self, template, topic, context, stream=False):
        """Подставить контекст в шаблон и сгенерировать ответ."""
        prompt = template.format(system=config.SYSTEM_PROMPT, topic=topic, context=context)
        if stream:
            return self.stream(prompt)
        return self.call(prompt)

    def is_available(self):
        """Проверить, доступен ли выбранный бэкенд."""
        return self._backend.is_available()