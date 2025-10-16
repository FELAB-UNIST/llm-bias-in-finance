import os
import time
from abc import ABC, abstractmethod
from utils import get_short_model_prefix # 1. utils에서 함수 가져오기

# ────────────── Configuration ──────────────
MAX_RETRIES = 3
RETRY_DELAY = 1

# ────────────── Abstract LLM Client Class ──────────────
class LLMClient(ABC):
    def __init__(self, model_id: str, temperature: float = 0.6):
        self.model_id = model_id
        self.temperature = temperature
        self.short_model_id = get_short_model_prefix(self.model_id)

    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

# ────────────── OpenAI Client ──────────────
class OpenAIClient(LLMClient):
    def __init__(self, model_id: str = "gpt-4.1", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def get_response(self, prompt: str) -> str:
        if self.model_id == "gpt-5":
            try:
                result = self.client.responses.create(
                    model="gpt-5",
                    input=prompt,
                    reasoning={"effort": "low"},
                )
                return result.output_text
            except Exception as e:
                return f"GPT-5 API call failed: {e}"
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"

# ────────────── Gemini Client ──────────────
class GeminiClient(LLMClient):
    def __init__(self, model_id: str = "gemini-2.5-flash", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)

    def get_response(self, prompt: str) -> str:
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=0)
                    ),
                )
                text = resp.text or ""
                if text.strip():
                    return text
                last_error = f"Empty response on attempt {attempt}"
            except Exception as e:
                last_error = f"Error on attempt {attempt}: {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts; last error: {last_error}"

# ────────────── Together Client ──────────────
class TogetherClient(LLMClient):
    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-V3", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from together import Together
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        self.client = Together(api_key=api_key)

    def get_response(self, prompt: str) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"

# ────────────── Anthropic Client ──────────────
class AnthropicClient(LLMClient):
    def __init__(self, model_id: str = "claude-sonnet-4-20250514", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)

    def get_response(self, prompt: str) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                message = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                ).content[0].text

                if message:
                    return message
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"

# ────────────── XAI Client ──────────────
class XAIClient(LLMClient):
    def __init__(self, model_id: str = "grok-4", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from openai import OpenAI
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable not set")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
        )

    def get_response(self, prompt: str) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"