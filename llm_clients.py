import os
import time
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from utils import get_short_model_prefix

# Load environment variables from .env file
load_dotenv()

# ────────────── Configuration ──────────────
MAX_RETRIES = 3
RETRY_DELAY = 1

# ────────────── Pricing (USD per 1M tokens) ──────────────
PRICING = {
    # OpenAI
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-5": {"input": 1.25, "output": 10.00},

    # Anthropic
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},

    # Google
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},

    # Together AI / DeepSeek
    "deepseek-ai/DeepSeek-V3": {"input": 1.25, "output": 1.25},
    "Qwen/Qwen3-235B-A22B-Instruct-2507-tput": {"input": 0.20, "output": 0.60},
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": {"input": 0.18, "output": 0.59},
    "mistralai/Mistral-Small-24B-Instruct-2501": {"input": 0.80, "output": 0.80},
    
    # XAI
    "grok-4-fast-non-reasoning": {"input": 3.00, "output": 15.00},
}

# ────────────── Abstract LLM Client Class ──────────────
class LLMClient(ABC):
    def __init__(self, model_id: str, temperature: float = 0.6):
        self.model_id = model_id
        self.temperature = temperature
        self.short_model_id = get_short_model_prefix(self.model_id)
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0  # Time to first token

    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage"""
        pricing = PRICING.get(self.model_id, {"input": 0.0, "output": 0.0})
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

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
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                # GPT-5 uses different API
                if self.model_id == "gpt-5":
                    result = self.client.responses.create(
                        model="gpt-5",
                        input=prompt,
                        reasoning={"effort": "low"},
                    )
                    self.last_ttft = time.time() - start_time
                    
                    # Extract response text
                    full_response = result.output_text if hasattr(result, 'output_text') else str(result)
                    
                    # Get usage if available
                    if hasattr(result, 'usage'):
                        self.last_input_tokens = result.usage.input_tokens
                        self.last_output_tokens = result.usage.output_tokens
                        self.last_call_cost = self.calculate_cost(
                            self.last_input_tokens,
                            self.last_output_tokens
                        )
                    
                    if full_response:
                        return full_response.strip()
                else:
                    # Standard chat completions API for other models
                    first_token_received = False
                    full_response = ""
                    
                    stream = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        stream=True,
                        stream_options={"include_usage": True}
                    )
                    
                    for chunk in stream:
                        if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
                            self.last_ttft = time.time() - start_time
                            first_token_received = True
                        
                        if chunk.choices and chunk.choices[0].delta.content:
                            full_response += chunk.choices[0].delta.content
                        
                        # Get usage from final chunk
                        if hasattr(chunk, 'usage') and chunk.usage:
                            self.last_input_tokens = chunk.usage.prompt_tokens
                            self.last_output_tokens = chunk.usage.completion_tokens
                            self.last_call_cost = self.calculate_cost(
                                self.last_input_tokens,
                                self.last_output_tokens
                            )
                    
                    if full_response:
                        return full_response.strip()
                
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
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        last_error = None
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start_time = time.time()
                first_token_received = False
                full_response = ""
                
                # Gemini doesn't support streaming with usage metadata easily
                # So we'll use non-streaming for now
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=0)
                    ),
                )
                
                # Approximate TTFT as we can't get true streaming
                self.last_ttft = time.time() - start_time
                
                # Calculate cost
                if hasattr(resp, 'usage_metadata'):
                    self.last_input_tokens = resp.usage_metadata.prompt_token_count
                    self.last_output_tokens = resp.usage_metadata.candidates_token_count
                    self.last_call_cost = self.calculate_cost(
                        self.last_input_tokens,
                        self.last_output_tokens
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
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                first_token_received = False
                full_response = ""
                
                stream = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=True,
                )
                
                for chunk in stream:
                    if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
                        self.last_ttft = time.time() - start_time
                        first_token_received = True
                    
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                
                # Get token count from non-streaming call (Together doesn't provide usage in streaming)
                # Estimate based on response length
                self.last_input_tokens = len(prompt) // 4  # Rough estimate
                self.last_output_tokens = len(full_response) // 4  # Rough estimate
                self.last_call_cost = self.calculate_cost(
                    self.last_input_tokens,
                    self.last_output_tokens
                )
                
                if full_response:
                    return full_response.strip()
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
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                first_token_received = False
                full_response = ""
                
                with self.client.messages.stream(
                    model=self.model_id,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                ) as stream:
                    for text in stream.text_stream:
                        if not first_token_received:
                            self.last_ttft = time.time() - start_time
                            first_token_received = True
                        full_response += text
                    
                    # Get usage from final message
                    message = stream.get_final_message()
                    if message.usage:
                        self.last_input_tokens = message.usage.input_tokens
                        self.last_output_tokens = message.usage.output_tokens
                        self.last_call_cost = self.calculate_cost(
                            self.last_input_tokens,
                            self.last_output_tokens
                        )
                
                if full_response:
                    return full_response.strip()
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"

# ────────────── XAI Client ──────────────
class XAIClient(LLMClient):
    def __init__(self, model_id: str = "grok-beta", temperature: float = 0.6):
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
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                first_token_received = False
                full_response = ""
                
                stream = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    stream=True,
                )
                
                for chunk in stream:
                    if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
                        self.last_ttft = time.time() - start_time
                        first_token_received = True
                    
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                
                # XAI might not provide usage in streaming
                # Estimate tokens
                self.last_input_tokens = len(prompt) // 4
                self.last_output_tokens = len(full_response) // 4
                self.last_call_cost = self.calculate_cost(
                    self.last_input_tokens,
                    self.last_output_tokens
                )
                
                if full_response:
                    return full_response.strip()
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"