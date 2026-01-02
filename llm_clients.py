import os
import time
import requests
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ────────────── Configuration ──────────────
MAX_RETRIES = 3
RETRY_DELAY = 2
CONNECT_TIMEOUT = 30 
READ_TIMEOUT = 180  
READ_TIMEOUT_REASONING = 300  
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ────────────── Unified LLM Client using OpenRouter ──────────────
class LLMClient:
    """
    Unified LLM client using OpenRouter API.
    
    Supports all providers through OpenRouter.
    See https://openrouter.ai/models for full list.
    
    Usage:
        client = LLMClient("openai/gpt-4.1")
        client = LLMClient("anthropic/claude-sonnet-4")
        client = LLMClient("google/gemini-2.5-flash-preview")
    """
    
    def __init__(
        self,
        model_id: str,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            model_id: OpenRouter model ID (e.g., "openai/gpt-4.1", "anthropic/claude-sonnet-4")
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response (None for model default)
            reasoning_effort: Reasoning effort level ("low", "medium", "high") for reasoning models
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        # Use only the part after '/' in model name (e.g., "openai/gpt-4.1" -> "gpt-4.1")
        self.short_model_id = model_id.split('/')[-1] if '/' in model_id else model_id
        
        # Get API key
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Headers for OpenRouter API
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Last call metrics (updated after each get_response call)
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0  # Time to first token (approximate for non-streaming)
        self.last_generation_id = ""
        self.last_prompt_cost = 0.0
        self.last_completion_cost = 0.0

    def get_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Get a response from the LLM.
        
        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt
            
        Returns:
            Model response text, or error message if all retries fail
        """
        # Reset metrics
        self.last_call_cost = 0.0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_ttft = 0.0
        self.last_generation_id = ""
        self.last_prompt_cost = 0.0
        self.last_completion_cost = 0.0
        
        if self.reasoning_effort:
            return self._get_response_reasoning(prompt, system_prompt)
        else:
            return self._get_response_chat(prompt, system_prompt)
    
    def _get_response_reasoning(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if system_prompt:
            full_input = f"{system_prompt}\n\n{prompt}"
        else:
            full_input = prompt
        
        # Build request payload for /responses endpoint
        payload = {
            "model": self.model_id,
            "input": full_input,
            "reasoning": {
                "enabled": True,
                "effort": self.reasoning_effort
            },
        }
        
        if self.max_tokens:
            payload["reasoning"]["max_tokens"] = self.max_tokens
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/responses",
                    headers=self.headers,
                    json=payload,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_REASONING)
                )
                
                self.last_ttft = time.time() - start_time
                
                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("error", {}).get("message", response.text)
                    except:
                        pass
                    last_error = f"HTTP {response.status_code}: {error_detail} (Attempt {attempt+1})"
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * (attempt + 1) * 2))
                        print(f"Rate limited. Waiting {retry_after}s before retry...")
                        time.sleep(retry_after)
                    else:
                        time.sleep(RETRY_DELAY * (attempt + 1))  # 지수 백오프
                    continue
                
                response_json = response.json()
                
                full_response = ""
                
                if response_json.get("output_text"):
                    full_response = response_json["output_text"]
                elif "output" in response_json:
                    output = response_json["output"]
                    if isinstance(output, list):
                        for item in output:
                            if isinstance(item, dict) and item.get("type") == "message":
                                content = item.get("content", [])
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") == "output_text":
                                        full_response += c.get("text", "")
                
                if not full_response or not full_response.strip():
                    last_error = f"Empty response (Attempt {attempt+1})"
                    time.sleep(RETRY_DELAY)
                    continue
                
                # Extract usage and cost
                self.last_generation_id = response_json.get("id", "")
                usage = response_json.get("usage", {})
                
                self.last_input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                self.last_output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
                
                # If cost is 0, use upstream_inference_cost from cost_details
                cost_details = usage.get("cost_details", {})
                self.last_call_cost = usage.get("cost", 0.0)
                if self.last_call_cost == 0:
                    self.last_call_cost = cost_details.get("upstream_inference_cost", 0.0)
                
                self.last_prompt_cost = cost_details.get("upstream_inference_input_cost", cost_details.get("upstream_inference_prompt_cost", 0.0))
                self.last_completion_cost = cost_details.get("upstream_inference_output_cost", cost_details.get("upstream_inference_completions_cost", 0.0))
                
                return full_response.strip()
                
            except requests.exceptions.Timeout:
                last_error = f"Request timeout (Attempt {attempt+1})"
                print(f"Timeout on attempt {attempt+1}, retrying...")
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e} (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error: {e} (Attempt {attempt+1})"
            
            time.sleep(RETRY_DELAY * (attempt + 1))
        
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    
    def _get_response_chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Use /chat/completions endpoint for general models"""
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Build request payload
        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "reasoning": {
                "enabled": False
            },
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
                )
                
                self.last_ttft = time.time() - start_time
                
                # Check for HTTP errors
                if response.status_code != 200:
                    error_detail = response.text
                    try:
                        error_json = response.json()
                        error_detail = error_json.get("error", {}).get("message", response.text)
                    except:
                        pass
                    last_error = f"HTTP {response.status_code}: {error_detail} (Attempt {attempt+1})"
                    # Rate limit인 경우 더 오래 대기
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", RETRY_DELAY * (attempt + 1) * 2))
                        print(f"Rate limited. Waiting {retry_after}s before retry...")
                        time.sleep(retry_after)
                    else:
                        time.sleep(RETRY_DELAY * (attempt + 1))  # 지수 백오프
                    continue
                
                response_json = response.json()
                
                # Extract response text
                if not response_json.get("choices"):
                    last_error = f"No choices in response (Attempt {attempt+1})"
                    time.sleep(RETRY_DELAY)
                    continue
                
                full_response = response_json["choices"][0].get("message", {}).get("content", "")
                
                if not full_response or not full_response.strip():
                    last_error = f"Empty response (Attempt {attempt+1})"
                    time.sleep(RETRY_DELAY)
                    continue
                
                # Extract usage and cost information
                self.last_generation_id = response_json.get("id", "")
                usage = response_json.get("usage", {})
                
                self.last_input_tokens = usage.get("prompt_tokens", 0)
                self.last_output_tokens = usage.get("completion_tokens", 0)
                
                # If cost is 0, use upstream_inference_cost from cost_details
                cost_details = usage.get("cost_details", {})
                self.last_call_cost = usage.get("cost", 0.0)
                if self.last_call_cost == 0:
                    self.last_call_cost = cost_details.get("upstream_inference_cost", 0.0)
                
                self.last_prompt_cost = cost_details.get("upstream_inference_input_cost", cost_details.get("upstream_inference_prompt_cost", 0.0))
                self.last_completion_cost = cost_details.get("upstream_inference_output_cost", cost_details.get("upstream_inference_completions_cost", 0.0))
                
                return full_response.strip()
                
            except requests.exceptions.Timeout:
                last_error = f"Request timeout (Attempt {attempt+1})"
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e} (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error: {e} (Attempt {attempt+1})"
            
            time.sleep(RETRY_DELAY)
        
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"



    def get_usage_summary(self) -> dict:
        """Get a summary of the last API call's usage and cost."""
        return {
            "model": self.model_id,
            "generation_id": self.last_generation_id,
            "input_tokens": self.last_input_tokens,
            "output_tokens": self.last_output_tokens,
            "total_tokens": self.last_input_tokens + self.last_output_tokens,
            "prompt_cost": self.last_prompt_cost,
            "completion_cost": self.last_completion_cost,
            "total_cost": self.last_call_cost,
            "ttft_seconds": self.last_ttft,
        }

    @staticmethod
    def list_models(api_key: Optional[str] = None) -> list:
        """
        List all available models on OpenRouter.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            
        Returns:
            List of model info dicts
        """
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        headers = {"Authorization": f"Bearer {key}"}
        response = requests.get(f"{OPENROUTER_BASE_URL}/models", headers=headers)
        response.raise_for_status()
        return response.json().get("data", [])

    def __repr__(self) -> str:
        return f"LLMClient(model_id='{self.model_id}', temperature={self.temperature})"