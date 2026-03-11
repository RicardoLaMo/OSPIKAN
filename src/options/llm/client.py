"""
Ollama HTTP Client for LLM inference.

Communicates with Ollama API for model generation and health checks.
"""

import requests
from typing import Optional, Dict, Any
import json


class OllamaClient:
    """HTTP client for Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5-coder:7b"):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama server URL (default: localhost:11434)
            model: Model name (default: qwen2.5-coder:7b for code-tuned output)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoints = {
            "generate": f"{self.base_url}/api/generate",
            "chat": f"{self.base_url}/api/chat",
            "health": f"{self.base_url}/api/tags",  # Use /api/tags for health check
        }

    def health_check(self) -> bool:
        """
        Check if Ollama server is running and model is available.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(self.endpoints["health"], timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Check if our model is in the available models
                models = data.get("models", [])
                model_names = [m.get("name") for m in models]
                return self.model in model_names or any(self.model.split(":")[0] in m for m in model_names)
            return False
        except (requests.ConnectionError, requests.Timeout):
            return False
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 500,
    ) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            system: System prompt/instructions
            temperature: Temperature for sampling (0-1, lower = more deterministic)
            top_p: Top-p nucleus sampling (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            RuntimeError: If Ollama server is not available
            requests.RequestException: If API call fails
        """
        if not self.health_check():
            raise RuntimeError(
                f"Ollama server not available at {self.base_url} with model {self.model}. "
                "Please start Ollama and ensure the model is available."
            )

        # Build full prompt with system instruction
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        }

        try:
            response = requests.post(
                self.endpoints["generate"],
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            return data.get("response", "").strip()

        except requests.Timeout:
            raise RuntimeError("Ollama request timeout (60s)")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 500,
    ) -> str:
        """
        Chat with Ollama model (message-based interface).

        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Temperature for sampling
            top_p: Top-p nucleus sampling
            max_tokens: Maximum tokens to generate

        Returns:
            Model response text

        Raises:
            RuntimeError: If Ollama server is not available
            requests.RequestException: If API call fails
        """
        if not self.health_check():
            raise RuntimeError(
                f"Ollama server not available at {self.base_url} with model {self.model}. "
                "Please start Ollama and ensure the model is available."
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        }

        try:
            response = requests.post(
                self.endpoints["chat"],
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            return data.get("message", {}).get("content", "").strip()

        except requests.Timeout:
            raise RuntimeError("Ollama request timeout (60s)")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")

    def stream_generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Stream text generation (yields tokens as they arrive).

        Args:
            prompt: Input prompt
            system: System prompt/instructions
            temperature: Temperature for sampling
            top_p: Top-p nucleus sampling

        Yields:
            Generated text chunks

        Raises:
            RuntimeError: If Ollama server is not available
        """
        if not self.health_check():
            raise RuntimeError(
                f"Ollama server not available at {self.base_url} with model {self.model}. "
                "Please start Ollama and ensure the model is available."
            )

        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
        }

        try:
            response = requests.post(
                self.endpoints["generate"],
                json=payload,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk

        except requests.Timeout:
            raise RuntimeError("Ollama request timeout (60s)")
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama streaming error: {e}")
