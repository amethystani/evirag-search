"""
Ollama Cloud Client for EVIRAG
Wraps the ollama.com cloud API (/api/chat) — identical interface to OllamaClient
so it is a transparent drop-in for every engine that previously hit localhost:11434.

Configuration (via .env or environment):
  OLLAMA_API_KEY   — bearer token for ollama.com
  OLLAMA_HOST      — base URL, default https://ollama.com
  OLLAMA_CLOUD_MODEL — model slug, default gpt-oss:120b
"""

import os
import json
import re
import time
import requests
from typing import Optional, Any

# ── Load .env if present ───────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional; key must be in environment already

# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_HOST  = "https://ollama.com"
_DEFAULT_MODEL = "gpt-oss:120b"


def _get_api_key() -> str:
    key = os.getenv("OLLAMA_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "OLLAMA_API_KEY is not set. "
            "Add it to your .env file or export it before starting the server."
        )
    return key.strip()


class OllamaCloudClient:
    """
    HTTP client for the ollama.com cloud /api/chat endpoint.

    Public interface mirrors OllamaClient:
      • generate(model, prompt, temperature, max_tokens, system) -> str
      • extract_json(text) -> Any | None
    """

    MAX_RETRIES  = 3
    RETRY_DELAY  = 1.5  # seconds (doubles on each retry)

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.host  = (host  or os.getenv("OLLAMA_HOST",          _DEFAULT_HOST )).rstrip("/")
        self.model = (model or os.getenv("OLLAMA_CLOUD_MODEL",   _DEFAULT_MODEL))
        self._key  = api_key or _get_api_key()

    # ── Low-level POST ─────────────────────────────────────────────────────────
    def _post(self, endpoint: str, payload: dict, timeout: int = 120) -> dict:
        url = f"{self.host}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type":  "application/json",
        }
        delay = self.RETRY_DELAY
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429:
                    # Rate-limited — exponential back-off
                    if attempt < self.MAX_RETRIES - 1:
                        print(f"[ollama-cloud] Rate-limited. Retrying in {delay:.1f}s …")
                        time.sleep(delay)
                        delay *= 2
                        continue
                print(f"[ollama-cloud] HTTP {status}: {exc}")
                return {}
            except requests.exceptions.Timeout:
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[ollama-cloud] Timeout (attempt {attempt + 1}). Retrying …")
                    time.sleep(delay)
                    delay *= 2
                    continue
                print("[ollama-cloud] Request timed out after all retries.")
                return {}
            except Exception as exc:
                print(f"[ollama-cloud] Unexpected error: {exc}")
                return {}
        return {}

    # ── Public interface (mirrors OllamaClient) ────────────────────────────────
    def generate(
        self,
        model: Optional[str]   = None,
        prompt: str            = "",
        temperature: float     = 0.7,
        max_tokens: int        = 1024,
        system: Optional[str]  = None,
    ) -> str:
        """
        Send a prompt to ollama.com cloud and return the response text.

        Args:
            model:       Override the default cloud model (optional).
            prompt:      User message.
            temperature: Sampling temperature.
            max_tokens:  Maximum tokens to generate.
            system:      Optional system prompt prepended to the conversation.

        Returns:
            Assistant response as a plain string (empty string on failure).
        """
        _model = model or self.model
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":   _model,
            "messages": messages,
            "stream":  False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        data = self._post("/api/chat", payload, timeout=120)
        if not data:
            return ""

        # Ollama /api/chat response: {"message": {"role": "assistant", "content": "…"}}
        content = data.get("message", {}).get("content", "")
        if isinstance(content, str):
            # Strip any residual <think>…</think> tags from reasoning models
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def chat(
        self,
        messages: list,
        model: Optional[str]  = None,
        temperature: float    = 0.7,
        max_tokens: int       = 1024,
        stream: bool          = False,
    ) -> dict:
        """
        Low-level chat call returning the raw response dict.
        Useful for callers that need the full message object.
        """
        _model = model or self.model
        payload = {
            "model":    _model,
            "messages": messages,
            "stream":   stream,
            "options":  {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        return self._post("/api/chat", payload, timeout=120)

    def extract_json(self, text: str) -> Optional[Any]:
        """Extract a JSON object or array from arbitrary model output."""
        # 1. ```json … ``` fenced block
        for pattern in [
            r'```json\s*([\[{].*?[\]}])\s*```',
            r'```\s*([\[{].*?[\]}])\s*```',
        ]:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:
                    pass

        # 2. Bare JSON array
        m = re.search(r'(\[.*\])', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # 3. Bare JSON object
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return None


# ── Singleton accessor ─────────────────────────────────────────────────────────
_cloud_client: Optional[OllamaCloudClient] = None


def get_cloud_client() -> OllamaCloudClient:
    """Return (or lazily create) the shared OllamaCloudClient singleton."""
    global _cloud_client
    if _cloud_client is None:
        _cloud_client = OllamaCloudClient()
    return _cloud_client


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Ollama Cloud Client self-test ===")
    client = OllamaCloudClient()
    print(f"Host  : {client.host}")
    print(f"Model : {client.model}")

    resp = client.generate(prompt="Say 'API working' and nothing else.")
    print(f"Response: {resp!r}")
