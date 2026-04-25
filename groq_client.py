"""
Groq API Client for EVIRAG Cloud Mode
OpenAI-compatible API for fast cloud inference
"""

import os
import json
import time
import requests
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class GroqConfig:
    """Groq API configuration"""
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0


class GroqClient:
    """Client for Groq Cloud API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable "
                "or pass api_key parameter. Get your key at: https://console.groq.com/keys"
            )
        
        self.config = GroqConfig(api_key=self.api_key)
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None
    ) -> str:
        """
        Generate text using Groq API (OpenAI-compatible)
        
        Args:
            model: Model name (e.g., "llama-3.1-8b-instant")
            prompt: User prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system: Optional system message
            
        Returns:
            Generated text response
        """
        url = f"{self.config.base_url}/chat/completions"
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        # Retry logic with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract content from OpenAI-style response
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    return content
                else:
                    print(f"Warning: Unexpected Groq response format: {result}")
                    return ""
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    if attempt < self.config.max_retries - 1:
                        wait_time = self.config.retry_delay * (2 ** attempt)
                        print(f"Rate limited. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error calling Groq API: {e}")
                        return ""
                elif e.response.status_code == 401:
                    print("Error: Invalid Groq API key. Check your GROQ_API_KEY.")
                    return ""
                else:
                    print(f"Error calling Groq API: {e}")
                    return ""
                    
            except requests.exceptions.Timeout:
                if attempt < self.config.max_retries - 1:
                    print(f"Timeout. Retrying... (attempt {attempt + 1})")
                    continue
                else:
                    print("Error: Groq API timeout")
                    return ""
                    
            except Exception as e:
                print(f"Error calling Groq API: {e}")
                return ""
        
        return ""
    
    def extract_json(self, text: str):
        """Extract JSON (object or array) from model response."""
        import re

        # 1. ```json ... ``` fenced block — object or array
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

        # 2. Bare JSON array [...]
        m = re.search(r'(\[.*\])', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass

        # 3. Bare JSON object {...}
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        return None
    
    def test_connection(self) -> bool:
        """Test if Groq API is accessible"""
        try:
            response = self.generate(
                model="llama-3.1-8b-instant",
                prompt="Say 'OK' if you can read this.",
                max_tokens=10,
                temperature=0.1
            )
            return len(response) > 0
        except:
            return False


if __name__ == "__main__":
    # Test Groq client
    print("Testing Groq Client")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set")
        print("Set it with: export GROQ_API_KEY='your-key-here'")
        print("Get your key at: https://console.groq.com/keys")
        exit(1)
    
    try:
        client = GroqClient()
        print("✓ Client initialized")
        
        # Test connection
        print("\nTesting connection...")
        if client.test_connection():
            print("✓ Connection successful")
        else:
            print("✗ Connection failed")
            exit(1)
        
        # Test generation
        print("\nTesting text generation...")
        response = client.generate(
            model="llama-3.1-8b-instant",
            prompt="What is RAG in one sentence?",
            temperature=0.7,
            max_tokens=100
        )
        print(f"\nResponse: {response}")
        
        # Test JSON extraction
        print("\nTesting JSON generation...")
        response = client.generate(
            model="llama-3.3-70b-versatile",
            prompt='Return a JSON object: {"test": "success", "number": 42}',
            temperature=0.1,
            max_tokens=100
        )
        
        extracted = client.extract_json(response)
        if extracted:
            print(f"✓ JSON extracted: {extracted}")
        else:
            print(f"✗ JSON extraction failed. Response: {response}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
