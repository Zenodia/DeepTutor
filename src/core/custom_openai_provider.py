#!/usr/bin/env python
"""
Custom OpenAI-Compatible Provider Utility
Allows easy integration of OpenAI API-compatible models (e.g., Ollama, LocalAI, vLLM, etc.)
into the DeepTutor system.

Inspired by: https://github.com/Zenodia/AgenticTA/blob/main/test_OpenAI_w_API_catalog_models.py
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    print("Error: openai package not installed. Install it with: pip install openai")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.llm_provider import LLMProvider, provider_manager
from src.core.core import get_llm_config


class CustomOpenAIProvider:
    """
    Utility for testing and adding custom OpenAI-compatible providers.
    Supports various OpenAI API-compatible backends like:
    - Ollama
    - LocalAI
    - vLLM
    - Text Generation Inference (TGI)
    - LM Studio
    - Jan
    - Any other OpenAI-compatible API
    """

    # Common OpenAI-compatible providers and their typical configurations
    PROVIDER_TEMPLATES = {
        "ollama": {
            "binding": "openai",
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama2",
            "requires_key": False,
            "description": "Ollama - Run large language models locally"
        },
        "lm_studio": {
            "binding": "openai",
            "base_url": "http://localhost:1234/v1",
            "default_model": "local-model",
            "requires_key": False,
            "description": "LM Studio - Local model serving"
        },
        "vllm": {
            "binding": "openai",
            "base_url": "http://localhost:8000/v1",
            "default_model": "model",
            "requires_key": False,
            "description": "vLLM - High-throughput LLM serving"
        },
        "text_generation_inference": {
            "binding": "openai",
            "base_url": "http://localhost:8080/v1",
            "default_model": "tgi",
            "requires_key": False,
            "description": "Text Generation Inference - Hugging Face serving"
        },
        "localai": {
            "binding": "openai",
            "base_url": "http://localhost:8080/v1",
            "default_model": "gpt-3.5-turbo",
            "requires_key": False,
            "description": "LocalAI - OpenAI drop-in replacement"
        },
        "jan": {
            "binding": "openai",
            "base_url": "http://localhost:1337/v1",
            "default_model": "mistral-ins-7b-q4",
            "requires_key": False,
            "description": "Jan - Desktop AI application"
        },
        "openai": {
            "binding": "openai",
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
            "requires_key": True,
            "description": "OpenAI - Official OpenAI API"
        },
        "azure_openai": {
            "binding": "azure_openai",
            "base_url": "https://<your-resource-name>.openai.azure.com",
            "default_model": "gpt-4",
            "requires_key": True,
            "description": "Azure OpenAI Service"
        },
        "nvidia": {
            "binding": "openai",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "default_model": "openai/gpt-oss-120b",
            "requires_key": True,
            "description": "NVIDIA API - OpenAI-compatible endpoint for NVIDIA models"
        },
        "custom": {
            "binding": "openai",
            "base_url": "http://localhost:8000/v1",
            "default_model": "model",
            "requires_key": False,
            "description": "Custom OpenAI-compatible API"
        }
    }

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the custom provider utility.

        Args:
            api_key: API key (optional, can be None for local models)
            base_url: Base URL for the API endpoint
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "sk-no-key-required")
        self.base_url = base_url

    @classmethod
    def from_env(cls):
        """Create instance from environment variables."""
        try:
            config = get_llm_config()
            return cls(api_key=config.get("api_key"), base_url=config.get("base_url"))
        except Exception:
            return cls()

    def sanitize_base_url(self, base_url: str) -> str:
        """
        Sanitize base URL to ensure it's in the correct format.
        Removes common suffixes that users might accidentally include.

        Args:
            base_url: Raw base URL

        Returns:
            Sanitized base URL
        """
        base_url = base_url.rstrip("/")
        
        # Remove common endpoints that should not be in base URL
        suffixes_to_remove = [
            "/chat/completions",
            "/completions",
            "/v1/chat/completions",
            "/v1/completions",
        ]
        
        for suffix in suffixes_to_remove:
            if base_url.endswith(suffix):
                base_url = base_url[:-len(suffix)]
        
        # Ensure /v1 is present for OpenAI-compatible APIs
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        
        return base_url

    async def list_models(self, base_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available models from the API.

        Args:
            base_url: API base URL (uses instance base_url if not provided)

        Returns:
            List of model information dictionaries
        """
        url = base_url or self.base_url
        if not url:
            raise ValueError("base_url must be provided")

        url = self.sanitize_base_url(url)

        try:
            # Use longer timeout for model listing (default 120 seconds)
            timeout = int(os.getenv("LLM_TIMEOUT", "600"))
            client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=url,
                timeout=timeout,
                max_retries=3
            )
            models = await client.models.list()
            
            return [
                {
                    "id": model.id,
                    "created": getattr(model, "created", None),
                    "owned_by": getattr(model, "owned_by", "unknown"),
                }
                for model in models.data
            ]
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []

    async def test_model(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        prompt: str = "Hello! Please respond with a brief greeting.",
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Test a specific model with a prompt.

        Args:
            model_name: Name of the model to test
            base_url: API base URL
            api_key: API key (optional)
            prompt: Test prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Dictionary containing test results
        """
        url = base_url or self.base_url
        key = api_key or self.api_key

        if not url:
            raise ValueError("base_url must be provided")

        url = self.sanitize_base_url(url)

        try:
            print(f"\n{'='*60}")
            print(f"Testing model: {model_name}")
            print(f"Base URL: {url}")
            print(f"{'='*60}")
            print(f"Prompt: {prompt}")
            print(f"\nGenerating response...")

            # Use longer timeout for testing (default 600 seconds)
            timeout = int(os.getenv("LLM_TIMEOUT", "600"))
            client = AsyncOpenAI(
                api_key=key, 
                base_url=url,
                timeout=timeout,
                max_retries=3
            )

            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            result = {
                "model": response.model,
                "success": True,
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }

            print(f"\n✅ Response: {result['response']}")
            print(f"\nToken Usage:")
            print(f"  - Prompt: {result['usage']['prompt_tokens']}")
            print(f"  - Completion: {result['usage']['completion_tokens']}")
            print(f"  - Total: {result['usage']['total_tokens']}")

            return result

        except Exception as e:
            print(f"\n❌ Error testing model {model_name}: {e}")
            return {
                "model": model_name,
                "success": False,
                "error": str(e),
            }

    def add_provider(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        binding: str = "openai",
        set_as_active: bool = False,
    ) -> Optional[LLMProvider]:
        """
        Add a new provider to the system.

        Args:
            name: Unique name for the provider
            base_url: API base URL
            model: Model name
            api_key: API key (optional for local models)
            binding: Provider binding type (default: "openai")
            set_as_active: Whether to set as active provider

        Returns:
            LLMProvider instance if successful, None otherwise
        """
        base_url = self.sanitize_base_url(base_url)
        api_key = api_key or "sk-no-key-required"

        try:
            provider = LLMProvider(
                name=name,
                binding=binding,
                base_url=base_url,
                api_key=api_key,
                model=model,
                is_active=set_as_active,
            )

            added_provider = provider_manager.add_provider(provider)
            print(f"✅ Provider '{name}' added successfully!")
            
            if set_as_active:
                print(f"✅ Provider '{name}' set as active")
            
            return added_provider

        except ValueError as e:
            print(f"❌ Error adding provider: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None

    @classmethod
    def quick_add_template(
        cls,
        template_name: str,
        custom_name: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        set_as_active: bool = False,
    ) -> Optional[LLMProvider]:
        """
        Quickly add a provider using a predefined template.

        Args:
            template_name: Name of the template (ollama, lm_studio, etc.)
            custom_name: Custom name for the provider (optional)
            model: Model name (optional, uses template default)
            base_url: Custom base URL (optional, uses template default)
            api_key: API key (optional)
            set_as_active: Whether to set as active provider

        Returns:
            LLMProvider instance if successful, None otherwise
        """
        if template_name not in cls.PROVIDER_TEMPLATES:
            print(f"❌ Unknown template: {template_name}")
            print(f"Available templates: {', '.join(cls.PROVIDER_TEMPLATES.keys())}")
            return None

        template = cls.PROVIDER_TEMPLATES[template_name]
        
        name = custom_name or template_name
        model = model or template["default_model"]
        base_url = base_url or template["base_url"]
        binding = template["binding"]

        # Use dummy key if not required and not provided
        if not template["requires_key"] and not api_key:
            api_key = "sk-no-key-required"

        provider_util = cls(api_key=api_key, base_url=base_url)
        
        return provider_util.add_provider(
            name=name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            binding=binding,
            set_as_active=set_as_active,
        )

    @classmethod
    def list_templates(cls):
        """List all available provider templates."""
        print("\n" + "="*60)
        print("Available Provider Templates")
        print("="*60)
        
        for name, template in cls.PROVIDER_TEMPLATES.items():
            print(f"\n📦 {name}")
            print(f"   Description: {template['description']}")
            print(f"   Base URL: {template['base_url']}")
            print(f"   Default Model: {template['default_model']}")
            print(f"   Requires API Key: {template['requires_key']}")

    def list_providers(self):
        """List all configured providers in the system."""
        providers = provider_manager.list_providers()
        
        if not providers:
            print("\n📋 No providers configured yet.")
            return []
        
        print("\n" + "="*60)
        print("Configured Providers")
        print("="*60)
        
        for provider in providers:
            status = "✅ ACTIVE" if provider.is_active else "⚪ Inactive"
            print(f"\n{status} {provider.name}")
            print(f"   Binding: {provider.binding}")
            print(f"   Model: {provider.model}")
            print(f"   Base URL: {provider.base_url}")
        
        return providers


# Convenience functions for quick operations

async def quick_test(
    base_url: str,
    model: str,
    api_key: Optional[str] = None,
    prompt: str = "Say hello!",
) -> bool:
    """
    Quick test of a custom OpenAI-compatible API.

    Args:
        base_url: API base URL
        model: Model name
        api_key: API key (optional)
        prompt: Test prompt

    Returns:
        True if successful, False otherwise
    """
    provider = CustomOpenAIProvider(api_key=api_key, base_url=base_url)
    result = await provider.test_model(model_name=model, prompt=prompt)
    return result.get("success", False)


def add_ollama(
    model: str = "llama2",
    name: str = "ollama",
    base_url: str = "http://localhost:11434/v1",
    set_as_active: bool = True,
) -> Optional[LLMProvider]:
    """
    Convenience function to add Ollama as a provider.

    Args:
        model: Ollama model name (e.g., "llama2", "mistral", "codellama")
        name: Provider name
        base_url: Ollama API base URL
        set_as_active: Whether to set as active provider

    Returns:
        LLMProvider instance if successful
    """
    return CustomOpenAIProvider.quick_add_template(
        template_name="ollama",
        custom_name=name,
        model=model,
        base_url=base_url,
        set_as_active=set_as_active,
    )


def add_lm_studio(
    model: str = "local-model",
    name: str = "lm_studio",
    base_url: str = "http://localhost:1234/v1",
    set_as_active: bool = True,
) -> Optional[LLMProvider]:
    """
    Convenience function to add LM Studio as a provider.

    Args:
        model: Model name
        name: Provider name
        base_url: LM Studio API base URL
        set_as_active: Whether to set as active provider

    Returns:
        LLMProvider instance if successful
    """
    return CustomOpenAIProvider.quick_add_template(
        template_name="lm_studio",
        custom_name=name,
        model=model,
        base_url=base_url,
        set_as_active=set_as_active,
    )


def add_nvidia(
    model: str = "openai/gpt-oss-120b",
    name: str = "nvidia",
    base_url: str = "https://integrate.api.nvidia.com/v1",
    api_key: Optional[str] = None,
    set_as_active: bool = True,
) -> Optional[LLMProvider]:
    """
    Convenience function to add NVIDIA API as a provider.

    Args:
        model: NVIDIA model name (e.g., "openai/gpt-oss-120b", "nvidia/llama-3.3-nemotron-super-49b-v1")
        name: Provider name
        base_url: NVIDIA API base URL
        api_key: NVIDIA API key (reads from LLM_BINDING_API_KEY env var if not provided)
        set_as_active: Whether to set as active provider

    Returns:
        LLMProvider instance if successful
    """
    # Try to read API key from environment if not provided
    # Check NVIDIA_API_KEY first (more specific), then fallback to LLM_BINDING_API_KEY
    if not api_key:
        api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
        
    return CustomOpenAIProvider.quick_add_template(
        template_name="nvidia",
        custom_name=name,
        model=model,
        base_url=base_url,
        api_key=api_key,
        set_as_active=set_as_active,
    )


# CLI interface for interactive use
async def main():
    """Main CLI interface for testing and adding custom providers."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Custom OpenAI-Compatible Provider Utility for DeepTutor"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List templates
    parser_templates = subparsers.add_parser("templates", help="List available templates")

    # List providers
    parser_list = subparsers.add_parser("list", help="List configured providers")

    # List models from API
    parser_models = subparsers.add_parser("models", help="List models from API")
    parser_models.add_argument("--base-url", required=True, help="API base URL")
    parser_models.add_argument("--api-key", help="API key (optional)")

    # Test model
    parser_test = subparsers.add_parser("test", help="Test a model")
    parser_test.add_argument("--base-url", required=True, help="API base URL")
    parser_test.add_argument("--model", required=True, help="Model name")
    parser_test.add_argument("--api-key", help="API key (optional)")
    parser_test.add_argument("--prompt", default="Hello!", help="Test prompt")

    # Add provider
    parser_add = subparsers.add_parser("add", help="Add a new provider")
    parser_add.add_argument("--name", required=True, help="Provider name")
    parser_add.add_argument("--base-url", required=True, help="API base URL")
    parser_add.add_argument("--model", required=True, help="Model name")
    parser_add.add_argument("--api-key", help="API key (optional)")
    parser_add.add_argument("--binding", default="openai", help="Provider binding")
    parser_add.add_argument("--active", action="store_true", help="Set as active provider")

    # Quick add from template
    parser_quick = subparsers.add_parser("quick-add", help="Quick add from template")
    parser_quick.add_argument("template", help="Template name")
    parser_quick.add_argument("--name", help="Custom provider name")
    parser_quick.add_argument("--model", help="Model name")
    parser_quick.add_argument("--base-url", help="Custom base URL")
    parser_quick.add_argument("--api-key", help="API key")
    parser_quick.add_argument("--active", action="store_true", help="Set as active provider")

    args = parser.parse_args()

    provider_util = CustomOpenAIProvider()

    if args.command == "templates":
        CustomOpenAIProvider.list_templates()
    
    elif args.command == "list":
        provider_util.list_providers()
    
    elif args.command == "models":
        provider_util.api_key = args.api_key or "sk-no-key-required"
        models = await provider_util.list_models(base_url=args.base_url)
        if models:
            print(f"\n✅ Found {len(models)} models:")
            for model in models:
                print(f"  - {model['id']}")
    
    elif args.command == "test":
        provider_util.api_key = args.api_key or "sk-no-key-required"
        await provider_util.test_model(
            model_name=args.model,
            base_url=args.base_url,
            prompt=args.prompt,
        )
    
    elif args.command == "add":
        provider_util.add_provider(
            name=args.name,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            binding=args.binding,
            set_as_active=args.active,
        )
    
    elif args.command == "quick-add":
        CustomOpenAIProvider.quick_add_template(
            template_name=args.template,
            custom_name=args.name,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            set_as_active=args.active,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

