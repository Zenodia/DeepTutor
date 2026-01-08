#!/usr/bin/env python
"""
LLM Provider Initialization Script
Automatically creates and activates LLM provider from environment variables on startup.
This is especially useful for Docker deployments where providers need to be configured
from the .env file.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def init_provider_from_env():
    """
    Initialize LLM provider from environment variables.
    
    This function checks if LLM configuration is provided via environment variables
    and automatically creates/updates the provider in llm_providers.json.
    
    Priority:
    1. Check if there's already an active provider -> skip initialization
    2. Check environment variables and create provider if all required vars are present
    """
    try:
        from src.core.llm_provider import provider_manager
        from src.core.custom_openai_provider import CustomOpenAIProvider
        
        # Check if there's already an active provider
        active_provider = provider_manager.get_active_provider()
        if active_provider:
            print(f"✅ Active LLM provider already configured: {active_provider.name}")
            print(f"   Model: {active_provider.model}")
            print(f"   Binding: {active_provider.binding}")
            return True
        
        # Get environment variables
        binding = os.getenv("LLM_BINDING", "openai").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        api_key = os.getenv("LLM_BINDING_API_KEY", "").strip()
        base_url = os.getenv("LLM_BINDING_HOST", "").strip()
        
        # Check if all required variables are present
        if not model:
            print("⚠️  LLM_MODEL not set - skipping provider initialization")
            return False
        
        if not base_url:
            print("⚠️  LLM_BINDING_HOST not set - skipping provider initialization")
            return False
        
        if not api_key:
            # Check if API key is required
            requires_key = os.getenv("LLM_API_KEY_REQUIRED", "true").lower() == "true"
            if requires_key:
                print("⚠️  LLM_BINDING_API_KEY not set - skipping provider initialization")
                return False
            else:
                api_key = "sk-no-key-required"
        
        # Determine provider name based on base_url
        provider_name = "auto_env_provider"
        
        # Detect provider type from base_url
        if "nvidia.com" in base_url.lower():
            provider_name = "nvidia_api"
            print("🔍 Detected NVIDIA API configuration")
        elif "openai.com" in base_url.lower():
            provider_name = "openai_api"
            print("🔍 Detected OpenAI API configuration")
        elif "localhost:11434" in base_url or "ollama" in base_url.lower():
            provider_name = "ollama"
            print("🔍 Detected Ollama configuration")
        elif "localhost:1234" in base_url:
            provider_name = "lm_studio"
            print("🔍 Detected LM Studio configuration")
        else:
            print(f"🔍 Custom OpenAI-compatible API detected: {base_url}")
        
        # Create provider utility
        provider_util = CustomOpenAIProvider(api_key=api_key, base_url=base_url)
        
        # Add provider
        print(f"📦 Creating LLM provider from environment variables...")
        print(f"   Name: {provider_name}")
        print(f"   Model: {model}")
        print(f"   Binding: {binding}")
        print(f"   Base URL: {base_url}")
        
        provider = provider_util.add_provider(
            name=provider_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            binding=binding,
            set_as_active=True,
        )
        
        if provider:
            print(f"✅ Successfully initialized LLM provider: {provider_name}")
            print(f"   This provider is now active and will be used by DeepTutor")
            return True
        else:
            print("❌ Failed to initialize LLM provider")
            return False
            
    except Exception as e:
        print(f"⚠️  Error initializing LLM provider: {e}")
        print(f"   Provider will be configured from environment variables at runtime")
        return False


def main():
    """Main entry point for provider initialization."""
    print("\n" + "=" * 70)
    print("LLM Provider Initialization")
    print("=" * 70)
    
    success = init_provider_from_env()
    
    print("=" * 70 + "\n")
    
    # Return exit code (0 = success, 1 = failure)
    # Note: We don't exit with error even if initialization fails,
    # as the system can still work with environment variables
    return 0


if __name__ == "__main__":
    sys.exit(main())

