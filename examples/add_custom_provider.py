#!/usr/bin/env python
"""
Example: Adding and Testing Custom OpenAI-Compatible Providers

This script demonstrates how to add custom OpenAI-compatible providers
to DeepTutor, such as Ollama, LM Studio, vLLM, NVIDIA API, and other local models.

Usage:
    python examples/add_custom_provider.py

The .env.example file (lines 17-32) contains configuration for NVIDIA API:
    LLM_BINDING=openai
    LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1
    LLM_BINDING_HOST=https://integrate.api.nvidia.com/v1
    LLM_BINDING_API_KEY=nvapi-...
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.custom_openai_provider import (
    CustomOpenAIProvider,
    add_ollama,
    add_lm_studio,
    add_nvidia,
    quick_test,
)


async def example_1_list_templates():
    """Example 1: List all available provider templates."""
    print("\n" + "="*70)
    print("EXAMPLE 1: List Available Templates")
    print("="*70)
    
    CustomOpenAIProvider.list_templates()


async def example_2_add_ollama():
    """Example 2: Add Ollama as a provider."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Add Ollama Provider")
    print("="*70)
    
    # Add Ollama with llama2 model
    provider = add_ollama(
        model="llama2",
        name="my_ollama",
        base_url="http://localhost:11434/v1",
        set_as_active=True,
    )
    
    if provider:
        print(f"\n✅ Successfully added Ollama provider: {provider.name}")
        print(f"   Model: {provider.model}")
        print(f"   Base URL: {provider.base_url}")


async def example_3_add_lm_studio():
    """Example 3: Add LM Studio as a provider."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Add LM Studio Provider")
    print("="*70)
    
    # Add LM Studio
    provider = add_lm_studio(
        model="local-model",
        name="my_lm_studio",
        base_url="http://localhost:1234/v1",
        set_as_active=False,  # Don't set as active since Ollama might be active
    )
    
    if provider:
        print(f"\n✅ Successfully added LM Studio provider: {provider.name}")


async def example_4_add_custom_vllm():
    """Example 4: Add a custom vLLM provider."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Add Custom vLLM Provider")
    print("="*70)
    
    provider_util = CustomOpenAIProvider()
    
    provider = provider_util.add_provider(
        name="vllm_mistral",
        base_url="http://localhost:8000/v1",
        model="mistralai/Mistral-7B-Instruct-v0.2",
        api_key="sk-no-key-required",  # vLLM doesn't require API key
        binding="openai",
        set_as_active=False,
    )
    
    if provider:
        print(f"\n✅ Successfully added vLLM provider: {provider.name}")


async def example_5_test_model():
    """Example 5: Test a model before adding it."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Test Model Connection")
    print("="*70)
    
    # Test Ollama connection
    print("\nTesting Ollama connection...")
    
    success = await quick_test(
        base_url="http://localhost:11434/v1",
        model="llama2",
        prompt="What is 2+2? Answer briefly.",
    )
    
    if success:
        print("\n✅ Test successful! Model is working correctly.")
    else:
        print("\n❌ Test failed. Check if Ollama is running and model is available.")


async def example_6_list_available_models():
    """Example 6: List all available models from an API."""
    print("\n" + "="*70)
    print("EXAMPLE 6: List Available Models from API")
    print("="*70)
    
    provider_util = CustomOpenAIProvider(api_key="sk-no-key-required")
    
    # List models from Ollama
    print("\nFetching models from Ollama...")
    models = await provider_util.list_models(base_url="http://localhost:11434/v1")
    
    if models:
        print(f"\n✅ Found {len(models)} models:")
        for model in models:
            print(f"  - {model['id']}")
    else:
        print("\n⚠️ No models found. Make sure Ollama is running.")


async def example_7_list_configured_providers():
    """Example 7: List all configured providers in the system."""
    print("\n" + "="*70)
    print("EXAMPLE 7: List Configured Providers")
    print("="*70)
    
    provider_util = CustomOpenAIProvider()
    provider_util.list_providers()


async def example_8_quick_add_with_template():
    """Example 8: Quick add using a template."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Quick Add Using Template")
    print("="*70)
    
    # Quick add LocalAI
    provider = CustomOpenAIProvider.quick_add_template(
        template_name="localai",
        custom_name="my_localai",
        model="gpt-3.5-turbo",  # LocalAI model alias
        set_as_active=False,
    )
    
    if provider:
        print(f"\n✅ Successfully added LocalAI provider: {provider.name}")


async def example_9_add_openai_compatible_api():
    """Example 9: Add any custom OpenAI-compatible API."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Add Custom OpenAI-Compatible API")
    print("="*70)
    
    provider_util = CustomOpenAIProvider()
    
    # Example: Add a custom API endpoint
    provider = provider_util.add_provider(
        name="my_custom_api",
        base_url="http://192.168.1.100:8000/v1",  # Your custom API
        model="custom-model-name",
        api_key="your-api-key-here",  # Or "sk-no-key-required" if not needed
        binding="openai",
        set_as_active=False,
    )
    
    if provider:
        print(f"\n✅ Successfully added custom API provider: {provider.name}")


async def example_10_add_nvidia_api():
    """Example 10: Add NVIDIA API as a provider (as shown in .env.example)."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Add NVIDIA API Provider")
    print("="*70)
    
    # Method 1: Using convenience function (easiest)
    print("\n📝 Method 1: Using add_nvidia() convenience function")
    provider = add_nvidia(
        model="nvidia/llama-3.3-nemotron-super-49b-v1",
        name="nvidia_api",
        set_as_active=True,
        # api_key will be read from LLM_BINDING_API_KEY env var automatically
    )
    
    if provider:
        print(f"\n✅ Successfully added NVIDIA API provider: {provider.name}")
        print(f"   Model: {provider.model}")
        print(f"   Base URL: {provider.base_url}")
        print(f"\n💡 To use this provider:")
        print(f"   1. Set LLM_BINDING_API_KEY in your .env file")
        print(f"   2. The provider is now set as active")
        print(f"   3. Timeout is set to 300s (as per .env.example)")
        
    # Method 2: Using template (alternative)
    print("\n\n📝 Method 2: Using quick_add_template()")
    print("   (Uncomment to use)")
    print("   CustomOpenAIProvider.quick_add_template(")
    print("       template_name='nvidia',")
    print("       custom_name='my_nvidia',")
    print("       set_as_active=True,")
    print("   )")


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DeepTutor: Custom OpenAI Provider Examples")
    print("="*70)
    print("\nThese examples show how to integrate custom OpenAI-compatible")
    print("models into DeepTutor (Ollama, LM Studio, vLLM, NVIDIA API, etc.)")
    print("\nNote: Some examples will fail if the services are not running.")
    print("For NVIDIA API: Set LLM_BINDING_API_KEY in .env (see .env.example L17-32)")
    print("="*70)
    
    # Run examples
    await example_1_list_templates()
    
    # Uncomment the examples you want to run:
    
    # await example_2_add_ollama()
    # await example_3_add_lm_studio()
    # await example_4_add_custom_vllm()
    # await example_5_test_model()
    # await example_6_list_available_models()
    # await example_7_list_configured_providers()
    # await example_8_quick_add_with_template()
    # await example_9_add_openai_compatible_api()
    # await example_10_add_nvidia_api()
    
    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
    print("\nTo use these examples:")
    print("1. Uncomment the examples you want to run")
    print("2. For local servers: Make sure your server is running (Ollama, LM Studio, etc.)")
    print("3. For NVIDIA API: Set LLM_BINDING_API_KEY in .env (see .env.example)")
    print("4. Run: python examples/add_custom_provider.py")
    print("\n" + "="*70)


if __name__ == "__main__":
    asyncio.run(main())

