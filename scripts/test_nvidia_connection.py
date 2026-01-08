#!/usr/bin/env python
"""
Test NVIDIA API Connection
Quick script to verify your NVIDIA API is working correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_nvidia_api():
    """Test NVIDIA API connection with a simple prompt."""
    
    print("\n" + "="*70)
    print("Testing NVIDIA API Connection")
    print("="*70)
    
    # Get configuration
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    base_url = os.getenv("LLM_BINDING_HOST", "https://integrate.api.nvidia.com/v1")
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    timeout = int(os.getenv("LLM_TIMEOUT", "600"))
    
    # Validate configuration
    if not api_key or api_key == "your_nvidia_api_key_here":
        print("\n❌ Error: NVIDIA API key not found!")
        print("\nPlease set your API key in one of these ways:")
        print("1. Create a .env file with: NVIDIA_API_KEY=your_key")
        print("2. Set environment variable: export NVIDIA_API_KEY=your_key")
        print("\nGet your key from: https://build.nvidia.com/")
        return False
    
    print(f"\n📋 Configuration:")
    print(f"   Model: {model}")
    print(f"   Base URL: {base_url}")
    print(f"   API Key: {'*' * 10}{api_key[-4:] if len(api_key) > 4 else '****'}")
    print(f"   Timeout: {timeout} seconds")
    
    try:
        print(f"\n🔄 Connecting to NVIDIA API...")
        
        # Create client with timeout
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=3
        )
        
        print(f"✅ Client created successfully")
        print(f"\n💬 Sending test prompt...")
        
        # Test with a simple prompt
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello! I am working correctly.' and tell me a joke about atoms."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        # Extract response
        content = response.choices[0].message.content
        
        print(f"\n✅ SUCCESS! NVIDIA API is working correctly!\n")
        print(f"{'='*70}")
        print(f"Response from {model}:")
        print(f"{'='*70}")
        print(content)
        print(f"{'='*70}\n")
        
        # Display token usage
        if hasattr(response, 'usage') and response.usage:
            print(f"📊 Token Usage:")
            print(f"   Prompt tokens: {response.usage.prompt_tokens}")
            print(f"   Completion tokens: {response.usage.completion_tokens}")
            print(f"   Total tokens: {response.usage.total_tokens}")
        
        print(f"\n✅ Connection test passed! You can now use DeepTutor with NVIDIA API.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Verify your API key is correct")
        print(f"2. Check your internet connection")
        print(f"3. Ensure the model '{model}' is available")
        print(f"4. Try increasing LLM_TIMEOUT if you see timeout errors")
        print(f"\nFor more help, see NVIDIA_SETUP.md")
        
        import traceback
        print(f"\nDetailed error:")
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    try:
        success = asyncio.run(test_nvidia_api())
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

