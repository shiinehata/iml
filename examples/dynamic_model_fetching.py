#!/usr/bin/env python3
"""
Example: Dynamic Model Fetching Comparison
Shows how both OpenAI and Gemini can fetch models dynamically
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def example_openai_dynamic_models():
    """Example of OpenAI dynamic model fetching"""
    print("🔵 OpenAI Dynamic Model Fetching")
    print("-" * 40)
    
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ OPENAI_API_KEY not set - skipping OpenAI example")
        return
    
    try:
        from iML.llm.openai_chat import get_openai_models
        
        # Simple approach - direct API call
        models = get_openai_models()
        
        if models:
            print(f"✅ Found {len(models)} OpenAI models")
            print("📋 Sample models:")
            for model in models[:10]:  # Show first 10
                print(f"   • {model}")
            if len(models) > 10:
                print(f"   ... and {len(models) - 10} more")
        else:
            print("❌ No models returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def example_gemini_dynamic_models():
    """Example of Gemini dynamic model fetching"""
    print("\n🟡 Gemini Dynamic Model Fetching")
    print("-" * 40)
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("❌ GOOGLE_API_KEY not set - skipping Gemini example")
        return
    
    try:
        from iML.llm.gemini_chat import (
            get_gemini_models, 
            refresh_gemini_models,
            test_gemini_api_connection
        )
        
        # Test connection first
        print("🔍 Testing API connection...")
        if not test_gemini_api_connection():
            print("❌ Cannot connect to Gemini API")
            return
        
        # Enhanced approach with caching and fallback
        print("📥 Fetching models (with caching)...")
        models = get_gemini_models()
        
        print(f"✅ Found {len(models)} Gemini models")
        print("📋 All models:")
        for model in models:
            print(f"   • {model}")
        
        # Demonstrate cache refresh
        print("\n🔄 Refreshing cache...")
        refreshed_models = refresh_gemini_models()
        print(f"✅ Refreshed: {len(refreshed_models)} models")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_gemini_fallback():
    """Example of Gemini fallback behavior"""
    print("\n🟠 Gemini Fallback Behavior")
    print("-" * 40)
    
    # Temporarily hide API key to test fallback
    original_key = os.environ.get("GOOGLE_API_KEY")
    if original_key:
        del os.environ["GOOGLE_API_KEY"]
    
    try:
        from iML.llm.gemini_chat import get_gemini_models
        
        print("🔍 Testing without API key...")
        models = get_gemini_models()
        
        print(f"✅ Fallback successful: {len(models)} default models")
        print("📋 Default models:")
        for model in models:
            print(f"   • {model}")
            
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
    finally:
        # Restore API key
        if original_key:
            os.environ["GOOGLE_API_KEY"] = original_key

def compare_implementations():
    """Compare the two implementations"""
    print("\n📊 Implementation Comparison")
    print("=" * 50)
    
    print("\n🔵 OpenAI Approach:")
    print("```python")
    print("def get_openai_models() -> List[str]:")
    print("    try:")
    print("        client = OpenAI()")
    print("        models = client.models.list()")
    print("        return [model.id for model in models]")
    print("    except Exception as e:")
    print("        logger.error(f'Error: {e}')")
    print("        return []  # Returns empty on failure")
    print("```")
    
    print("\n🟡 Gemini Approach:")
    print("```python")
    print("@lru_cache(maxsize=1)")
    print("def get_gemini_models() -> List[str]:")
    print("    try:")
    print("        if 'GOOGLE_API_KEY' not in os.environ:")
    print("            return _get_default_gemini_models()")
    print("        ")
    print("        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])")
    print("        models = genai.list_models()")
    print("        # Filter and clean model names")
    print("        return sorted(valid_models)")
    print("    except Exception:")
    print("        return _get_default_gemini_models()  # Always usable")
    print("```")
    
    print("\n🎯 Key Differences:")
    print("┌─────────────────┬──────────────┬──────────────┐")
    print("│ Feature         │ OpenAI       │ Gemini       │")
    print("├─────────────────┼──────────────┼──────────────┤")
    print("│ Caching         │ No           │ Yes (@lru)   │")
    print("│ Fallback        │ Empty list   │ Default list │")
    print("│ Filtering       │ All models   │ Compatible   │")
    print("│ Error Handling  │ Basic        │ Robust       │")
    print("│ Reliability     │ May fail     │ Always works │")
    print("└─────────────────┴──────────────┴──────────────┘")

def main():
    """Run all examples"""
    print("🚀 Dynamic Model Fetching Examples")
    print("=" * 50)
    
    # Run examples
    example_openai_dynamic_models()
    example_gemini_dynamic_models()
    example_gemini_fallback()
    
    # Show comparison
    compare_implementations()
    
    print("\n🎉 Summary:")
    print("• Both approaches support dynamic model fetching")
    print("• Gemini approach is more robust with caching and fallback")
    print("• No more hardcoded model lists needed!")
    print("• API keys still required for live fetching")

if __name__ == "__main__":
    main()
