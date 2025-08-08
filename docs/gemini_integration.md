# Gemini Integration Guide

This guide explains how to use Google Gemini models with the iML framework.

## Setup

1. **Install dependencies**: The required packages are already included in `requirements.txt`:
   - `google-generativeai`
   - `langchain-google-genai`

2. **Set up API key**: You need to set your Google API key as an environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. **Get API key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to create a free API key.

## Configuration

Use the `gemini_example.yaml` configuration file as a template, or modify your existing config:

```yaml
llm:
  provider: gemini
  model: gemini-1.5-pro  # or gemini-1.5-flash, gemini-1.0-pro, etc.
  max_tokens: 8192
  temperature: 1.0
  top_p: 0.9
  top_k: 40
  verbose: True
```

## Dynamic Model Discovery

Gemini integration now follows the **exact same structure** as OpenAI:

```python
from src.iML.llm.gemini_chat import get_gemini_models

# Simple approach - identical to OpenAI
models = get_gemini_models()
print(f"Available models: {models}")
```

### How it works:
1. **Identical to OpenAI**: Same function signature and behavior
2. **Smart Filtering**: Only includes models that support `generateContent`
3. **Consistent Error Handling**: Returns empty list `[]` on failure
4. **Same Patterns**: Easy to understand if you know OpenAI approach

### Structure Comparison:
```python
# OpenAI
def get_openai_models() -> List[str]:
    try:
        client = OpenAI()
        models = client.models.list()
        return [model.id for model in models]
    except Exception as e:
        logger.error(f"Error: {e}")
        return []

# Gemini (identical structure)
def get_gemini_models() -> List[str]:
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        models = genai.list_models()
        # Filter and process models
        return filtered_models
    except Exception as e:
        logger.error(f"Error: {e}")
        return []
```

## Available Models

The integration supports these Gemini models:
- `gemini-1.5-pro` - Most capable model for complex reasoning
- `gemini-1.5-flash` - Faster model for quick responses
- `gemini-1.0-pro` - Previous generation model
- `gemini-1.0-pro-vision` - Model with vision capabilities
- `gemini-pro` - Alias for current pro model
- `gemini-pro-vision` - Alias for current vision model

## Parameters

- `model`: The Gemini model to use
- `max_tokens`: Maximum number of tokens in the response (mapped to `max_output_tokens`)
- `temperature`: Controls randomness (0.0 to 1.0)
- `top_p`: Controls diversity via nucleus sampling (0.0 to 1.0)
- `top_k`: Controls diversity by limiting token choices (1 to 40)
- `verbose`: Enable verbose logging

## Usage Example

```python
from src.iML.llm import ChatLLMFactory
from omegaconf import DictConfig

# Create config
config = DictConfig({
    "provider": "gemini",
    "model": "gemini-1.5-pro",
    "max_tokens": 8192,
    "temperature": 0.7
})

# Create chat model
chat_model = ChatLLMFactory.get_chat_model(config, "my_session")

# Use the model
response = chat_model.assistant_chat("Hello, how can you help me with data science?")
print(response)
```

## Rate Limits

Be aware of Gemini API rate limits:
- Free tier: 15 requests per minute
- Paid tier: Higher limits based on your plan

## Error Handling

Common errors and solutions:
- `ValueError: Google API key not found`: Set the `GOOGLE_API_KEY` environment variable
- Rate limit errors: Reduce request frequency or upgrade your plan
- Model not found: Check that the model name is correct and available in your region
