import os
import requests
import base64
from datetime import datetime

# API Keys
OPENROUTER_API_KEY = "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"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your actual Gemini API key

# Try to import Gemini libraries
try:
    from google import genai
    from google.genai import types
    from PIL import Image
    from io import BytesIO
    GEMINI_AVAILABLE = True
    
    # Initialize Gemini
    genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False

# Available models
AVAILABLE_MODELS = {
    "deepseek": "deepseek/deepseek-r1:free",
    "gemini-2.5": "google/gemini-2.5-pro-exp-03-25:free",
    "gemini-2.0": "google/gemini-2.0-flash-exp:free",
    "llama3": "nvidia/llama-3.1-nemotron-70b-instruct:free"
}

# Image generation models
IMAGE_MODELS = {
    "gemini": "gemini-2.0-flash-exp-image-generation"
}

# Default model settings
DEFAULT_MODEL = "gemini-2.0"
DEFAULT_IMAGE_MODEL = "gemini"  # Options: "dall-e" or "gemini"

# Create images directory if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("static/images"):
    os.makedirs("static/images")

def chat_with_openrouter(user_input, model_key=DEFAULT_MODEL, image_model_key=DEFAULT_IMAGE_MODEL):
    # Check if this is an image generation request
    if user_input.lower().startswith(("generate image", "create image", "draw", "make an image")):
        return generate_image(user_input, image_model_key)
    
    # If there's a model switch command
    if user_input.lower().startswith("/model "):
        parts = user_input.split(" ", 1)
        if len(parts) > 1:
            requested_model = parts[1].strip().lower()
            if requested_model in AVAILABLE_MODELS:
                return f"Model switched to {requested_model} ({AVAILABLE_MODELS[requested_model]})"
            else:
                available = ", ".join(f"'{k}'" for k in AVAILABLE_MODELS.keys())
                return f"Model '{requested_model}' not found. Available models: {available}"
        else:
            available = ", ".join(f"'{k}'" for k in AVAILABLE_MODELS.keys())
            return f"Current model: '{model_key}'. Available models: {available}"
    
    # If there's an image model switch command
    if user_input.lower().startswith("/image "):
        parts = user_input.split(" ", 1)
        if len(parts) > 1:
            requested_model = parts[1].strip().lower()
            if requested_model in IMAGE_MODELS or requested_model in IMAGE_MODELS.keys():
                DEFAULT_IMAGE_MODEL = requested_model
                return f"Image generation model switched to {requested_model}"
            else:
                available = ", ".join(f"'{k}'" for k in IMAGE_MODELS.keys())
                return f"Image model '{requested_model}' not found. Available models: {available}"
        else:
            available = ", ".join(f"'{k}'" for k in IMAGE_MODELS.keys())
            return f"Current image model: '{DEFAULT_IMAGE_MODEL}'. Available models: {available}"
    
    # Regular text generation
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app-url.com",  # Replace with your app's URL
        "X-Title": "Chatbot App"  # Replace with your app's name
    }
    
    # Get the actual model ID from the key
    model_id = AVAILABLE_MODELS.get(model_key, AVAILABLE_MODELS[DEFAULT_MODEL])
    
    # Using the selected model for text generation
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "temperature": 1.0,  # Maximum creativity
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        result = response.json()
        
        # Debug: print the response structure
        print("API Response:", result)
        
        # Handle the response structure correctly
        if "choices" in result and len(result["choices"]) > 0:
            if "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
                return result["choices"][0]["message"]["content"]
            elif "text" in result["choices"][0]:
                return result["choices"][0]["text"]
        
        # If we can't find the expected structure, return the raw response
        return f"API returned unexpected structure: {result}"
    except Exception as e:
        # Handle potential API blocking of unsafe content
        return f"The model declined to respond: {str(e)}"

def generate_image_with_gemini(prompt):
    """Generate an image using Gemini's native image generation"""
    if not GEMINI_AVAILABLE:
        return None, "Gemini image generation is not available. Please install the required packages: pip install google-generativeai pillow"
    
    try:
        # Clean up the prompt
        image_prompt = prompt.replace("generate image", "").replace("create image", "").replace("draw", "").replace("make an image", "").strip()
        
        # If prompt is too short, enhance it
        if len(image_prompt) < 10:
            image_prompt += ", detailed, high quality, 4k"
            
        # Generate the image
        response = genai.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=image_prompt,
            generation_config=types.GenerateContentConfig(
                response_modalities=['Text', 'Image']
            )
        )
        
        # Extract image and text
        description = ""
        image_data = None
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text') and part.text is not None:
                description = part.text
            elif hasattr(part, 'inline_data') and part.inline_data is not None:
                image_data = base64.b64decode(part.inline_data.data)
        
        if not image_data:
            return None, "No image was generated"
            
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/images/gemini_{timestamp}.png"
        
        with open(filename, "wb") as f:
            f.write(image_data)
            
        return filename, description
        
    except Exception as e:
        return None, f"Error generating image with Gemini: {str(e)}"

def generate_image(prompt, image_model=DEFAULT_IMAGE_MODEL):
    """Generate an image using the specified provider"""
    # Extract the actual prompt from the command
    image_prompt = prompt.replace("generate image", "").replace("create image", "").replace("draw", "").replace("make an image", "").strip()
    
    # If prompt is too short, enhance it
    if len(image_prompt) < 10:
        image_prompt += ", detailed, high quality, 4k"
    
    # Use Gemini if requested
    if image_model == "gemini" and GEMINI_AVAILABLE:
        try:
            filename, description = generate_image_with_gemini(prompt)
            if filename:
                return f"""**Generated image with Gemini for:** '{image_prompt}'

![Generated Image]({filename})

{description}"""
            else:
                # If Gemini fails, fall back to DALL-E
                return generate_image_with_dalle(image_prompt)
        except Exception as e:
            print(f"Gemini image generation failed: {e}")
            # Fall back to DALL-E
            return generate_image_with_dalle(image_prompt)
    else:
        # Use DALL-E
        return generate_image_with_dalle(image_prompt)

def generate_image_with_dalle(image_prompt):
    """Generate an image using OpenRouter's DALL-E 3 integration"""
    try:
        # Use OpenRouter's API to generate the image with Dalle-3
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-url.com",  # Replace with your app's URL
            "X-Title": "Chatbot App"  # Replace with your app's name
        }
        
        data = {
            "model": "openai/dall-e-3",  # Using DALL-E 3 through OpenRouter
            "prompt": image_prompt,
            "n": 1,
            "size": "1024x1024",
            "response_format": "b64_json"
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/images/generations",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
        result = response.json()
        
        # Extract and save the image
        image_data = base64.b64decode(result["data"][0]["b64_json"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/images/dalle_{timestamp}.png"
        
        with open(filename, "wb") as f:
            f.write(image_data)
            
        # Return markdown with image reference
        return f"**Generated image with DALL-E for:** '{image_prompt}'\n\n![Generated Image]({filename})"
            
    except Exception as e:
        # If image generation fails, provide a detailed error and fallback to text description
        try:
            # Generate a text description of what the image would look like
            description_response = chat_with_openrouter(f"Describe in vivid detail what an image of {image_prompt} would look like. Be extremely descriptive about colors, composition, lighting, and style.")
            
            return f"""## Failed to generate image: {str(e)}

Instead, here's a detailed description of what the image would look like:

{description_response}
"""
        except:
            return f"Error generating image: {str(e)}"

# Only run this part when the script is run directly, not when imported
if __name__ == "__main__":
    # Chatbot loop
    print("🤖 Chatbot: Hello! Ask me anything or ask me to generate an image. Type 'exit' to quit.")
    while True:
        user_input = input('👤 User: ')
        if user_input.lower() == 'exit':
            print("🤖 Chatbot: Goodbye!")
            break
        response = chat_with_openrouter(user_input)
        print(f'🤖 Chatbot: {response}')