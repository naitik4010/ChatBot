A web-based chatbot application that connects to multiple AI models through the OpenRouter API. This project features a responsive UI with conversation management, dark mode, and model switching capabilities.

## Features

- **Multiple AI Models**: Seamlessly switch between different AI models (Gemini, Deepseek, Llama 3) through OpenRouter
- **Image Generation**: Generate images using either DALL-E or Gemini's image generation capabilities
- **Conversation Management**: Save, restore, and manage multiple conversations
- **Responsive Design**: Works on desktop and mobile devices with a sidebar for conversation history
- **Dark Mode**: Toggle between light and dark themes
- **Markdown Support**: Format text with code blocks, lists, headings, and more
- **Local Storage**: Conversations persist between sessions using browser storage

## Technical Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python with Flask
- **APIs**: OpenRouter for text generation, Gemini/DALL-E for image generation
- **Deployment**: Easily deployable on any system with Python and a web browser

## Getting Started

1. Install required packages: `pip install flask flask-cors google-generativeai requests pillow`
2. Add your OpenRouter API key in main.py
3. Run the app: `python app.py`
4. Open your browser and navigate to `http://localhost:5000`

This project demonstrates how to build a modern AI assistant interface that can leverage multiple AI models and capabilities through a single, unified experience.
