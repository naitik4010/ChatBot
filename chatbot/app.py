from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import main

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Serve static files (for images)
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/')
def serve_ui():
    return send_from_directory('.', 'ui.html')

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'models': main.AVAILABLE_MODELS,
        'image_models': main.IMAGE_MODELS,
        'default': main.DEFAULT_MODEL,
        'default_image': main.DEFAULT_IMAGE_MODEL
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    model_key = data.get('model', main.DEFAULT_MODEL)
    
    # Process the message with your OpenRouter chatbot
    response = main.chat_with_openrouter(user_message, model_key)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)