from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import main

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def serve_ui():
    return send_from_directory('.', 'ui.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    # Process the message with your Gemini chatbot
    response = main.chat_with_gemini(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)