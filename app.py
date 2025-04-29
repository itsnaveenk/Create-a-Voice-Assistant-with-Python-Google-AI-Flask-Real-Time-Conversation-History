import google.generativeai as genai
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

app = Flask(__name__)

# Get API key from environment variable
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("No API key found. Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-pro")

# Global variables to store conversation history and context
# Limit history to last 10 conversations to prevent memory issues
MAX_HISTORY_LENGTH = 10
conversation_history = []

# Voice assistance function with enhanced topic management
def voice_assistance(user_input):
    global conversation_history

    # Build context from previous conversation
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for entry in conversation_history[-3:]:  # Use last 3 exchanges for context
            context += f"User: {entry['user']}\nAI: {entry['ai']}\n"
        context += "\n"

    # Improved prompt with conversation history
    prompt = f"""
    You are an AI assistant in an engaging conversation with a user. 
    
    {context}
    
    The user just asked the following question:
    '{user_input}'
    
    Provide a direct and informative answer, focusing on the exact details the user is asking for. Avoid unnecessary elaboration or asking follow-up questions unless essential to the user's inquiry. Keep the response clear, concise, and to the point. If the topic is complex, briefly summarize the key aspects.
    """

    try:
        response = model.generate_content(prompt).text
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I'm sorry, I couldn't process that request. Please try again later."

    # Update conversation history
    conversation_history.append({
        'user': user_input,
        'ai': response
    })
    
    # Limit conversation history size
    if len(conversation_history) > MAX_HISTORY_LENGTH:
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH:]

    return response

# Route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle voice input and return model response with conversation history
@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400
            
        user_input = data.get("user_input", "")
        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400
        
        response = voice_assistance(user_input)
        
        return jsonify({
            'response': response, 
            'conversation_history': conversation_history
        })
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Server error processing request'}), 500

if __name__ == '__main__':
    app.run(debug=True)