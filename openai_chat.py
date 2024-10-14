# openai_chat.py

import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI API
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")
openai.api_key = openai_api_key

def get_ai_response(chat_history, user_message, model="gpt-3.5-turbo"):
    """
    Sends the chat history and the latest user message to OpenAI API
    and returns the AI's response.

    Parameters:
    - chat_history (list): List of dictionaries containing previous messages.
    - user_message (str): The latest message from the user.
    - model (str): The OpenAI model to use.

    Returns:
    - str: AI's response message.
    """
    # Append the latest user message to the chat history
    chat_history.append({"role": "user", "content": user_message})

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=chat_history
        )
        ai_message = response['choices'][0]['message']['content'].strip()
        # Append AI's response to the chat history
        chat_history.append({"role": "assistant", "content": ai_message})
        return ai_message
    except Exception as e:
        return f"Error: {e}"
