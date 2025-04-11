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
    and returns only the AI's response, without modifying the chat history.

    Parameters:
    - chat_history (list): List of dictionaries containing previous messages.
    - user_message (str): The latest message from the user.
    - model (str): The OpenAI model to use.

    Returns:
    - str: AI's response message.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=chat_history + [{"role": "user", "content": user_message}]
        )
        ai_message = response['choices'][0]['message']['content'].strip()
        return ai_message
    except Exception as e:
        return f"Error: {e}"
