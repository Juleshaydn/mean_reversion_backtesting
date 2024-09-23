import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Explicitly set the API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_prompt(portfolio, target_allocation):
    template = """
    I have a portfolio with the following assets:
    {portfolio_details}
    I need to rebalance it to achieve the following target allocation:
    {target_allocation_details}.
    Please explain how much to buy or sell of each asset to achieve this.
    """
    portfolio_details = '\n'.join(f"{row['Asset']}: {row['Current Value']}" for index, row in portfolio.iterrows())
    target_allocation_details = ', '.join(f"{k}: {v * 100}%" for k, v in target_allocation.items())
    return template.format(portfolio_details=portfolio_details, target_allocation_details=target_allocation_details)

def get_rebalancing_explanation(portfolio, target_allocation):
    prompt = generate_prompt(portfolio, target_allocation)
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial advisor explaining how to rebalance a portfolio."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']
