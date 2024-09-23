import streamlit as st
import pandas as pd
from rebalance import calculate_allocation, rebalance_portfolio
from openai_integration import get_rebalancing_explanation
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key)  # This should print your API key in the console

# App Title
st.title("Portfolio Rebalancing App")

# Upload portfolio CSV
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"])

if uploaded_file is not None:
    # Read and display the portfolio
    portfolio = pd.read_csv(uploaded_file)
    st.write("Your Portfolio", portfolio)
    
    # Calculate and display allocation
    portfolio, total_value = calculate_allocation(portfolio)
    st.write("Current Allocation", portfolio)

    # Set target allocation for rebalancing
    target_allocation = {'Stocks': 0.60, 'Bonds': 0.40}
    
    # Rebalance the portfolio
    rebalanced_portfolio = rebalance_portfolio(portfolio, total_value, target_allocation)
    st.write("Rebalanced Portfolio", rebalanced_portfolio)
    
    # Get explanation from OpenAI
    explanation = get_rebalancing_explanation(rebalanced_portfolio, target_allocation)
    st.write("Explanation from OpenAI:", explanation)
