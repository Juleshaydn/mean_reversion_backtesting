# app.py

import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine
import yfinance as yf
from db import create_tables
import matplotlib.pyplot as plt
from openai_chat import get_ai_response
from dotenv import load_dotenv
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller

# Load environment variables from .env file
load_dotenv()

# Initialize Streamlit page configuration
st.set_page_config(layout="wide", page_title="Stock Data App with AI Chat")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create two columns: Left for AI Chat, Right for Stock Analysis
left_col, right_col = st.columns([2, 2])

with left_col:
    st.markdown("## AI Chat Interface")

    # Apply custom CSS for text wrapping, fixed height with internal scrolling, and message styling
    st.markdown("""
    <style>
        .wrap-text {
            word-wrap: break-word;
            white-space: pre-wrap;
            background-color: rgba(245, 245, 245, 0.1);  /* Light grey background with opacity */
            border: 1px solid #ccc;  /* Grey border */
            border-radius: 8px;  /* Rounded corners */
            padding: 8px;  /* Padding inside the message box */
            margin-bottom: 10px;  /* Margin between messages */
            margin-left: 5px;
            margin-right: 5px;
        }
        .fixed-height-container {
            height: 90vh; /* Adjust based on your header/footer height */
            overflow-y: auto;
            padding: 5px;  /* Padding inside the container */
        }
    </style>
    """, unsafe_allow_html=True)

    # Use a container with fixed height and scrolling for the chat history
    st.markdown("<div class='fixed-height-container'>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role, content = message['role'], message['content']
            # Use markdown with a custom class for wrapping and styling
            st.markdown(f"<div class='wrap-text'><b>{role}:</b> {content}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    user_input = st.text_input("Type your message:", key="user_input")
    send_button = st.button("Send")

    if send_button and user_input.strip() != "":
        # Temporarily hold chat history to avoid multiple updates
        temp_chat_history = st.session_state.chat_history.copy()
        temp_chat_history.append({"role": "user", "content": user_input})

        # Display loading message
        with st.spinner('Waiting for AI response...'):
            ai_response = get_ai_response(temp_chat_history, user_input)
            temp_chat_history.append({"role": "assistant", "content": ai_response})

        # Update the session state chat history once
        st.session_state.chat_history = temp_chat_history

        # Refresh chat display
        chat_container.empty()
        st.markdown("<div class='fixed-height-container'>", unsafe_allow_html=True)
        with chat_container:
            for message in st.session_state.chat_history:
                role, content = message['role'], message['content']
                # Reapply custom class for wrapping and enhanced styling
                st.markdown(f"<div class='wrap-text'><b>{role}:</b> {content}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    # Check if DATABASE_URL is set
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        st.error("DATABASE_URL is not set. Please check your environment variables.")
        st.stop()

    # Initialize database tables and show connection status
    try:
        create_tables()
        st.success("Connected to the database successfully.")
    except Exception as e:
        st.error(f"Failed to connect to the database: {e}")

    # App Title
    st.title("Stock Mean-Reversion Analysis")

    # List of tickers for the dropdown
    tickers = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA', 'BRK-B']

    # Available periods and intervals in yfinance
    periods = [
        '1d', '5d', '7d', '1mo', '3mo', '6mo',
        '1y', '2y', '5y', '10y', 'ytd', 'max'
    ]

    intervals = [
        '1m', '2m', '5m', '15m', '30m', '60m', '90m',
        '1d', '5d', '1wk', '1mo', '3mo'
    ]

    # Select tickers, period, and interval (aligned horizontally)
    st.subheader("Select Parameters for Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_ticker1 = st.selectbox("First Ticker", tickers, key='ticker1')
    with col2:
        selected_ticker2 = st.selectbox("Second Ticker", tickers, key='ticker2')
    with col3:
        selected_period = st.selectbox("Period", periods, index=3, key='period_stock')  # Default to '1mo'
    with col4:
        selected_interval = st.selectbox("Interval", intervals, index=7, key='interval_stock')  # Default to '1d'

    # Function to validate period and interval combination
    def validate_period_interval(period, interval):
        invalid_combination = False
        error_message = ""
        if interval == '1m' and period not in ['1d', '5d', '7d']:
            invalid_combination = True
            error_message = "1 minute interval is only available for periods up to 7 days."
        elif interval == '2m' and period not in ['1d', '5d', '7d', '1mo', '3mo']:
            invalid_combination = True
            error_message = "2 minute interval is only available for periods up to 60 days."
        elif interval in ['5m', '15m', '30m', '60m', '90m'] and period not in ['1d', '5d', '7d', '1mo', '3mo', '6mo']:
            invalid_combination = True
            error_message = "Intraday intervals are only available for periods up to 60 days."
        elif interval in ['1d', '5d', '1wk', '1mo', '3mo'] and period == '1d':
            invalid_combination = True
            error_message = "Daily and higher intervals are not available for a period of 1 day."
        return invalid_combination, error_message

    # Function to import stock data
    def import_stock_data(symbols, period, interval):
        all_data = pd.DataFrame()
        for symbol in symbols:
            # Fetch historical data based on user selection
            ticker = yf.Ticker(symbol)
            try:
                hist = ticker.history(period=period, interval=interval)
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
                continue
            # Check if data is empty
            if hist.empty:
                st.error(f"No data found for {symbol} with the selected parameters.")
                continue
            # Reset index to turn date index into a column
            hist.reset_index(inplace=True)
            # Prepare the data
            hist['symbol'] = symbol
            # Ensure the 'Date' or 'Datetime' column exists
            date_column = 'Date' if 'Date' in hist.columns else 'Datetime'
            hist = hist[['symbol', date_column, 'Close']]
            hist.rename(columns={
                date_column: 'date',
                'Close': 'close',
            }, inplace=True)
            hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)  # Remove timezone
            # Append to all_data
            all_data = pd.concat([all_data, hist], ignore_index=True)
        return all_data

    # Add a button to import and analyze data
    if st.button("Run Analysis"):
        symbols = [selected_ticker1, selected_ticker2]
        # Validate period and interval combination
        invalid_combination, error_message = validate_period_interval(selected_period, selected_interval)
        if invalid_combination:
            st.error(f"Invalid combination of period and interval: {error_message}")
        else:
            try:
                df_stock_data = import_stock_data(symbols, selected_period, selected_interval)
                if df_stock_data.empty:
                    st.warning("No data found for the selected tickers. Please try different parameters.")
                else:
                    # Prepare data for analysis
                    df_stock_data.set_index('date', inplace=True)
                    # Pivot the data
                    df_pivot = df_stock_data.pivot_table(values='close', index='date', columns='symbol')
                    # Ensure data is aligned and drop NaN values
                    df_pivot = df_pivot.dropna(subset=[selected_ticker1, selected_ticker2])

                    # Ensure the selected symbols are in the data
                    if selected_ticker1 in df_pivot.columns and selected_ticker2 in df_pivot.columns:
                        # Calculate the spread
                        df_pivot['spread'] = df_pivot[selected_ticker1] - df_pivot[selected_ticker2]

                        # Calculate Z-score of the spread
                        df_pivot['z_score'] = (df_pivot['spread'] - df_pivot['spread'].mean()) / df_pivot['spread'].std()

                        # Perform cointegration test
                        coint_t, p_value, critical_values = coint(df_pivot[selected_ticker1], df_pivot[selected_ticker2])

                        # Generate buy and sell signals
                        df_pivot['buy_signal'] = np.where(df_pivot['z_score'] <= -1, df_pivot['spread'], np.nan)
                        df_pivot['sell_signal'] = np.where(df_pivot['z_score'] >= 1, df_pivot['spread'], np.nan)

                        # Create two columns for side-by-side plots
                        col_plot1, col_plot2 = st.columns(2)

                        with col_plot1:
                            st.write("### Z-score of the Spread")
                            fig_zscore, ax_zscore = plt.subplots(figsize=(10, 6))
                            ax_zscore.plot(df_pivot.index, df_pivot['z_score'], label='Z-score')
                            ax_zscore.axhline(0, color='black', linestyle='--')
                            ax_zscore.axhline(1, color='red', linestyle='--')
                            ax_zscore.axhline(-1, color='green', linestyle='--')
                            ax_zscore.set_xlabel('Date')
                            ax_zscore.set_ylabel('Z-score')
                            ax_zscore.set_title('Z-score of the Spread')
                            ax_zscore.legend()
                            st.pyplot(fig_zscore)

                        with col_plot2:
                            st.write("### Spread with Buy and Sell Signals")
                            fig_signal, ax_signal = plt.subplots(figsize=(10, 6))
                            ax_signal.plot(df_pivot.index, df_pivot['spread'], label='Spread')
                            ax_signal.plot(df_pivot.index, df_pivot['buy_signal'], '^', markersize=10, color='green', label='Buy Signal')
                            ax_signal.plot(df_pivot.index, df_pivot['sell_signal'], 'v', markersize=10, color='red', label='Sell Signal')
                            ax_signal.set_xlabel('Date')
                            ax_signal.set_ylabel('Price Difference')
                            ax_signal.set_title(f"Spread between {selected_ticker1} and {selected_ticker2} with Signals")
                            ax_signal.legend()
                            st.pyplot(fig_signal)
                        # Display the cointegration test results
                        st.write("### Cointegration Test Results")
                        st.write(f"t-statistic: {coint_t:.4f}")
                        st.write(f"p-value: {p_value:.4f}")
                        st.write(f"Critical Values:")
                        st.write(f"1%: {critical_values[0]:.4f}")
                        st.write(f"5%: {critical_values[1]:.4f}")
                        st.write(f"10%: {critical_values[2]:.4f}")

                        if p_value < 0.05:
                            st.success("The series are cointegrated.")
                        else:
                            st.warning("The series are not cointegrated.")

                    else:
                        st.error("Selected tickers are not in the data.")
            except Exception as e:
                st.error(f"Failed to fetch and analyze stock data: {e}")