# app.py

import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine
import yfinance as yf
from db import create_tables
import seaborn as sns
import matplotlib.pyplot as plt
from openai_chat import get_ai_response
from dotenv import load_dotenv

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

# The rest of your right_col and stock data handling remains unchanged

with right_col:
    # The stock data section remains unchanged
    ...

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
    st.title("Stock Data App")
    
    # List of tickers for the dropdown
    tickers = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA']
    
    # Select period and interval for correlation matrix
    st.header("Correlation Matrix of Selected Stocks")
    periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    intervals = ['1d', '1wk', '1mo']
    
    selected_period_corr = st.selectbox("Select a period for correlation", periods, index=2)  # Default to '1mo'
    selected_interval_corr = st.selectbox("Select an interval for correlation", intervals, index=0)  # Default to '1d'
    
    # Button to fetch and display correlation matrix
    if st.button("Display Correlation Matrix"):
        # Function to fetch data for correlation matrix
        def fetch_data_for_correlation(tickers, period, interval):
            engine = create_engine(database_url)
            all_data = pd.DataFrame()
            for symbol in tickers:
                # Check if data exists in the database
                query = """
                SELECT * FROM historical_stock_data
                WHERE symbol = %s AND period = %s AND interval = %s
                ORDER BY date ASC
                """
                params = (symbol, period, interval)
                df = pd.read_sql_query(query, engine, params=params)
    
                if df.empty:
                    # Fetch data from yfinance and insert into the database
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period=period, interval=interval)
                    if hist.empty:
                        st.error(f"No data found for {symbol} with the selected parameters.")
                        continue
                    hist.reset_index(inplace=True)
                    hist['symbol'] = symbol
                    hist['period'] = period
                    hist['interval'] = interval
                    date_column = 'Date' if 'Date' in hist.columns else 'Datetime'
                    hist.rename(columns={date_column: 'date', 'Close': 'close'}, inplace=True)
                    hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)  # Remove timezone
                    hist = hist[['symbol', 'date', 'period', 'interval', 'close']]
                    # Insert data into the database
                    try:
                        hist.to_sql('historical_stock_data', engine, if_exists='append', index=False, method='multi')
                        st.success(f"{symbol} stock data imported successfully for correlation.")
                        df = hist
                    except Exception as e:
                        st.error(f"Failed to insert data for {symbol} into the database: {e}")
                        continue
                else:
                    # Process data fetched from the database
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone
    
                # Prepare df for joining
                df = df[['date', 'close']].copy()
                df.set_index('date', inplace=True)
                df.rename(columns={'close': symbol}, inplace=True)
                df.index = df.index.tz_localize(None)  # Ensure tz-naive index
    
                # Join data
                if all_data.empty:
                    all_data = df
                else:
                    all_data.index = all_data.index.tz_localize(None)  # Ensure tz-naive index
                    all_data = all_data.join(df, how='outer')
            return all_data
    
        # Fetch data and compute correlation matrix
        stock_data = fetch_data_for_correlation(tickers, selected_period_corr, selected_interval_corr)
        if not stock_data.empty:
            stock_data = stock_data.dropna()
            corr_matrix = stock_data.corr()
            st.write("### Correlation Matrix")
    
            # Display the correlation matrix as a heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No data available to display correlation matrix. Please try different parameters.")
    
    st.write("---")  # Separator
    
    # Select tickers from dropdowns
    selected_ticker1 = st.selectbox("Select the first stock ticker", tickers, key='ticker1')
    selected_ticker2 = st.selectbox("Select the second stock ticker", tickers, key='ticker2')
    
    # Select period and interval for stock data
    selected_period = st.selectbox("Select a period for stock data", periods, index=2, key='period_stock')  # Default to '1mo'
    selected_interval = st.selectbox("Select an interval for stock data", intervals, index=0, key='interval_stock')  # Default to '1d'
    
    # Function to import stock data
    def import_stock_data(symbols, period, interval):
        engine = create_engine(database_url)
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
            # Prepare the data for insertion
            hist['symbol'] = symbol
            hist['period'] = period
            hist['interval'] = interval
            # Ensure the 'Date' or 'Datetime' column exists
            date_column = 'Date' if 'Date' in hist.columns else 'Datetime'
            hist = hist[['symbol', date_column, 'period', 'interval', 'Open', 'High', 'Low', 'Close', 'Volume']]
            hist.rename(columns={
                date_column: 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)  # Remove timezone
            # Insert data into the database
            try:
                hist.to_sql('historical_stock_data', engine, if_exists='append', index=False, method='multi')
                st.success(f"{symbol} stock data imported successfully.")
            except Exception as e:
                st.error(f"Failed to insert data for {symbol} into the database: {e}")
    
    # Add a button to import selected stock data
    if st.button("Import Selected Stock Data"):
        symbols = [selected_ticker1, selected_ticker2]
        try:
            import_stock_data(symbols, selected_period, selected_interval)
        except Exception as e:
            st.error(f"Failed to import stock data: {e}")
    
    # Fetch and display selected stock data from the database
    try:
        engine = create_engine(database_url)
        query = """
        SELECT * FROM historical_stock_data
        WHERE symbol IN (%s, %s) AND period = %s AND interval = %s
        ORDER BY date ASC
        """
        params = (selected_ticker1, selected_ticker2, selected_period, selected_interval)
        df_stock_data = pd.read_sql_query(query, engine, params=params)
        if df_stock_data.empty:
            st.warning("No data found for the selected tickers. Please import data first.")
        else:
            st.write("### Stock Data Stored in the Database")
            st.dataframe(df_stock_data)
            # Prepare data for plotting
            df_stock_data['date'] = pd.to_datetime(df_stock_data['date']).dt.tz_localize(None)
            df_stock_data.set_index('date', inplace=True)
            # Pivot the data
            df_pivot = df_stock_data.pivot_table(values='close', index='date', columns='symbol')
            # Plot the data
            st.write("### Stock Closing Price Chart")
            st.line_chart(df_pivot)
    except Exception as e:
        st.error(f"Failed to fetch stock data from the database: {e}")
