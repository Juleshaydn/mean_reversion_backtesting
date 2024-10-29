# app.py

import streamlit as st
import pandas as pd
import os
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import numpy as np
from statsmodels.tsa.stattools import coint
from openai_chat import get_ai_response
from db import create_tables, get_db_connection, clear_signals_table  # Import clear_signals_table

# Load environment variables from .env file
load_dotenv()

# Initialize Streamlit page configuration
st.set_page_config(layout="wide", page_title="Stock Data App with AI Chat")

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

if 'df_pivot' not in st.session_state:
    st.session_state.df_pivot = None

if 'coint_results' not in st.session_state:
    st.session_state.coint_results = None

# Initialize session state for selected inputs
if 'selected_ticker1' not in st.session_state:
    st.session_state.selected_ticker1 = 'AAPL'

if 'selected_ticker2' not in st.session_state:
    st.session_state.selected_ticker2 = 'AMZN'

if 'selected_period' not in st.session_state:
    st.session_state.selected_period = '1mo'

if 'selected_interval' not in st.session_state:
    st.session_state.selected_interval = '1d'

# Define periods and intervals at the top
periods = [
    '1d', '5d', '7d', '1mo', '3mo', '6mo',
    '1y', '2y', '5y', '10y', 'ytd', 'max'
]

intervals = [
    '1m', '2m', '5m', '15m', '30m', '60m', '90m',
    '1d', '5d', '1wk', '1mo', '3mo'
]

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
    send_button = st.button("Send", key='send_button')

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
    except Exception as e:
        st.error(f"Failed to connect to the database: {e}")

    # App Title
    st.title("Stock Mean-Reversion Analysis")

    # List of tickers for the dropdown
    tickers = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA', 'BRK-B']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_ticker1 = st.selectbox(
            "First Ticker", tickers, index=tickers.index(st.session_state.selected_ticker1), key='ticker1')
    with col2:
        selected_ticker2 = st.selectbox(
            "Second Ticker", tickers, index=tickers.index(st.session_state.selected_ticker2), key='ticker2')
    with col3:
        selected_period = st.selectbox(
            "Period", periods, index=periods.index(st.session_state.selected_period), key='period_stock')
    with col4:
        selected_interval = st.selectbox(
            "Interval", intervals, index=intervals.index(st.session_state.selected_interval), key='interval_stock')

    # Update session state with current selections
    st.session_state.selected_ticker1 = selected_ticker1
    st.session_state.selected_ticker2 = selected_ticker2
    st.session_state.selected_period = selected_period
    st.session_state.selected_interval = selected_interval

    # Validate that the selected tickers are different
    if selected_ticker1 == selected_ticker2:
        st.error("Please select two different tickers.")
        st.stop()

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
            # Ensure the date column exists
            if date_column not in hist.columns:
                st.error(f"No date column found for {symbol}.")
                continue
            hist = hist[['symbol', date_column, 'Close']]
            hist.rename(columns={
                date_column: 'date',
                'Close': 'close',
            }, inplace=True)
            hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)  # Remove timezone
            # Append to all_data
            all_data = pd.concat([all_data, hist], ignore_index=True)
        return all_data

    # Function to calculate profits
    def calculate_profits(signals_df):
        signals_df = signals_df.sort_values('date')
        signals_df['profit'] = np.nan
        position = None
        entry_spread = None
        entry_date = None

        for index, row in signals_df.iterrows():
            signal_type = row['signal_type']
            date = row['date']
            spread = row['spread']

            if position is None:
                # No position, open position
                position = signal_type
                entry_spread = spread
                entry_date = date
                signals_df.loc[index, 'profit'] = 0  # Profit is zero at entry
            else:
                # Close position
                if position != signal_type:
                    # Positions are opposite, calculate profit
                    if position == 'buy':
                        profit = spread - entry_spread
                    else:
                        profit = entry_spread - spread
                    signals_df.loc[index, 'profit'] = profit
                    # Reset position
                    position = None
                    entry_spread = None
                    entry_date = None
                else:
                    # Consecutive same signals, skip
                    signals_df.loc[index, 'profit'] = 0  # No profit change
        return signals_df

    # Function to insert signals into the database
    def insert_signals_to_db(signals_df):
        conn = get_db_connection()
        cursor = conn.cursor()
        for index, row in signals_df.iterrows():
            cursor.execute("""
                INSERT INTO signals (date, ticker1, ticker2, signal_type, spread, profit)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (row['date'], row['symbol1'], row['symbol2'], row['signal_type'], row['spread'], row['profit']))
        conn.commit()
        cursor.close()
        conn.close()

    # Function to calculate expected return based on trading value
    def calculate_expected_return(signals_df, trading_value):
        # Calculate the return for each trade based on the trading value and profit/loss from each trade
        signals_df['trade_return'] = signals_df['profit'] * trading_value
        total_return = signals_df['trade_return'].sum()
        return total_return
    
    # Arrange inputs and button horizontally in columns
    col_bollinger, col_moving_avg, col_trading_val, col_button = st.columns(4)

    with col_bollinger:
        bollinger_multiplier = st.number_input("Bollinger Band Multiplier", min_value=0.5, max_value=5.0, value=1.0, step=0.1, key="bollinger_multiplier")

    with col_moving_avg:
        moving_average_window = st.number_input("Moving Average Window", min_value=5, max_value=100, value=20, step=1, key="moving_average_window")

    with col_trading_val:
        trading_value = st.number_input("Trading Value per Trade (£)", min_value=100.0, value=1000.0, step=50.0, key="trading_value")

    with col_button:
        run_analysis = st.button("Run Analysis", key='run_analysis_button')

    # Add the logic to run analysis only if the button is clicked
    if run_analysis:
        symbols = [selected_ticker1, selected_ticker2]
        # Validate period and interval combination
        invalid_combination, error_message = validate_period_interval(selected_period, selected_interval)
        if invalid_combination:
            st.error(f"Invalid combination of period and interval: {error_message}")
        else:
            try:
                # Fetch the data
                df_stock_data = import_stock_data(symbols, selected_period, selected_interval)
                if df_stock_data.empty:
                    st.warning("No data found for the selected tickers. Please try different parameters.")
                else:
                    # Prepare data for analysis
                    df_stock_data.set_index('date', inplace=True)
                    df_pivot = df_stock_data.pivot_table(values='close', index='date', columns='symbol')
                    df_pivot = df_pivot.dropna(subset=[selected_ticker1, selected_ticker2])

                    # Calculate the spread
                    df_pivot['spread'] = df_pivot[selected_ticker1] - df_pivot[selected_ticker2]

                    # Calculate the Moving Average and standard deviation for Bollinger Bands
                    df_pivot['moving_avg'] = df_pivot['spread'].rolling(window=int(moving_average_window)).mean()
                    df_pivot['std_dev'] = df_pivot['spread'].rolling(window=int(moving_average_window)).std()
                    
                    # Calculate Bollinger Bands using user-defined multiplier
                    df_pivot['upper_band'] = df_pivot['moving_avg'] + (df_pivot['std_dev'] * bollinger_multiplier)
                    df_pivot['lower_band'] = df_pivot['moving_avg'] - (df_pivot['std_dev'] * bollinger_multiplier)

                    # Calculate Z-score of the spread based on the Moving Average and standard deviation
                    df_pivot['z_score'] = (df_pivot['spread'] - df_pivot['moving_avg']) / df_pivot['std_dev']

                    # Perform cointegration test
                    coint_t, p_value, critical_values = coint(df_pivot[selected_ticker1], df_pivot[selected_ticker2])

                    # Generate buy and sell signals based on Bollinger Bands
                    df_pivot['buy_signal'] = np.where(df_pivot['spread'] < df_pivot['lower_band'], df_pivot['spread'], np.nan)
                    df_pivot['sell_signal'] = np.where(df_pivot['spread'] > df_pivot['upper_band'], df_pivot['spread'], np.nan)

                    # Add symbol columns
                    df_pivot['symbol1'] = selected_ticker1
                    df_pivot['symbol2'] = selected_ticker2

                    # Clear existing data and insert signals into the database
                    clear_signals_table()  # Clear the signals table before inserting new data
                    df_pivot.reset_index(inplace=True)
                    
                    # Prepare signals DataFrame
                    buy_signals = df_pivot[df_pivot['buy_signal'].notnull()].copy()
                    buy_signals['signal_type'] = 'buy'
                    buy_signals['signal_value'] = buy_signals['buy_signal']

                    sell_signals = df_pivot[df_pivot['sell_signal'].notnull()].copy()
                    sell_signals['signal_type'] = 'sell'
                    sell_signals['signal_value'] = sell_signals['sell_signal']

                    signals_df = pd.concat([buy_signals, sell_signals]).sort_values('date')
                    signals_df = signals_df[['date', 'symbol1', 'symbol2', 'spread', 'signal_type', 'signal_value']]
                    signals_df.reset_index(drop=True, inplace=True)

                    # Calculate profits
                    signals_df = calculate_profits(signals_df)
                    insert_signals_to_db(signals_df)

                    # Calculate expected return based on trading value
                    expected_return = calculate_expected_return(signals_df, trading_value)

                    # Save to session for displaying charts and results
                    st.session_state.df_pivot = df_pivot.copy()
                    st.session_state.coint_results = (coint_t, p_value, critical_values)
            except Exception as e:
                st.error(f"Failed to fetch and analyze stock data: {e}")

        # Display Charts and Results
        if st.session_state.get('df_pivot') is not None:
            df_pivot = st.session_state.df_pivot.copy()
            coint_t, p_value, critical_values = st.session_state.coint_results

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

            if p_value < 0.05:
                st.success("The series are cointegrated.")
            else:
                st.warning("The series are not cointegrated.")
                
            # Display the cointegration test results
            # st.write("### Cointegration Test Results")
            # st.write(f"Expected Return: £{expected_return:.2f}")
            # st.write(f"t-statistic: {coint_t:.4f}")
            # st.write(f"p-value: {p_value:.4f}")
            # st.write("Critical Values:")
            # st.write(f"1%: {critical_values[0]:.4f}")
            # st.write(f"5%: {critical_values[1]:.4f}")
            # st.write(f"10%: {critical_values[2]:.4f}")

            # OpenAI Response
            st.write("AI Analysis")

            # Define a prompt template with context for OpenAI
            prompt_template = f"""
            Analyze the following strategy parameters based on the user's inputs and calculations:
            - Selected tickers: {selected_ticker1} and {selected_ticker2}
            - Period: {selected_period}, Interval: {selected_interval}
            - Bollinger Band Multiplier: {bollinger_multiplier}
            - Moving Average Window: {moving_average_window}
            - Trading Value per Trade: £{trading_value}
            - Cointegration Test Results:
            - t-statistic: {coint_t:.4f}
            - p-value: {p_value:.4f}
            - Critical Values: 1%: {critical_values[0]:.4f}, 5%: {critical_values[1]:.4f}, 10%: {critical_values[2]:.4f}
            - Expected Return: £{expected_return:.2f}

            Please provide a simple analysis of this trading strategy's viability, strengths, and potential risks, make sure to include the expected return from this strategy first.
            """

            # Get AI response
            ai_response = get_ai_response(st.session_state.chat_history, prompt_template)

            # Display AI response
            st.write(ai_response)

            with st.expander("Data Stored in the Database"):
                # Fetch data from the database
                try:
                    conn = get_db_connection()
                    query = """
                        SELECT date, ticker1, ticker2, signal_type, spread, profit
                        FROM signals
                        ORDER BY date DESC;
                    """
                    df_signals_db = pd.read_sql(query, conn)
                    conn.close()

                    # Display the data
                    st.dataframe(df_signals_db)
                except Exception as e:
                    st.error(f"Failed to fetch data from the database: {e}")
        else:
            st.info("Run analysis to display charts and results.")
