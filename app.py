import streamlit as st
import pandas as pd
import os
import yfinance as yf
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import numpy as np
from statsmodels.tsa.stattools import coint
from openai_chat import get_ai_response
from db import create_tables, get_db_connection, clear_signals_table

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit page
st.set_page_config(layout="wide", page_title="Stock Data App with AI Chat")

# -------------------- Initialise Session State --------------------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'coint_results' not in st.session_state:
    st.session_state.coint_results = None

if 'df_pivot' not in st.session_state:
    st.session_state.df_pivot = None

if 'expected_return' not in st.session_state:
    st.session_state.expected_return = None

# Default selections for the user interface
if 'selected_ticker1' not in st.session_state:
    st.session_state.selected_ticker1 = 'AAPL'
if 'selected_ticker2' not in st.session_state:
    st.session_state.selected_ticker2 = 'AMZN'
if 'selected_period' not in st.session_state:
    st.session_state.selected_period = '1mo'
if 'selected_interval' not in st.session_state:
    st.session_state.selected_interval = '1d'
if 'bollinger_multiplier' not in st.session_state:
    st.session_state.bollinger_multiplier = 1.0
if 'moving_average_window' not in st.session_state:
    st.session_state.moving_average_window = 20
if 'trading_value' not in st.session_state:
    st.session_state.trading_value = 1000.0

# Periods and intervals
periods = [
    '1d', '5d', '7d', '1mo', '3mo', '6mo',
    '1y', '2y', '5y', '10y', 'ytd', 'max'
]

intervals = [
    '1m', '2m', '5m', '15m', '30m', '60m', '90m',
    '1d', '5d', '1wk', '1mo', '3mo'
]

# -------------------- Layout: Two Columns --------------------
col_left, col_right = st.columns(2)

# =====================================================================
#                     LEFT COLUMN: MAIN ANALYSIS
# =====================================================================
with col_left:
    # Check database connection
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        st.error("DATABASE_URL is not set. Please check your environment variables.")
        st.stop()

    # Create database tables if they don't exist
    try:
        create_tables()
    except Exception as e:
        st.error(f"Failed to connect or create tables in the database: {e}")

    # Title
    st.title("Stock Mean-Reversion Analysis")

    # Ticker, Period, Interval selection
    tickers = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'TSLA', 'BRK-B','BTC-USD', 'ETH-USD']
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_ticker1 = st.selectbox(
            "First Ticker", tickers, index=tickers.index(st.session_state.selected_ticker1)
        )
    with col2:
        selected_ticker2 = st.selectbox(
            "Second Ticker", tickers, index=tickers.index(st.session_state.selected_ticker2)
        )
    with col3:
        selected_period = st.selectbox(
            "Period", periods, index=periods.index(st.session_state.selected_period)
        )
    with col4:
        selected_interval = st.selectbox(
            "Interval", intervals, index=intervals.index(st.session_state.selected_interval)
        )

    # Update session state
    st.session_state.selected_ticker1 = selected_ticker1
    st.session_state.selected_ticker2 = selected_ticker2
    st.session_state.selected_period = selected_period
    st.session_state.selected_interval = selected_interval

    # Error out if both tickers are the same
    if selected_ticker1 == selected_ticker2:
        st.error("Please select two different tickers.")
        st.stop()

    # -------------------- Helper Functions --------------------
    def validate_period_interval(period, interval):
        """
        Ensures the chosen period and interval combination is valid on yfinance.
        """
        invalid = False
        msg = ""
        if interval == '1m' and period not in ['1d', '5d', '7d']:
            invalid = True
            msg = "1 minute interval is only available for periods up to 7 days."
        elif interval == '2m' and period not in ['1d', '5d', '7d', '1mo', '3mo']:
            invalid = True
            msg = "2 minute interval is only available for periods up to 60 days."
        elif interval in ['5m', '15m', '30m', '60m', '90m'] and period not in ['1d', '5d', '7d', '1mo', '3mo', '6mo']:
            invalid = True
            msg = "Intraday intervals are only available for periods up to 60 days."
        elif interval in ['1d', '5d', '1wk', '1mo', '3mo'] and period == '1d':
            invalid = True
            msg = "Daily and higher intervals are not available for a period of just 1 day."
        return invalid, msg

    def import_stock_data(symbols, period, interval):
        """
        Fetches historical data using yfinance for the given symbols, period, and interval.
        Returns a single DataFrame with columns: [symbol, date, close].
        """
        all_data = pd.DataFrame()
        for sym in symbols:
            ticker = yf.Ticker(sym)
            try:
                hist = ticker.history(period=period, interval=interval)
            except Exception as exc:
                st.error(f"Error fetching data for {sym}: {exc}")
                continue

            if hist.empty:
                st.error(f"No data found for {sym} with the selected parameters.")
                continue

            hist.reset_index(inplace=True)
            hist['symbol'] = sym
            # Find the date column (Date or Datetime)
            date_col = 'Date' if 'Date' in hist.columns else 'Datetime'
            if date_col not in hist.columns:
                st.error(f"No valid date column found for {sym}.")
                continue

            hist.rename(columns={date_col: 'date', 'Close': 'close'}, inplace=True)
            hist['date'] = pd.to_datetime(hist['date']).dt.tz_localize(None)
            hist = hist[['symbol', 'date', 'close']]
            all_data = pd.concat([all_data, hist], ignore_index=True)

        return all_data

    def insert_signals_to_db(signals_df):
        """
        Inserts each row of the signals DataFrame into the signals table.
        """
        conn = get_db_connection()
        cursor = conn.cursor()
        for _, row in signals_df.iterrows():
            cursor.execute("""
                INSERT INTO signals (date, ticker1, ticker2, signal_type, spread, profit)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (
                row['date'],
                row['symbol1'],
                row['symbol2'],
                row['signal_type'],
                row['spread'],
                row['profit']
            ))
        conn.commit()
        cursor.close()
        conn.close()

    # ----------- **Key Improvement**: More Accurate Pairs-Trading P&L -----------
    def calculate_profits(signals_df, df_pivot, t1, t2, capital):
        """
        Given the buy/sell signals (in signals_df) and the pivoted price data (df_pivot),
        computes profit/loss for each entry -> exit round trip under a standard pairs approach:
          - On a "buy" signal, we BUY Ticker1 (half of capital) and SHORT Ticker2 (other half).
          - On a "sell" signal, we SHORT Ticker1 (half of capital) and BUY Ticker2 (other half).
        We only close a position once we see an opposite signal. If we end with a position
        still open, we close it on the final row of signals_df.
        
        Returns a new DataFrame with the 'profit' column filled in for each exit.
        """
        signals_df = signals_df.sort_values('date').reset_index(drop=True)
        signals_df['profit'] = 0.0
        position = None     # "buy" or "sell"
        entry_price1 = 0.0
        entry_price2 = 0.0
        units1 = 0.0
        units2 = 0.0
        capital_per_side = capital / 2.0

        for i in range(len(signals_df)):
            date = signals_df.loc[i, 'date']
            signal_type = signals_df.loc[i, 'signal_type']

            # Get the Ticker1 and Ticker2 prices on this signal date
            # (df_pivot has columns = [date, T1, T2, spread, ...])
            row_prices = df_pivot[df_pivot['date'] == date]
            if row_prices.empty:
                # No matching date in pivot => no price data => no P&L
                signals_df.loc[i, 'profit'] = 0.0
                continue

            price1 = row_prices[t1].values[0]
            price2 = row_prices[t2].values[0]

            if position is None:
                # No open position => open one now
                position = signal_type
                entry_price1 = price1
                entry_price2 = price2

                # For either buy or sell, we allocate half capital to T1, half to T2
                if signal_type == 'buy':
                    # Long T1, short T2
                    units1 = capital_per_side / entry_price1
                    units2 = capital_per_side / entry_price2
                else:
                    # "sell": short T1, long T2
                    units1 = capital_per_side / entry_price1
                    units2 = capital_per_side / entry_price2

                # Profit is 0 on the day we open
                signals_df.loc[i, 'profit'] = 0.0

            else:
                # We already have a position open
                if position != signal_type:
                    # We have a new signal that is the opposite => close the old position
                    if position == 'buy':
                        # We were long T1 + short T2
                        # initial_value = (units1 * entry_price1) - (units2 * entry_price2)
                        # final_value   = (units1 * price1) - (units2 * price2)
                        final_val = (units1 * price1) - (units2 * price2)
                        init_val  = (units1 * entry_price1) - (units2 * entry_price2)
                        trade_pnl = final_val - init_val
                    else:
                        # position == 'sell'
                        # We were short T1 + long T2
                        # initial_value = - (units1 * entry_price1) + (units2 * entry_price2)
                        # final_value   = - (units1 * price1) + (units2 * price2)
                        final_val = - (units1 * price1) + (units2 * price2)
                        init_val  = - (units1 * entry_price1) + (units2 * entry_price2)
                        trade_pnl = final_val - init_val

                    # Record the profit on the closing day
                    signals_df.loc[i, 'profit'] = trade_pnl

                    # Now we open a *new* position on the same day,
                    # because we have a brand new opposite signal
                    position = signal_type
                    entry_price1 = price1
                    entry_price2 = price2
                    if position == 'buy':
                        units1 = capital_per_side / entry_price1
                        units2 = capital_per_side / entry_price2
                    else:
                        units1 = capital_per_side / entry_price1
                        units2 = capital_per_side / entry_price2

                else:
                    # Same signal => remain in the same position, no realised P&L
                    signals_df.loc[i, 'profit'] = 0.0

        # If we still have a position open at the end, close it on the final row
        if position is not None and len(signals_df) > 0:
            i = len(signals_df) - 1  # final row
            date = signals_df.loc[i, 'date']
            row_prices = df_pivot[df_pivot['date'] == date]
            if not row_prices.empty:
                price1 = row_prices[t1].values[0]
                price2 = row_prices[t2].values[0]
                if position == 'buy':
                    final_val = (units1 * price1) - (units2 * price2)
                    init_val  = (units1 * entry_price1) - (units2 * entry_price2)
                else:
                    final_val = - (units1 * price1) + (units2 * price2)
                    init_val  = - (units1 * entry_price1) + (units2 * entry_price2)
                trade_pnl = final_val - init_val
                signals_df.loc[i, 'profit'] += trade_pnl  # Add to any existing P&L
            # Clear the position
            position = None

        return signals_df

    def calculate_expected_return(signals_df):
        """
        Simply sums the 'profit' column to see the total P&L from all closed trades.
        """
        return signals_df['profit'].sum()

    # --------------- UI Inputs for Bollinger & Strategy ---------------
    col_bollinger, col_moving_avg, col_trading_val, col_button = st.columns(4)
    with col_bollinger:
        bollinger_multiplier = st.number_input(
            "Bollinger Band Multiplier", 
            min_value=0.5, max_value=5.0, 
            value=st.session_state.bollinger_multiplier, 
            step=0.1
        )
    with col_moving_avg:
        moving_average_window = st.number_input(
            "Moving Average Window", 
            min_value=5, max_value=100, 
            value=st.session_state.moving_average_window, 
            step=1
        )
    with col_trading_val:
        trading_value = st.number_input(
            "Trading Value per Trade (£)", 
            min_value=100.0, 
            value=st.session_state.trading_value, 
            step=50.0
        )
    with col_button:
        run_analysis = st.button("Run Analysis")

    # Update session state
    st.session_state.bollinger_multiplier = bollinger_multiplier
    st.session_state.moving_average_window = moving_average_window
    st.session_state.trading_value = trading_value

    # -------------------- Main Analysis Logic --------------------
    if run_analysis:
        # Validate period/interval
        invalid, msg = validate_period_interval(selected_period, selected_interval)
        if invalid:
            st.error(f"Invalid combination of period/interval: {msg}")
        else:
            # Attempt data fetch
            try:
                symbols = [selected_ticker1, selected_ticker2]
                df_stock_data = import_stock_data(symbols, selected_period, selected_interval)
                if df_stock_data.empty:
                    st.warning("No data returned. Try different parameters.")
                else:
                    # Reshape data
                    df_stock_data.set_index('date', inplace=True)
                    df_pivot = df_stock_data.pivot_table(
                        values='close', 
                        index='date', 
                        columns='symbol'
                    ).dropna(subset=[selected_ticker1, selected_ticker2])

                    # Compute spread
                    df_pivot['spread'] = df_pivot[selected_ticker1] - df_pivot[selected_ticker2]

                    # Rolling stats for Bollinger
                    df_pivot['moving_avg'] = df_pivot['spread'].rolling(window=int(moving_average_window)).mean()
                    df_pivot['std_dev'] = df_pivot['spread'].rolling(window=int(moving_average_window)).std()

                    df_pivot['upper_band'] = df_pivot['moving_avg'] + (df_pivot['std_dev'] * bollinger_multiplier)
                    df_pivot['lower_band'] = df_pivot['moving_avg'] - (df_pivot['std_dev'] * bollinger_multiplier)
                    df_pivot['z_score'] = (
                        df_pivot['spread'] - df_pivot['moving_avg']
                    ) / df_pivot['std_dev']

                    # Cointegration test
                    coint_t, p_value, critical_values = coint(
                        df_pivot[selected_ticker1], 
                        df_pivot[selected_ticker2]
                    )

                    # Generate signals:
                    # "buy" when spread < lower_band, "sell" when spread > upper_band
                    df_pivot['buy_signal']  = np.where(df_pivot['spread'] < df_pivot['lower_band'],
                                                       df_pivot['spread'], np.nan)
                    df_pivot['sell_signal'] = np.where(df_pivot['spread'] > df_pivot['upper_band'],
                                                       df_pivot['spread'], np.nan)

                    # For reference in DB
                    df_pivot['symbol1'] = selected_ticker1
                    df_pivot['symbol2'] = selected_ticker2

                    # Clear old signals from DB
                    clear_signals_table()

                    # Prepare signals DataFrame
                    df_pivot.reset_index(inplace=True)  # date is now a column
                    buy_signals = df_pivot[df_pivot['buy_signal'].notnull()].copy()
                    buy_signals['signal_type'] = 'buy'
                    buy_signals['signal_value'] = buy_signals['buy_signal']

                    sell_signals = df_pivot[df_pivot['sell_signal'].notnull()].copy()
                    sell_signals['signal_type'] = 'sell'
                    sell_signals['signal_value'] = sell_signals['sell_signal']

                    signals_df = pd.concat([buy_signals, sell_signals], ignore_index=True)
                    signals_df.sort_values('date', inplace=True)
                    signals_df = signals_df[['date','symbol1','symbol2','spread','signal_type','signal_value']]
                    signals_df.reset_index(drop=True, inplace=True)

                    # --------- Calculate P&L with improved pairs logic ---------
                    signals_df = calculate_profits(
                        signals_df,
                        df_pivot, 
                        selected_ticker1, 
                        selected_ticker2, 
                        trading_value
                    )

                    # Insert into DB
                    insert_signals_to_db(signals_df)

                    # Sum the realised profits
                    total_pnl = calculate_expected_return(signals_df)
                    st.session_state.df_pivot = df_pivot.copy()
                    st.session_state.coint_results = (coint_t, p_value, critical_values)
                    st.session_state.expected_return = total_pnl

                    # -------------------- Plotting --------------------
                    col_plot1, col_plot2 = st.columns(2)
                    with col_plot1:
                        st.write("### Z-score of the Spread")
                        fig_zscore, ax_zscore = plt.subplots(figsize=(8, 5))
                        ax_zscore.plot(df_pivot['date'], df_pivot['z_score'], label='Z-score')
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
                        fig_signal, ax_signal = plt.subplots(figsize=(8, 5))
                        ax_signal.plot(df_pivot['date'], df_pivot['spread'], label='Spread')
                        ax_signal.plot(df_pivot['date'], df_pivot['buy_signal'], '^', 
                                       markersize=8, color='green', label='Buy Signal')
                        ax_signal.plot(df_pivot['date'], df_pivot['sell_signal'], 'v', 
                                       markersize=8, color='red', label='Sell Signal')
                        ax_signal.set_xlabel('Date')
                        ax_signal.set_ylabel('Price Difference')
                        ax_signal.set_title(f"Spread {selected_ticker1} - {selected_ticker2} with Signals")
                        ax_signal.legend()
                        st.pyplot(fig_signal)

                    # Cointegration feedback
                    if p_value < 0.05:
                        st.success("The series are cointegrated.")
                    else:
                        st.warning("The series are not cointegrated.")

                    st.write(f"#### Total Realised P&L from this Strategy: £{total_pnl:,.2f}")

            except Exception as e:
                st.error(f"Failed to fetch or analyse data: {e}")

# =====================================================================
#                     RIGHT COLUMN: AI + DATABASE
# =====================================================================
with col_right:
    st.write("### AI Analysis")
    if st.session_state.coint_results is not None and st.session_state.df_pivot is not None:
        coint_t, p_value, critical_values = st.session_state.coint_results
        expected_return = st.session_state.expected_return or 0.0
        boll_mult = st.session_state.bollinger_multiplier
        ma_window = st.session_state.moving_average_window
        trade_val = st.session_state.trading_value
        t1 = st.session_state.selected_ticker1
        t2 = st.session_state.selected_ticker2
        per = st.session_state.selected_period
        interval = st.session_state.selected_interval

        # Prompt to OpenAI or another model
        with st.spinner('Analyzing strategy parameters...'):
            prompt_template = f"""
Analyse the following strategy calculations:
- Selected tickers: {t1} and {t2}
- Period: {per}, Interval: {interval}
- Bollinger Band Multiplier: {boll_mult}
- Moving Average Window: {ma_window}
- Trading Value per Trade: £{trade_val}
- Cointegration Test Results:
  - t-statistic: {coint_t:.4f}
  - p-value: {p_value:.4f}
  - Critical Values: 
      1%: {critical_values[0]:.4f}, 
      5%: {critical_values[1]:.4f}, 
     10%: {critical_values[2]:.4f}
- Total Realised P&L: £{expected_return:.2f}

Write a short summary of this trading strategy's viability showing the P&L first, then the cointegration metrics in a small table.
"""
            ai_response = get_ai_response(st.session_state.chat_history, prompt_template)
        st.write(ai_response)

        # Expandable database content
        with st.expander("Data Stored in the Database"):
            try:
                conn = get_db_connection()
                query = """
                    SELECT date, ticker1, ticker2, signal_type, spread, profit
                    FROM signals
                    ORDER BY date DESC;
                """
                df_signals_db = pd.read_sql(query, conn)
                conn.close()
                st.dataframe(df_signals_db)
            except Exception as e:
                st.error(f"Failed to fetch data from the database: {e}")
    else:
        st.info("Run the analysis on the right to see AI insights and stored trade data.")
