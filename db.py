import psycopg2
import os

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create the historical_stock_data table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_stock_data (
        symbol VARCHAR(10),
        date TIMESTAMP,
        period VARCHAR(10),
        interval VARCHAR(10),
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume BIGINT,
        PRIMARY KEY (symbol, date, period, interval)
    );
    """)

    # Create the signals table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS signals (
        id SERIAL PRIMARY KEY,
        date TIMESTAMP,
        ticker1 VARCHAR(10),
        ticker2 VARCHAR(10),
        signal_type VARCHAR(4),  -- 'buy' or 'sell'
        spread REAL,
        profit REAL
    );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def clear_signals_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM signals;")
    conn.commit()
    cursor.close()
    conn.close()
