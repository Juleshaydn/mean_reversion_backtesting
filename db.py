# db.py

import psycopg2
import os

def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "stock_price_db"),
        user=os.getenv("DB_USER", "your_db_username"),
        password=os.getenv("DB_PASSWORD", "your_secure_password")
    )
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create table for stock prices
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stock_prices (
        symbol VARCHAR(10) PRIMARY KEY,
        price REAL,
        timestamp TIMESTAMP
    );
    """)
    # Create or alter the historical_stock_data table
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
    conn.commit()
    cursor.close()
    conn.close()
