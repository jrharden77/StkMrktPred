import sqlite3



db_file = 'market_intel.db'

try:
    # Connect to the database
    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        
# Historial Prices (OHLCV) Table Creation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price (
                ticker STRING,
                timestamp DATETIME,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume INTEGER,
                PRIMARY KEY (ticker, timestamp)
            )
        """)
# Sentiment Data (Qualitative)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news (
                news_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker STRING,
                timestamp DATETIME,
                headline TEXT,
                source STRING,
                sentiment_score FLOAT,
                UNIQUE (ticker, timestamp, headline) 
            )
        """)

# Macro Indicators (Environment)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro (
                indicator_name STRING,
                timestamp DATETIME,
                value FLOAT,
                PRIMARY KEY (indicator_name, timestamp)
            )
        """)

#Trade Sandbox (Performance Tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id STRING PRIMARY KEY,
                ticker STRING,
                side STRING,
                entry_price FLOAT,
                exit_price FLOAT,
                qty INTEGER,
                timestamp DATETIME,
                model_confidence FLOAT,
                status STRING
            )
        """)

        # No need to call conn.commit() manually here!
        # The 'with' block automatically commits if everything succeeds,
        # or rolls back (undoes) changes if an error occurs.
        
    print("Database connected and initialized.")

except sqlite3.Error as e:
    print(f"An error occurred: {e}")