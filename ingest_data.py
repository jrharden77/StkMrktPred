import sqlite3
import yfinance as yf
import pandas_datareader.data as web
from textblob import TextBlob
import datetime
import os

db_file = 'market_intel.db'

def run_ingestion(tickers, period="5y", log_func=print):
    log_func(f"📥 Starting Ingestion for: {len(tickers)} tickers (Period: {period})")
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    failed_tickers = []

    # 1. Price Data
    for t in tickers:
        log_func(f"   Fetching Price for {t}...") 
        try:
            df = yf.download(t, period=period, interval="1d", progress=False)
            
            if df.empty or df.dropna().empty:
                log_func(f"   [!] Warning: No data for {t}")
                failed_tickers.append(t)
                continue
            
            count = 0
            for index, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO price (ticker, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (t, str(index.date()), row['Open'].iloc[0], row['High'].iloc[0], 
                          row['Low'].iloc[0], row['Close'].iloc[0], int(row['Volume'].iloc[0])))
                    count += 1
                except: pass
            log_func(f"   -> Saved {count} rows for {t}")
        except Exception as e:
            log_func(f"   [!] Error fetching Price for {t}: {e}")
            failed_tickers.append(t)
        conn.commit()

    # 2. News Data
    for t in tickers:
        if t in failed_tickers: continue

        log_func(f"   Fetching News for {t}...")
        try:
            stock = yf.Ticker(t)
            news_items = stock.news
            count = 0
            for item in news_items:
                content = item.get('content', {})
                headline = content.get('title')
                pub_date = content.get('pubDate')
                if not headline or not pub_date: continue
                
                # Sentiment
                blob = TextBlob(headline)
                sentiment = blob.sentiment.polarity
                
                try:
                    ts = pub_date
                    try:
                        dt = datetime.datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ")
                        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except: pass

                    cursor.execute("""
                        INSERT OR IGNORE INTO news (ticker, timestamp, headline, source, sentiment_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (t, ts, headline, "Yahoo", sentiment))
                    count += 1
                except: pass
            log_func(f"   -> Analyzed {count} articles for {t}")
        except Exception as e:
            log_func(f"   [!] Error fetching News for {t}: {e}")
        conn.commit()

    # 3. Macro Data
    log_func("   Fetching Macro Data (FRED)...")
    try:
        indicators = {'GDP': 'GDP', 'CPI': 'CPIAUCSL'}
        start_year = datetime.datetime.now().year - 6 # Dynamic start date
        start_date = datetime.datetime(start_year, 1, 1) 
        for name, code in indicators.items():
            try:
                df = web.DataReader(code, 'fred', start_date)
                for index, row in df.iterrows():
                    cursor.execute("INSERT OR IGNORE INTO macro VALUES (?, ?, ?)", 
                                 (name, str(index.date()), row[code]))
            except: pass
        log_func("   -> Macro data updated.")
    except Exception as e:
        log_func(f"   [!] Macro Error: {e}")

    conn.commit()
    conn.close()
    
    if failed_tickers:
        log_func("\n⚠️  The following tickers failed ingestion:")
        log_func(f"   {', '.join(failed_tickers)}")
    
    log_func("✅ Ingestion Complete.")

if __name__ == "__main__":
    # Example usage for manual run
    test_tickers = ['AAPL', 'TSLA']
    run_ingestion(test_tickers)