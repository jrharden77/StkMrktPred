import yfinance as yf
import json

ticker = "AAPL"
print(f"--- DIAGNOSTICS FOR {ticker} ---")

stock = yf.Ticker(ticker)
news_items = stock.news

if not news_items:
    print("CRITICAL: Yahoo returned 0 articles. The list is empty.")
else:
    # Get the first article
    item = news_items[0]
    
    # Print all available keys (labels) in the data
    print(f"Found {len(news_items)} articles.")
    print("\n--- AVAILABLE KEYS ---")
    print(list(item.keys()))
    
    print("\n--- RAW DATA (FIRST ARTICLE) ---")
    # Pretty print the dictionary so we can read it
    print(json.dumps(item, indent=4))