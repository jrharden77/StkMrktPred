import ingest_data
import train_model   # <--- NEW: Import your trainer
import backtest      # <--- NEW: Import your backtester
import performance
import time
import datetime

def run_full_cycle():
    print("\n" + "="*60)
    print(f"🚀 STARTING FULL MARKET CYCLE: {datetime.datetime.now()}")
    print("="*60)

    # STEP 1: INGEST (Get the fresh 3-year history)
    print("\n[1/3] 📥 INGESTING DATA (Price, News, Macro)...")
    try:
        tickers = ['AAPL', 'TSLA', 'NVDA']
        
        for t in tickers:
            ingest_data.ingest_price_data(t)
            ingest_data.ingest_news_data(t)
        
        ingest_data.ingest_macro_data()
        print("   -> Data Ingestion Complete.")
        
    except Exception as e:
        print(f"   [!] CRITICAL ERROR during ingestion: {e}")
        return 

    # STEP 2: TRAIN (Update the 'Brain')
    print("\n[2/3] 🧠 RETRAINING ML MODEL...")
    try:
        # This will fetch the new data from DB, train, and save 'ml_strategy_model.pkl'
        train_model.train_model()
        print("   -> Model Retrained & Saved.")
        
    except Exception as e:
        print(f"   [!] CRITICAL ERROR during training: {e}")
        return

    # STEP 3: BACKTEST (Simulate & Verify)
    print("\n[3/3] ⚔️ RUNNING BACKTEST & STRATEGY...")
    try:
        # Runs the simulation using the brand new model
        # You can change days_back=180 to test further back if you want
        backtest.run_backtest(days_back=90)
            
    except Exception as e:
        print(f"   [!] CRITICAL ERROR during backtest: {e}")

    print("\n" + "="*60)
    print("✅ CYCLE COMPLETE.")
    print("="*60)

if __name__ == "__main__":
    run_full_cycle()