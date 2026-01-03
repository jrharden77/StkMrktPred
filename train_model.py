import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DB_FILE = 'market_intel.db'
MODEL_FILE = 'ml_strategy_model.pkl'

def get_data():
    conn = sqlite3.connect(DB_FILE)
    price_df = pd.read_sql("SELECT ticker, timestamp, close, volume FROM price", conn)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    
    news_df = pd.read_sql("SELECT ticker, timestamp, sentiment_score FROM news", conn)
    news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
    news_df['date_only'] = news_df['timestamp'].dt.normalize()
    daily_sentiment = news_df.groupby(['ticker', 'date_only'])['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'date_only': 'timestamp'}, inplace=True)
    
    df = pd.merge(price_df, daily_sentiment, on=['ticker', 'timestamp'], how='left')
    df['sentiment_score'] = df['sentiment_score'].fillna(0)
    conn.close()
    return df.sort_values(['ticker', 'timestamp'])

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- NEW: Volume Profile Logic with Progress Logging ---
def calculate_volume_profile_features(df, lookback=60, bins=50, log_func=print):
    df['dist_from_poc'] = np.nan
    df['in_value_area'] = np.nan
    df['rel_vol_at_price'] = np.nan
    
    tickers = df['ticker'].unique()
    total_tickers = len(tickers)
    
    for t_idx, ticker in enumerate(tickers):
        # Progress log every 5 tickers
        if t_idx % 5 == 0:
            log_func(f"   ... Processing {ticker} ({t_idx+1}/{total_tickers})")
            
        ticker_mask = df['ticker'] == ticker
        idx_list = df.index[ticker_mask]
        
        closes = df.loc[idx_list, 'close'].values
        volumes = df.loc[idx_list, 'volume'].values
        
        for i in range(lookback, len(idx_list)):
            curr_idx = idx_list[i]
            
            window_closes = closes[i-lookback:i]
            window_vols = volumes[i-lookback:i]
            
            if len(window_closes) < 1: continue
            
            price_min = np.min(window_closes)
            price_max = np.max(window_closes)
            
            hist, bin_edges = np.histogram(window_closes, bins=bins, weights=window_vols, range=(price_min, price_max))
            
            max_vol_idx = np.argmax(hist)
            poc_price = (bin_edges[max_vol_idx] + bin_edges[max_vol_idx+1]) / 2
            poc_vol = hist[max_vol_idx]
            
            total_vol = np.sum(hist)
            target_vol = total_vol * 0.70
            
            current_vol = poc_vol
            left_idx = max_vol_idx
            right_idx = max_vol_idx
            
            while current_vol < target_vol:
                vol_left = hist[left_idx-1] if left_idx > 0 else 0
                vol_right = hist[right_idx+1] if right_idx < len(hist)-1 else 0
                
                if vol_left == 0 and vol_right == 0: break
                
                if vol_left > vol_right:
                    left_idx -= 1
                    current_vol += vol_left
                else:
                    right_idx += 1
                    current_vol += vol_right
            
            va_low = bin_edges[left_idx]
            va_high = bin_edges[right_idx+1]
            
            current_price = closes[i]
            dist_poc = (current_price - poc_price) / poc_price
            in_va = 1 if (current_price >= va_low and current_price <= va_high) else 0
            
            bin_idx = np.digitize(current_price, bin_edges) - 1
            if bin_idx < 0: bin_idx = 0
            if bin_idx >= len(hist): bin_idx = len(hist) - 1
            vol_at_price = hist[bin_idx]
            rel_vol = vol_at_price / poc_vol if poc_vol > 0 else 0
            
            df.at[curr_idx, 'dist_from_poc'] = dist_poc
            df.at[curr_idx, 'in_value_area'] = in_va
            df.at[curr_idx, 'rel_vol_at_price'] = rel_vol

    return df

def add_features(df, log_func=print):
    df = df.copy()
    df['pct_change'] = df.groupby('ticker')['close'].pct_change()
    df['return_lag_1'] = df.groupby('ticker')['pct_change'].shift(1)
    df['return_lag_2'] = df.groupby('ticker')['pct_change'].shift(2)
    df['volatility_5'] = df.groupby('ticker')['pct_change'].transform(lambda x: x.rolling(5).std())
    df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).mean())
    df['sma_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(50).mean())
    df['dist_sma_10'] = (df['close'] - df['sma_10']) / df['sma_10']
    df['rsi_14'] = df.groupby('ticker')['close'].transform(lambda x: calculate_rsi(x, 14))
    df['vol_change'] = df.groupby('ticker')['volume'].pct_change()
    
    log_func("   Calculating Volume Profile features (this will take a minute)...")
    # Pass log_func down to the heavy calculator
    df = calculate_volume_profile_features(df, log_func=log_func)
    
    return df

def run_training(n_estimators=200, max_depth=10, split_date_str='2025-10-01', log_func=print):
    log_func(f"🧠 Training Model (Trees={n_estimators}, Depth={max_depth})...")
    
    if os.path.exists(MODEL_FILE):
        try:
            os.remove(MODEL_FILE)
            log_func(f"   [Debug] Deleted old model file to ensure clean train.")
        except Exception as e:
            log_func(f"   [!] Warning: Could not delete old model: {e}")

    try:
        raw_data = get_data()
        if raw_data.empty:
            log_func("   [!] Error: No data found in DB.")
            return

        # Pass log_func to add_features
        df = add_features(raw_data, log_func=log_func)
        
        train_df = df[df['timestamp'] <= split_date_str].copy()
        
        if len(train_df) < 100:
            log_func(f"   [!] Error: Not enough data before {split_date_str}")
            return

        train_df['next_close'] = train_df.groupby('ticker')['close'].shift(-1)
        train_df['target'] = (train_df['next_close'] > train_df['close']).astype(int)
        
        # --- CLEAN INFINITE VALUES ---
        train_df = train_df.replace([np.inf, -np.inf], 0)
        
        train_df = train_df.dropna()
        
        features = ['sentiment_score', 'pct_change', 'return_lag_1', 'return_lag_2',
                    'volatility_5', 'dist_sma_10', 'rsi_14', 'vol_change',
                    'dist_from_poc', 'in_value_area', 'rel_vol_at_price'] 
        
        log_func(f"   [Debug] Training on features: {features}")
        
        X = train_df[features]
        y = train_df['target']
        
        model = RandomForestClassifier(n_estimators=int(n_estimators), 
                                     max_depth=int(max_depth), 
                                     min_samples_split=20,
                                     min_samples_leaf=10,
                                     random_state=42)
        model.fit(X, y)
        
        joblib.dump(model, MODEL_FILE)
        log_func("✅ Model Saved.")
        
    except Exception as e:
        log_func(f"   [!] Training Failed: {e}")