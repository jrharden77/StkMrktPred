import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
import time

# Configuration
DB_FILE = 'market_intel.db'
MODEL_FILE = 'ml_strategy_model.pkl'

_cached_model = None
_last_load_time = 0

def get_model():
    global _cached_model, _last_load_time
    if not os.path.exists(MODEL_FILE): return None
    file_mtime = os.path.getmtime(MODEL_FILE)
    if _cached_model is None or file_mtime > _last_load_time:
        try:
            print(f"🔄 Reloading ML Model (Timestamp: {file_mtime})...")
            _cached_model = joblib.load(MODEL_FILE)
            _last_load_time = file_mtime
        except Exception as e:
            return None
    return _cached_model

def calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_volume_profile_features(df, lookback=60, bins=50):
    if len(df) < lookback: return 0.0, 0, 0.0
    
    window_data = df.iloc[-lookback:]
    closes = window_data['close'].values
    volumes = window_data['volume'].values
    current_price = closes[-1]
    
    price_min = np.min(closes)
    price_max = np.max(closes)
    
    if price_min == price_max: return 0.0, 0, 0.0
    
    hist, bin_edges = np.histogram(closes, bins=bins, weights=volumes, range=(price_min, price_max))
    
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
    
    # Division by zero protection
    if poc_price == 0: dist_poc = 0.0
    else: dist_poc = (current_price - poc_price) / poc_price
    
    in_va = 1 if (current_price >= va_low and current_price <= va_high) else 0
    
    bin_idx = np.digitize(current_price, bin_edges) - 1
    if bin_idx < 0: bin_idx = 0
    if bin_idx >= len(hist): bin_idx = len(hist) - 1
    vol_at_price = hist[bin_idx]
    
    if poc_vol == 0: rel_vol = 0.0
    else: rel_vol = vol_at_price / poc_vol
    
    return dist_poc, in_va, rel_vol

def get_ml_signal(ticker, current_date_str):
    model = get_model()
    if model is None: return "ERR_NO_MODEL", 0.0

    conn = sqlite3.connect(DB_FILE)
    
    query = f"""
        SELECT timestamp, close, volume 
        FROM price 
        WHERE ticker = '{ticker}' AND timestamp <= '{current_date_str}'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql(query, conn)
    
    news_query = f"""
        SELECT avg(sentiment_score) as sentiment
        FROM news 
        WHERE ticker = '{ticker}' 
        AND timestamp >= date('{current_date_str}', '-1 day')
        AND timestamp <= '{current_date_str} 23:59:59'
    """
    try:
        sentiment_val = pd.read_sql(news_query, conn)['sentiment'].iloc[0]
    except:
        sentiment_val = 0.0

    conn.close()
    
    if len(df) < 90: return "ERR_NO_DATA", 0.0
    
    df['pct_change'] = df['close'].pct_change()
    df['return_lag_1'] = df['pct_change'].shift(1)
    df['return_lag_2'] = df['pct_change'].shift(2)
    df['volatility_5'] = df['pct_change'].rolling(5).std()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['dist_sma_10'] = (df['close'] - df['sma_10']) / df['sma_10']
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['vol_change'] = df['volume'].pct_change()
    
    row = df.iloc[-1]
    current_sentiment = sentiment_val if sentiment_val is not None else 0.0
    
    dist_poc, in_va, rel_vol = get_volume_profile_features(df, lookback=60)
    
    features = pd.DataFrame([{
        'sentiment_score': current_sentiment,
        'pct_change': row['pct_change'],
        'return_lag_1': row['return_lag_1'],
        'return_lag_2': row['return_lag_2'],
        'volatility_5': row['volatility_5'],
        'dist_sma_10': row['dist_sma_10'],
        'rsi_14': row['rsi_14'],
        'vol_change': row['vol_change'],
        'dist_from_poc': dist_poc,       
        'in_value_area': in_va,          
        'rel_vol_at_price': rel_vol      
    }])
    
    # Critical: Fill NaNs with 0 to prevent model crash
    features = features.fillna(0)
    
    # Check for infinity
    features = features.replace([np.inf, -np.inf], 0)
    
    try:
        probs = model.predict_proba(features)[0] 
        prob_up = probs[1]
        
        # --- FIX: HIGHER CONFIDENCE THRESHOLD ---
        # 0.55 / 0.45 threshold eliminates "coin flip" trades
        print(f"   [DEBUG CHECK] Signal Prob: {prob_up:.2f}")
        if prob_up > 0.55: return "BUY", prob_up
        elif prob_up < 0.45: return "SELL", prob_up
        else: return "HOLD", prob_up
        # ----------------------------------------
            
    except Exception as e:
        # Return error message as signal for debugging
        return f"ERR_{str(e)[:15]}", 0.0