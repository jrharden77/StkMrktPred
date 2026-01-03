import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import train_model

def analyze_and_rank(log_func=print):
    log_func("🚀 STARTING UNIVERSE ANALYSIS...")
    log_func("   Fetching data (this takes a moment)...")
    
    # 1. Get Data
    raw_df = train_model.get_data()
    
    # 2. Filter: Drop Penny Stocks (< $10)
    latest_prices = raw_df.sort_values('timestamp').groupby('ticker')['close'].last()
    valid_tickers = latest_prices[latest_prices >= 10.0].index.tolist()
    
    df = raw_df[raw_df['ticker'].isin(valid_tickers)].copy()
    log_func(f"   Filtered Universe: {len(valid_tickers)} stocks (Price > $10)")

    # 3. Add Features
    log_func("   Calculating features...")
    # Suppress verbose logs from add_features if desired, or pass a dummy
    df = train_model.add_features(df, log_func=lambda x: None)
    
    # 4. Prepare Audit
    df['next_close'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    df = df.replace([np.inf, -np.inf], 0).dropna()
    
    features = ['sentiment_score', 'pct_change', 'return_lag_1', 'return_lag_2',
                'volatility_5', 'dist_sma_10', 'rsi_14', 'vol_change',
                'dist_from_poc', 'in_value_area', 'rel_vol_at_price']
    
    dates = df['timestamp'].unique()
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    train_df = df[df['timestamp'] <= split_date]
    test_df = df[df['timestamp'] > split_date]
    
    log_func(f"   Training Audit Model (Split: {pd.to_datetime(split_date).date()})...")
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=42)
    model.fit(train_df[features], train_df['target'])
    
    # 5. Evaluate
    log_func("   Evaluating individual stock predictability...")
    
    results = []
    test_df = test_df.copy()
    test_df['predicted'] = model.predict(test_df[features])
    
    for ticker in test_df['ticker'].unique():
        t_data = test_df[test_df['ticker'] == ticker]
        if len(t_data) < 10: continue
        
        score = precision_score(t_data['target'], t_data['predicted'], zero_division=0)
        volatility = t_data['pct_change'].std()
        
        results.append({
            'ticker': ticker,
            'precision': score,
            'volatility': volatility
        })
        
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        log_func("   [!] No stocks passed data requirements.")
        return []

    # 6. Rank: High Precision, Low Volatility
    vol_cutoff = results_df['volatility'].quantile(0.70)
    
    # Thresholds: >52% Precision, <70th percentile volatility
    safe_bets = results_df[
        (results_df['precision'] > 0.52) & 
        (results_df['volatility'] < vol_cutoff)
    ].sort_values('precision', ascending=False)
    
    honor_roll = safe_bets['ticker'].tolist()
    
    log_func(f"   ✅ Identified {len(honor_roll)} 'Honor Roll' stocks.")
    return honor_roll

if __name__ == "__main__":
    analyze_and_rank()