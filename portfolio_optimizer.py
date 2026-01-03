import sqlite3
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import datetime

# Configuration
DB_FILE = 'market_intel.db'
RISK_FREE_RATE = 0.04 

def get_historical_prices(tickers, days_back=365):
    conn = sqlite3.connect(DB_FILE)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    start_date_str = start_date.strftime('%Y-%m-%d')
    
    ticker_str = "','" .join(tickers)
    query = f"""
        SELECT timestamp, ticker, close 
        FROM price 
        WHERE ticker IN ('{ticker_str}') 
        AND timestamp >= '{start_date_str}'
        ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty: return pd.DataFrame()

    pivot_df = df.pivot(index='timestamp', columns='ticker', values='close')
    # Use forward fill for missing days, then drop remaining NaNs
    pivot_df = pivot_df.ffill().dropna()
    return pivot_df

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret = np.sum(mean_returns * weights) * 252
    p_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return -(p_ret - risk_free_rate) / p_var

def get_target_weights(tickers, log_func=print):
    """
    Main entry point: Receives list of tickers, returns dict of {ticker: weight}
    """
    if not tickers:
        log_func("   [!] No tickers provided to optimizer.")
        return {}

    log_func(f"📊 Optimizing Allocation for {len(tickers)} stocks...")
    
    # 1. Get Data
    prices = get_historical_prices(tickers)
    if prices.empty or len(prices.columns) < 2:
        log_func("   [!] Not enough historical data for optimization. Using equal weights.")
        return {t: 1.0/len(tickers) for t in tickers}

    # 2. Calc Stats
    log_returns = np.log(prices / prices.shift(1)).dropna()
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, RISK_FREE_RATE)
    
    # 3. Optimize
    # Constraints: Sum = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Bounds: 0% to 30% per stock (Diversification Cap)
    bounds = tuple((0.0, 0.30) for asset in range(num_assets))
    
    init_guess = num_assets * [1. / num_assets,]
    
    try:
        result = minimize(neg_sharpe_ratio, init_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
        
        weights = result.x
        
        # Clean up dust (< 1%)
        allocations = {}
        for idx, ticker in enumerate(mean_returns.index):
            w = weights[idx]
            if w > 0.01: # 1% cutoff
                allocations[ticker] = round(w, 4)
                
        # Re-normalize to 100% after cutting dust
        total_w = sum(allocations.values())
        if total_w > 0:
            allocations = {k: v/total_w for k, v in allocations.items()}
            
        return allocations
        
    except Exception as e:
        log_func(f"   [!] Optimization failed: {e}. Using equal weights.")
        return {t: 1.0/len(tickers) for t in tickers}

if __name__ == "__main__":
    # Test run
    print(get_target_weights(['AAPL', 'MSFT', 'GOOGL']))