import train_model
import backtest
import strategy  # <--- IMPORT THIS so we can reload it
import pandas as pd
from datetime import datetime, timedelta
import time
import importlib # <--- REQUIRED for reloading the module

# --- ⚙️ CONFIGURATION ---
TICKERS = ['AAPL', 'TSLA', 'NVDA']
INITIAL_CASH = 10000
STOP_LOSS = 0.03    # 3%
TAKE_PROFIT = 0.06  # 6%
MONTHS_BACK = 3

# --- 🔍 GRID SETTINGS ---
# The script will test every combination of these lists
N_ESTIMATORS_LIST = [200, 500, 2000]
MAX_DEPTH_LIST = [5, 10, 15, 20]
# 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
# 5, 10, 15, 20

def run_grid_search():
    print(f"\n🚀 STARTING GRID SEARCH AUTOMATION")
    print(f"   Stocks:       {TICKERS}")
    print(f"   Start Cash:   ${INITIAL_CASH:,.2f}")
    print(f"   Strategy:     Stop {STOP_LOSS*100}% | Target {TAKE_PROFIT*100}%")
    print(f"   Combinations: {len(N_ESTIMATORS_LIST)} estimators x {len(MAX_DEPTH_LIST)} depths = {len(N_ESTIMATORS_LIST)*len(MAX_DEPTH_LIST)} runs")
    print("="*75)
    print(f"{'EST':<6} {'DEPTH':<8} {'TIME':<8} {'RETURN':<10} {'EQUITY':<15}")
    print("-" * 75)

    # Calculate date logic once
    start_date = datetime.now() - timedelta(days=MONTHS_BACK*30)
    split_date_str = start_date.strftime('%Y-%m-%d')

    results = []
    total_start_time = time.time()

    for n_est in N_ESTIMATORS_LIST:
        for depth in MAX_DEPTH_LIST:
            iter_start = time.time()
            
            try:
                # 1. Train Model (Silence output with lambda)
                train_model.run_training(n_est, depth, split_date_str, log_func=lambda x: None)
                
                # 2. FORCE RELOAD of Strategy to pick up the new model file
                importlib.reload(strategy)
                
                # 3. Run Backtest (Silence output with lambda)
                final_equity, ret_pct = backtest.run_backtest(
                    INITIAL_CASH, STOP_LOSS, TAKE_PROFIT, TICKERS, split_date_str, 
                    log_func=lambda x: None
                )
                
                duration = time.time() - iter_start
                
                # Print result row
                print(f"{n_est:<6} {depth:<8} {duration:>5.1f}s   {ret_pct:>7.2f}%   ${final_equity:,.2f}")
                
                results.append({
                    'EST': n_est,
                    'DEPTH': depth,
                    'RETURN': ret_pct,
                    'EQUITY': final_equity,
                    'TIME': duration
                })
                
            except Exception as e:
                print(f"{n_est:<6} {depth:<8} ERROR: {e}")

    total_duration = time.time() - total_start_time
    
    print("="*75)
    print(f"✅ GRID SEARCH COMPLETE in {total_duration/60:.1f} minutes")
    
    if results:
        # Create DataFrame and Sort by Return
        df = pd.DataFrame(results)
        df = df.sort_values(by='RETURN', ascending=False)
        
        # Reorder columns for display
        df = df[['EST', 'DEPTH', 'TIME', 'RETURN', 'EQUITY']]
        
        print("\n🏆 LEADERBOARD (Best to Worst):")
        print(df.to_string(index=False, formatters={
            'RETURN': '{:.2f}%'.format,
            'EQUITY': '${:,.2f}'.format,
            'TIME': '{:.1f}s'.format
        }))
        
        best = df.iloc[0]
        print("\n🌟 WINNING CONFIGURATION:")
        print(f"   n_estimators: {int(best['EST'])}")
        print(f"   max_depth:    {int(best['DEPTH'])}")
        print(f"   Return:       {best['RETURN']:.2f}%")
        print(f"   Final Equity: ${best['EQUITY']:,.2f}")
        print("\n   -> Enter these values into your Web Dashboard to apply them.")

if __name__ == "__main__":
    run_grid_search()