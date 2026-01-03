import sqlite3
import pandas as pd
import uuid
import math
import numpy as np
import time 
from datetime import datetime
import strategy

conn = None
cursor = None

def setup_db():
    global conn, cursor
    conn = sqlite3.connect('market_intel.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id TEXT PRIMARY KEY,
            ticker TEXT,
            side TEXT,
            entry_price REAL,
            exit_price REAL,
            qty REAL,
            timestamp TEXT,
            model_confidence REAL,
            status TEXT
        )
    """)
    conn.commit()

def clean_slate(log_func):
    log_func("🧹 Clearing old trades...")
    cursor.execute("DELETE FROM paper_trades")
    conn.commit()

def calculate_atr(ticker, current_date, window=14):
    query = f"""
        SELECT high, low, close 
        FROM price 
        WHERE ticker = '{ticker}' AND timestamp <= '{current_date}'
        ORDER BY timestamp DESC 
        LIMIT {window + 1}
    """
    df = pd.read_sql_query(query, conn)
    if len(df) < window + 1: return 0.0
    df = df.iloc[::-1].reset_index(drop=True)
    tr_list = []
    for i in range(1, len(df)):
        h = df.loc[i, 'high']
        l = df.loc[i, 'low']
        pc = df.loc[i-1, 'close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)
    return np.mean(tr_list)

def check_exit(ticker, current_date, open_trade, stop_pct, profit_pct, slippage_pct, tax_rate, log_func):
    trade_id = open_trade['trade_id']
    entry_price = open_trade['entry_price']
    side = open_trade['side']
    qty = open_trade['qty']
    
    atr = calculate_atr(ticker, current_date)
    min_stop = entry_price * stop_pct
    actual_stop = max(min_stop, atr * 3.0)
    
    stop_price = entry_price - actual_stop
    target_price = entry_price + (actual_stop * 1.5)
    
    query = f"SELECT high, low, close FROM price WHERE ticker = '{ticker}' AND timestamp = '{current_date}'"
    daily_price = pd.read_sql_query(query, conn)
    if daily_price.empty: return 0.0

    day_high = daily_price['high'].iloc[0]
    day_low = daily_price['low'].iloc[0]
    
    exit_price = None
    if day_low <= stop_price: exit_price = stop_price 
    elif day_high >= target_price: exit_price = target_price 
            
    if exit_price:
        adj_exit_price = exit_price * (1 - slippage_pct)
        cursor.execute("UPDATE paper_trades SET exit_price=?, status='CLOSED' WHERE trade_id=?", (adj_exit_price, trade_id))
        conn.commit()
        
        gross_return = adj_exit_price * qty
        profit = (adj_exit_price - entry_price) * qty
        tax = profit * tax_rate if profit > 0 else 0.0
        return gross_return - tax
            
    return 0.0

def run_backtest(initial_cash, stop_loss, take_profit, target_allocation, start_date_str, risk_per_trade=0.02, log_func=print):
    import train_model
    setup_db()
    
    tickers = list(target_allocation.keys())
    
    # Verify Data
    missing_data = []
    for t in tickers:
        try:
            res = pd.read_sql_query(f"SELECT count(*) as cnt FROM price WHERE ticker='{t}'", conn)
            if res['cnt'].iloc[0] < 50: missing_data.append(t)
        except: missing_data.append(t)
    
    if missing_data:
        log_func(f"⚠️ Removing {len(missing_data)} tickers due to missing data.")
        for t in missing_data: del target_allocation[t]
        tickers = list(target_allocation.keys())

    clean_slate(log_func)
    
    current_cash = float(initial_cash)
    start_cash = current_cash
    
    SLIPPAGE_PCT = 0.0025
    TAX_RATE = 0.25
    
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.now()
    
    log_func(f"⏳ Backtest: {start_date.date()} -> {end_date.date()}")
    log_func(f"   Strategy: Optimal Portfolio Rebalancing (Targeting {len(tickers)} stocks)")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    last_sim_date = start_date_str

    for single_date in date_range:
        sim_date = single_date.strftime('%Y-%m-%d')
        last_sim_date = sim_date

        check = pd.read_sql_query(f"SELECT 1 FROM price WHERE timestamp='{sim_date}' LIMIT 1", conn)
        if check.empty: continue
        
        # 1. Manage Exits (Stops/Targets)
        all_open_trades = pd.read_sql_query("SELECT * FROM paper_trades WHERE status='OPEN'", conn)
        if not all_open_trades.empty:
            for _, row in all_open_trades.iterrows():
                cash_back = check_exit(row['ticker'], sim_date, row, stop_loss, take_profit, SLIPPAGE_PCT, TAX_RATE, log_func)
                if cash_back > 0:
                    current_cash += cash_back
                    log_func(f"   [X] Closed {row['ticker']}. Cash: ${current_cash:,.2f}")

        # 2. Rebalancing Logic
        # Calculate Total Portfolio Value
        portfolio_val = current_cash
        open_pos_map = {}
        
        # Get current value of open positions
        current_opens = pd.read_sql_query("SELECT ticker, qty, entry_price FROM paper_trades WHERE status='OPEN'", conn)
        for _, row in current_opens.iterrows():
            # Use Entry Price as proxy for current val to speed up loop (or query DB for precision)
            # For strict rebalancing, let's query the day's close
            try:
                p_query = f"SELECT close FROM price WHERE ticker='{row['ticker']}' AND timestamp='{sim_date}'"
                curr_p = pd.read_sql_query(p_query, conn)['close'].iloc[0]
                val = curr_p * row['qty']
                portfolio_val += val
                open_pos_map[row['ticker']] = val
            except:
                open_pos_map[row['ticker']] = 0

        debug_printed = 0
        
        # Check every ticker in our Target Allocation
        for t, target_weight in target_allocation.items():
            
            # How much SHOULD we have?
            target_val = portfolio_val * target_weight
            current_val = open_pos_map.get(t, 0.0)
            
            # Are we already full?
            if current_val >= target_val * 0.95: # 5% buffer
                continue
                
            # ML Decision Time
            signal, confidence = strategy.get_ml_signal(t, sim_date)
            
            if debug_printed < 2:
                log_func(f"   [DEBUG] {t}: {signal} ({confidence:.2f}) [Target: {target_weight:.1%}]")
                debug_printed += 1

            if signal == "BUY":
                # We are underweight AND model says BUY. Fill the gap.
                price_df = pd.read_sql_query(f"SELECT close FROM price WHERE ticker='{t}' AND timestamp='{sim_date}'", conn)
                if price_df.empty: continue
                price = price_df['close'].iloc[0]
                
                # Calculate how much to buy to reach target
                # Cap it at current cash
                needed_val = target_val - current_val
                spendable = min(needed_val, current_cash)
                
                # Don't make tiny trades ($50 minimum)
                if spendable < 50: continue
                
                entry_price = price * (1 + SLIPPAGE_PCT)
                qty = round(spendable / entry_price, 4)
                
                if qty > 0:
                    current_cash -= (entry_price * qty)
                    trade_id = str(uuid.uuid4())
                    cursor.execute("""
                        INSERT INTO paper_trades (trade_id, ticker, side, entry_price, qty, timestamp, model_confidence, status) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (trade_id, t, 'LONG', entry_price, qty, sim_date, confidence, 'OPEN'))
                    conn.commit()
                    log_func(f"   [+] BUY {t} ({confidence:.0%} conf): {qty} shares @ ${entry_price:.2f} (Targeting {target_weight:.1%})")

            elif signal == "SELL" and current_val > 0:
                # Model says SELL. Close the position to protect capital.
                # We ignore the target ratio here because the model predicts a downturn.
                # Logic: "Ratio is a goal, but survival is mandatory."
                
                # Find the open trade to close it
                row = pd.read_sql_query(f"SELECT * FROM paper_trades WHERE ticker='{t}' AND status='OPEN'", conn)
                if not row.empty:
                    # Force close logic
                    trade_id = row['trade_id'].iloc[0]
                    qty = row['qty'].iloc[0]
                    entry_p = row['entry_price'].iloc[0]
                    
                    price_df = pd.read_sql_query(f"SELECT close FROM price WHERE ticker='{t}' AND timestamp='{sim_date}'", conn)
                    curr_p = price_df['close'].iloc[0]
                    
                    adj_exit = curr_p * (1 - SLIPPAGE_PCT)
                    
                    cursor.execute("UPDATE paper_trades SET exit_price=?, status='CLOSED' WHERE trade_id=?", (adj_exit, trade_id))
                    conn.commit()
                    
                    cash_returned = adj_exit * qty
                    profit = (adj_exit - entry_p) * qty
                    tax = profit * TAX_RATE if profit > 0 else 0
                    final_cash = cash_returned - tax
                    
                    current_cash += final_cash
                    log_func(f"   [-] SELL {t} ({confidence:.0%} conf): Closing pos (Model predicted drop)")

    # Final Equity
    total_equity = current_cash
    opens = pd.read_sql_query("SELECT * FROM paper_trades WHERE status='OPEN'", conn)
    for _, row in opens.iterrows():
        try:
            last_p = pd.read_sql_query(f"SELECT close FROM price WHERE ticker='{row['ticker']}' AND timestamp <= '{last_sim_date}' ORDER BY timestamp DESC LIMIT 1", conn)['close'].iloc[0]
            total_equity += last_p * row['qty']
        except: pass
            
    ret_pct = (total_equity - start_cash) / start_cash * 100
    
    log_func("="*30)
    log_func("🏁 SIMULATION COMPLETE")
    log_func(f"   Final Equity: ${total_equity:,.2f}")
    log_func(f"   Return:       {ret_pct:.2f}%")
    log_func("="*30)
    
    conn.close()
    return total_equity, ret_pct, 0