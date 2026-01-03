import sqlite3
import pandas as pd

# Connect to the database
db_file = 'market_intel.db'
conn = sqlite3.connect(db_file)

def generate_report():
    print("\n" + "="*40)
    print(" 📊 TRADING PERFORMANCE REPORT")
    print("="*40)

    # 1. Fetch only CLOSED trades (we can't score open ones yet)
    query = "SELECT * FROM paper_trades WHERE status = 'CLOSED'"
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("\n[i] No closed trades found yet.")
        print("    Wait for the strategy to exit a position first.")
        return

    # 2. Calculate Profit/Loss (PnL) for each trade
    # We need to handle LONG vs SHORT logic differently
    
    pnl_list = []
    
    for index, row in df.iterrows():
        entry = row['entry_price']
        exit_p = row['exit_price']
        qty = row['qty']
        side = row['side']
        
        if side == 'LONG':
            # Profit = (Exit - Entry) * Qty
            pnl = (exit_p - entry) * qty
        else:
            # Profit = (Entry - Exit) * Qty (Short selling)
            pnl = (entry - exit_p) * qty
            
        pnl_list.append(pnl)

    df['pnl'] = pnl_list

    # 3. Calculate Aggregate Metrics
    total_trades = len(df)
    winning_trades = df[df['pnl'] > 0]
    losing_trades = df[df['pnl'] <= 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    
    win_rate = (win_count / total_trades) * 100
    total_profit = df['pnl'].sum()
    avg_trade = df['pnl'].mean()
    
    best_trade = df['pnl'].max()
    worst_trade = df['pnl'].min()

    # 4. Print The Summary
    print(f"\nSummary:")
    print(f"--------")
    print(f"Total Trades:      {total_trades}")
    print(f"Wins:              {win_count}")
    print(f"Losses:            {loss_count}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"------------------------")
    print(f"TOTAL PnL:        ${total_profit:.2f}")
    print(f"Avg per Trade:    ${avg_trade:.2f}")
    print(f"Best Trade:       ${best_trade:.2f}")
    print(f"Worst Trade:      ${worst_trade:.2f}")
    
    # 5. Show Recent History
    print("\nRecent Trade Log:")
    print("-" * 65)
    print(f"{'TICKER':<8} {'SIDE':<6} {'ENTRY':<10} {'EXIT':<10} {'PnL ($)':<10}")
    print("-" * 65)
    
    for index, row in df.tail(10).iterrows():
        # Color coding for terminal (optional, simplified here)
        pnl_str = f"${row['pnl']:.2f}"
        print(f"{row['ticker']:<8} {row['side']:<6} ${row['entry_price']:<9.2f} ${row['exit_price']:<9.2f} {pnl_str:<10}")

if __name__ == "__main__":
    generate_report()
    conn.close()