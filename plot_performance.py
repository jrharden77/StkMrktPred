import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

conn = sqlite3.connect('market_intel.db')

def plot_equity_curve():
    # 1. Fetch Closed Trades
    df = pd.read_sql("SELECT * FROM paper_trades WHERE status='CLOSED'", conn)
    
    if df.empty:
        print("No closed trades to plot yet!")
        return

    # 2. Calculate PnL per trade (Same as before)
    pnl_list = []
    for index, row in df.iterrows():
        if row['side'] == 'LONG':
            pnl = (row['exit_price'] - row['entry_price']) * row['qty']
        else:
            pnl = (row['entry_price'] - row['exit_price']) * row['qty']
        pnl_list.append(pnl)
    
    df['pnl'] = pnl_list
    
    # --- NEW: AGGREGATE BY DATE ---
    # Convert timestamp string to actual Date object
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Group all trades from the same day and Sum them up
    daily_df = df.groupby('date')['pnl'].sum().reset_index()
    daily_df = daily_df.sort_values('date')
    
    # 3. Cumulative Sum (Equity Curve)
    daily_df['equity'] = daily_df['pnl'].cumsum()

    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Main Line
    plt.plot(daily_df['date'], daily_df['equity'], 
             color='#006400', linewidth=2, label='Total Profit')
    
    # Green fill under the line (Makes it look pro)
    plt.fill_between(daily_df['date'], daily_df['equity'], 0, 
                     where=(daily_df['equity'] >= 0), 
                     color='#32CD32', alpha=0.3, interpolate=True)
    
    # Red fill if we ever dip below zero
    plt.fill_between(daily_df['date'], daily_df['equity'], 0, 
                     where=(daily_df['equity'] < 0), 
                     color='#8B0000', alpha=0.3, interpolate=True)
    
    # Styles
    plt.title('Strategy Performance: Daily Equity Curve', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Net Profit ($)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=1)
    
    # Format Date Axis to look nice
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_equity_curve()
    conn.close()