from flask import Flask, render_template, request, Response, stream_with_context
import ingest_data
import train_model
import backtest
import analyze_universe     
import portfolio_optimizer  
import json
import time
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Default list to start ingestion if DB is empty
STOCK_UNIVERSE = [
    'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'PG',
    'MA', 'HD', 'CVX', 'MRK', 'KO', 'PEP', 'BAC', 'COST', 'WMT', 'MCD',
    'DIS', 'CSCO', 'VZ', 'CMCSA', 'ADBE', 'NKE', 'INTC', 'TMUS', 'WFC', 'QCOM'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ingest_data', methods=['POST'])
def ingest_data_route():
    def stream_ingestion():
        yield "data: 🚀 STARTING INGESTION PROCESS...\n\n"
        ingest_data.run_ingestion(STOCK_UNIVERSE, period="5y", log_func=lambda x: None)
        yield "data: ✅ Database populated successfully.\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(stream_ingestion()), mimetype='text/event-stream')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.json
    months_back = int(data['monthsBack'])
    init_cash = float(data['initialCash'])
    stop_loss = float(data['stopLoss']) / 100
    take_profit = float(data['takeProfit']) / 100
    n_est = int(data['rfEstimators'])
    depth = int(data['rfDepth'])
    
    # Check for optional risk parameter, default to 2.0%
    risk_per_trade = float(data.get('riskPerTrade', 2.0)) / 100
    
    backtest_start = datetime.now() - timedelta(days=months_back*30)
    split_date_str = backtest_start.strftime('%Y-%m-%d')
    
    def generate_logs():
        def log(msg):
            yield f"data: {msg}\n\n"

        yield "data: 🚀 INITIALIZING SIMULATION...\n\n"
        
        # 1. TRAIN
        yield "data: \n--- STEP 1: TRAINING BRAIN ---\n\n"
        train_logs = []
        # We capture logs in a list, then yield them
        train_model.run_training(n_est, depth, split_date_str, log_func=train_logs.append)
        for msg in train_logs: yield f"data: {msg}\n\n"
        
        # 2. ANALYZE UNIVERSE (The "Honor Roll")
        yield "data: \n--- STEP 1.5: SELECTING BEST STOCKS ---\n\n"
        try:
            # FIX: Use a list to capture logs instead of lambda: yield
            analysis_logs = []
            honor_roll = analyze_universe.analyze_and_rank(log_func=analysis_logs.append)
            
            # Now stream the captured logs
            for msg in analysis_logs:
                yield f"data: {msg}\n\n"
                
            if not honor_roll:
                yield "data: ❌ Error: No valid stocks found. Aborting.\n\n"
                yield "data: [DONE]\n\n"
                return
        except Exception as e:
            yield f"data: [!] Analysis Error: {e}\n\n"
            return

        # 3. OPTIMIZE PORTFOLIO (The "Target Ratios")
        yield "data: \n--- STEP 1.8: OPTIMIZING ALLOCATION ---\n\n"
        
        # FIX: Same fix here for the optimizer
        opt_logs = []
        target_weights = portfolio_optimizer.get_target_weights(honor_roll, log_func=opt_logs.append)
        
        for msg in opt_logs:
            yield f"data: {msg}\n\n"
        
        yield "data: 📊 TARGET ALLOCATION:\n\n"
        for t, w in target_weights.items():
            yield f"data:    {t:<6}: {w:.1%}\n\n"

        # 4. BACKTEST (The "Execution")
        yield "data: \n--- STEP 2: EXECUTING STRATEGY ---\n\n"
        log_queue = []
        def streamer(msg):
            log_queue.append(msg)
            
        final_eq, final_ret, _ = backtest.run_backtest(
            init_cash, stop_loss, take_profit, target_weights, split_date_str, 
            risk_per_trade=risk_per_trade,
            log_func=streamer
        )
        
        for l in log_queue:
            yield f"data: {l}\n\n"
            time.sleep(0.01) 
            
        report = {
            "equity": f"${final_eq:,.2f}",
            "return": f"{final_ret:.2f}%"
        }
        yield f"data: REPORT_JSON:{json.dumps(report)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate_logs()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)