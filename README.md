# Stock Market Prediction & Analysis System (StkMrktPred)

A comprehensive Python framework for algorithmic trading, machine learning-based market prediction, backtesting, and portfolio optimization. This project provides an end-to-end pipeline from data ingestion to a web-based performance dashboard.

## 🚀 Features

* **Data Ingestion**: Automated fetching and storage of historical market data into a SQLite database (`market_intel.db`).
* **Machine Learning Integration**: Includes scripts to train (`train_model.py`) and tune (`grid_search.py`) predictive models.
* **Robust Backtesting Engine**: Simulate trading strategies against historical data to evaluate performance before going live.
* **Portfolio Optimization**: Tools to analyze the stock universe and optimize portfolio allocation using modern portfolio theory.
* **Performance Metrics**: Detailed calculation of returns, Sharpe ratios, and drawdowns.
* **Web Dashboard**: A Flask-based web interface (`app.py`) to visualize strategy performance and market intelligence.

## 📂 Project Structure

```text
StkMrktPred/
├── app.py                   # Main entry point for the Web Dashboard
├── main.py                  # Main script for running backtests/simulations
├── ingest_data.py           # Script to fetch and store market data
├── market_intel_database.py # Database management and interface
├── train_model.py           # Logic for training the ML strategy model
├── grid_search.py           # Hyperparameter tuning for the ML model
├── strategy.py              # Core trading strategy logic
├── backtest.py              # Backtesting engine implementation
├── portfolio_optimizer.py   # Portfolio allocation optimization
├── analyze_universe.py      # Scripts to analyze stock universe properties
├── performance.py           # Metric calculations (Sharpe, Volatility, etc.)
├── plot_performance.py      # Visualization tools
├── templates/               # HTML templates for the web app
│   └── index.html
└── ml_strategy_model.pkl    # Serialized Machine Learning model
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/StkMrktPred.git
   cd StkMrktPred
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   *(Note: Ensure you have `pandas`, `numpy`, `scikit-learn`, `flask`, and data fetching libraries like `yfinance` installed)*
   ```bash
   pip install pandas numpy scikit-learn flask matplotlib yfinance
   ```

## 📖 Usage

### 1. Data Ingestion
Populate the database with the latest market data:
```bash
python ingest_data.py
```

### 2. Model Training
Train the machine learning model. This will update the `ml_strategy_model.pkl` file.
```bash
python train_model.py
```
*Optional: Run `python grid_search.py` first to find optimal hyperparameters.*

### 3. Running a Backtest
Execute the main script to run the strategy against historical data:
```bash
python main.py
```

### 4. Portfolio Optimization
Analyze the universe and generate optimal portfolio weights:
```bash
python portfolio_optimizer.py
```

### 5. Web Dashboard
Launch the web interface to view results interactively:
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

## 📊 Strategy & Performance

The system uses a `strategy.py` module to define entry/exit rules, potentially utilizing the `ml_strategy_model.pkl` for signal generation. Performance metrics are calculated in `performance.py` and visualized via `plot_performance.py`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
