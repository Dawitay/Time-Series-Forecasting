"""
task5_backtest.py

Task 5: Backtesting strategy vs benchmark for the last year:
- Backtest period default: 2024-08-01 to 2025-07-31
- Benchmark: static 60% SPY / 40% BND
- Strategy: initial optimized weights (Max Sharpe) computed from historical data (Task 4 flow)
- Rebalance modes: 'hold' or 'monthly' (rebalancing to initial weights each month)
- Outputs: cumulative returns plot, final total returns, annualized Sharpe ratios, CSV of daily portfolio values

Author: ChatGPT (GPT-5 Thinking mini)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# -----------------------
# User Config
# -----------------------
CSV_PATH = "gmf_assets.csv"   
DATE_COL = "Date"
PRICE_COL = "Close"
ASSETS = ["TSLA", "SPY", "BND"]

# Backtest window (inclusive)
BACKTEST_START = "2024-08-01"
BACKTEST_END = "2025-07-31"

# Historical window used to compute optimization weights (train era)
# We'll use all data up to start of backtest - 1 day for constructing expected returns & cov
HIST_END = (pd.to_datetime(BACKTEST_START) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

# Benchmark weights
BENCH_WEIGHTS = {"SPY": 0.60, "BND": 0.40}

# Rebalance mode: "hold" or "monthly"
REBALANCE_MODE = "monthly"  # set to "hold" for buy-and-hold

# Risk-free for Sharpe calculation (annual)
RISK_FREE_ANN = 0.02
TRADING_DAYS = 252

# Monte Carlo random portfolios for initial search (optional)
NUM_RANDOM_PORTS = 5000

# -----------------------
# Helpers
# -----------------------
def load_asset(df, asset):
    """Return pandas Series of Close prices indexed by Date (business days)"""
    sub = df[df['Asset'] == asset].copy()
    sub[DATE_COL] = pd.to_datetime(sub[DATE_COL])
    sub = sub.sort_values(DATE_COL)
    sub = sub[[DATE_COL, PRICE_COL]].dropna()
    sub = sub.set_index(DATE_COL)
    # align business-day freq and forward fill missing prices
    sub = sub.asfreq('B')
    sub[PRICE_COL].fillna(method='ffill', inplace=True)
    return sub[PRICE_COL]

def calculate_daily_returns(series):
    return series.pct_change().dropna()

def annualize_return(daily_ret_mean):
    return (1 + daily_ret_mean) ** TRADING_DAYS - 1

def portfolio_performance(weights, mean_returns, cov_matrix):
    """weights: array, mean_returns: annual returns array, cov_matrix: annualized"""
    port_ret = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_ret, port_vol

def neg_sharpe(weights, mean_returns, cov_matrix, rf=RISK_FREE_ANN):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (ret - rf) / vol

# -----------------------
# Step 1: Load CSV and prepare series
# -----------------------
df = pd.read_csv(CSV_PATH)
if df.empty:
    raise SystemExit("CSV empty or not found. Update CSV_PATH variable.")

tsla_ser = load_asset(df, "TSLA")
spy_ser = load_asset(df, "SPY")
bnd_ser = load_asset(df, "BND")

# Align by intersection of dates
common_idx = tsla_ser.index.intersection(spy_ser.index).intersection(bnd_ser.index)
tsla_ser = tsla_ser.loc[common_idx]
spy_ser = spy_ser.loc[common_idx]
bnd_ser = bnd_ser.loc[common_idx]

# -----------------------
# Step 2: Define historical sample to compute expected returns & covariances
# -----------------------
hist_end_date = pd.to_datetime(HIST_END)
hist_df = pd.DataFrame({
    "TSLA": tsla_ser.loc[:hist_end_date],
    "SPY": spy_ser.loc[:hist_end_date],
    "BND": bnd_ser.loc[:hist_end_date]
}).dropna()

if hist_df.empty:
    raise SystemExit("No historical data before BACKTEST_START. Adjust HIST_END or CSV.")

# Daily returns for historical sample
hist_returns = hist_df.pct_change().dropna()

# Annualized covariance matrix
cov_annual = hist_returns.cov() * TRADING_DAYS

# Historical mean daily returns
mean_daily = hist_returns.mean()
mean_annual = (1 + mean_daily) ** TRADING_DAYS - 1

# -----------------------
# Step 3: Obtain expected TSLA return from forecast model
# We'll use a simple ARIMA forecast to produce TSLA expected daily return over future horizon,
# then convert that to an annual expected return to replace historcal TSLA mean.
# (This follows Task 4 logic: forecast-based expected return for TSLA)
# -----------------------
# Train ARIMA on historical TSLA prices up to HIST_END, forecast next N days (we use 21*6=~126 days as sample)
forecast_steps_sample = 126  # used only to estimate average daily forecast return; not the backtest horizon
tsla_train_prices = tsla_ser.loc[:hist_end_date]

# Auto-arima pick (might take time)
print("Fitting auto_arima for TSLA to get expected returns (this may take a minute)...")
from pmdarima import auto_arima
arima_auto = auto_arima(tsla_train_prices, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
order = arima_auto.order
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(tsla_train_prices, order=order).fit()
fc_res = model.get_forecast(steps=forecast_steps_sample)
tsla_forecast_prices = fc_res.predicted_mean
# Compute average daily forecast return (first return relative to last train price, then subsequent pct_changes)
last_actual = tsla_train_prices.iloc[-1]
first_ret = (tsla_forecast_prices.iloc[0] / last_actual) - 1.0
subsequent = tsla_forecast_prices.pct_change().fillna(first_ret)
avg_daily_forecast_return = subsequent.mean()
tsla_expected_annual = (1 + avg_daily_forecast_return) ** TRADING_DAYS - 1
print(f"TSLA expected annual return (from ARIMA forecast sample): {tsla_expected_annual:.4f} ({tsla_expected_annual*100:.2f}%)")

# Replace TSLA annualized expected return in mean_annual
mean_annual_fore = mean_annual.copy()
mean_annual_fore['TSLA'] = tsla_expected_annual

# -----------------------
# Step 4: Optimize to get initial optimal weights (Max Sharpe)
# -----------------------
assets = ["TSLA","SPY","BND"]
mean_vect = mean_annual_fore.values  # order matches columns
cov_matrix = cov_annual.values

# optimization: bounds 0..1, sum=1
num_assets = len(assets)
x0 = np.array([1.0/num_assets]*num_assets)
bounds = tuple((0,1) for _ in range(num_assets))
cons = ({'type':'eq', 'fun': lambda x: np.sum(x)-1})

opt = minimize(neg_sharpe, x0, args=(mean_vect, cov_matrix), method='SLSQP', bounds=bounds, constraints=cons)
opt_weights = opt.x
opt_ret, opt_vol = portfolio_performance(opt_weights, mean_vect, cov_matrix)
opt_sharpe = (opt_ret - RISK_FREE_ANN) / opt_vol

print("\nInitial optimized weights (Max Sharpe):")
for a,w in zip(assets, opt_weights):
    print(f"  {a}: {w*100:.2f}%")
print(f"Expected annual return: {opt_ret:.4f}, Volatility: {opt_vol:.4f}, Sharpe: {opt_sharpe:.4f}")

# -----------------------
# Step 5: Backtest setup (get backtest price series)
# -----------------------
bt_start = pd.to_datetime(BACKTEST_START)
bt_end = pd.to_datetime(BACKTEST_END)

price_df = pd.DataFrame({
    "TSLA": tsla_ser.loc[bt_start:bt_end],
    "SPY": spy_ser.loc[bt_start:bt_end],
    "BND": bnd_ser.loc[bt_start:bt_end]
}).dropna()

if price_df.empty:
    raise SystemExit("No data in the backtest window. Check BACKTEST_START/END and CSV.")

# Daily returns during backtest
daily_ret_bt = price_df.pct_change().dropna()

# -----------------------
# Step 6: Simulate portfolios
# -----------------------
def simulate_hold(weights, daily_returns):
    """
    Simple buy-and-hold: start with 1.0 portfolio value, compute daily portfolio returns using fixed weights.
    Returns Series of cumulative portfolio values indexed same as daily_returns index.
    """
    # portfolio daily returns = dot(weights, daily_returns.T)
    port_daily = daily_returns.dot(weights)
    cum = (1 + port_daily).cumprod()
    return cum

def simulate_monthly_rebalance(weights, daily_returns):
    """
    Rebalance monthly to target weights. Start portfolio value =1.
    Implementation:
      - For each business day, apply that day's returns to holdings
      - At first trading day of each month (or user-defined schedule), rebalance holdings to target weights
    Returns cumulative portfolio value series.
    """
    dates = daily_returns.index
    port_value = 1.0
    holdings = None  # number of asset units, relative to price at last rebalance
    values = []
    # Pre-calc price series
    price_slice = price_df.loc[daily_returns.index]
    # Identify first trading day of each month
    month_starts = sorted(list({d.to_period('M').to_timestamp() for d in dates}))
    # Simpler approach: treat first available business day per month within dates
    month_first_days = {}
    for d in dates:
        pdm = d.to_period('M').to_timestamp()
        if pdm not in month_first_days:
            month_first_days[pdm] = d
    month_first_dates = set(month_first_days.values())

    # Initialize holdings at the first day with rebalance to weights
    current_date = dates[0]
    # initial prices for holdings (use close price at first day's price_df)
    init_prices = price_slice.iloc[0].values
    # buy asset units proportional to weights: units = (weight * port_value) / price
    holdings = (weights * port_value) / init_prices

    # iterate days
    for i, d in enumerate(dates):
        # apply returns for day d to holdings value
        todays_prices = price_slice.loc[d].values
        port_value = np.dot(holdings, todays_prices)
        values.append(port_value)

        # check next day (rebalancing happens at start of month, i.e., after daily update at day which is month_first? We'll rebalance at next day's open)
        # We'll do a simple rule: if d is the last business day of the month, rebalance at next day open (i+1)
        # Simpler: rebalance at every month_first_days[d] occurrence: but we already set initial holdings at first available date.
        next_i = i+1
        if next_i < len(dates):
            next_date = dates[next_i]
            # if next_date is first business day of a month (i.e., next_date in month_first_days values), then rebalance at next_date using closing price of next_date
            if next_date in month_first_dates:
                # compute holdings at next_date using port_value and next_date's prices
                next_prices = price_slice.loc[next_date].values
                holdings = (weights * port_value) / next_prices
    cum = pd.Series(values, index=dates)
    return cum

# Strategy portfolio using initial optimized weights
strategy_weights = opt_weights  # order [TSLA, SPY, BND]

if REBALANCE_MODE == "hold":
    cum_strategy = simulate_hold(strategy_weights, daily_ret_bt)
elif REBALANCE_MODE == "monthly":
    cum_strategy = simulate_monthly_rebalance(strategy_weights, daily_ret_bt)
else:
    raise ValueError("REBALANCE_MODE must be 'hold' or 'monthly'")

# Benchmark portfolio: static 60/40 SPY/BND (TSLA 0)
bench_weights_arr = np.array([0.0, BENCH_WEIGHTS["SPY"], BENCH_WEIGHTS["BND"]])
cum_benchmark = simulate_hold(bench_weights_arr, daily_ret_bt)  # usually benchmark is hold

# -----------------------
# Step 7: Performance metrics
# -----------------------
def total_return(cum_series):
    return cum_series.iloc[-1] - 1.0

def annualized_sharpe(cum_series, rf=RISK_FREE_ANN):
    # compute daily returns from cumulative series
    daily = cum_series.pct_change().dropna()
    mean_excess = daily.mean() - (rf / TRADING_DAYS)
    std = daily.std(ddof=1)
    if std == 0:
        return np.nan
    sharpe_ann = (mean_excess / std) * math.sqrt(TRADING_DAYS)
    return sharpe_ann

strat_total = total_return(cum_strategy)
bench_total = total_return(cum_benchmark)
strat_sharpe = annualized_sharpe(cum_strategy)
bench_sharpe = annualized_sharpe(cum_benchmark)

print("\n=== Backtest Summary ===")
print(f"Backtest window: {BACKTEST_START} to {BACKTEST_END}")
print(f"Strategy rebal mode: {REBALANCE_MODE}")
print(f"Initial strategy weights (Max Sharpe): TSLA {opt_weights[0]:.3f}, SPY {opt_weights[1]:.3f}, BND {opt_weights[2]:.3f}")
print(f"Strategy total return: {strat_total*100:.2f}%")
print(f"Benchmark (60/40 SPY/BND) total return: {bench_total*100:.2f}%")
print(f"Strategy annualized Sharpe: {strat_sharpe:.4f}")
print(f"Benchmark annualized Sharpe: {bench_sharpe:.4f}")

# -----------------------
# Step 8: Plot results
# -----------------------
plt.figure(figsize=(12,6))
plt.plot(cum_strategy.index, cum_strategy.values, label='Strategy (opt weights)', lw=2)
plt.plot(cum_benchmark.index, cum_benchmark.values, label='Benchmark 60%SPY/40%BND', lw=2)
plt.title(f"Backtest Cumulative Returns: {BACKTEST_START} to {BACKTEST_END}")
plt.xlabel("Date")
plt.ylabel("Cumulative Return (Start=1.0)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("backtest_cumulative_returns.png", dpi=150)
plt.show()

# Save daily results to CSV
out_df = pd.DataFrame({
    "Strategy_Cumulative": cum_strategy,
    "Benchmark_Cumulative": cum_benchmark
})
out_df.to_csv("task5_backtest_results.csv", index_label="Date")

# -----------------------
# Step 9: Brief conclusion printed
# -----------------------
print("\nConclusion:")
if strat_total > bench_total:
    print("- Strategy outperformed the benchmark in total return over this backtest period.")
else:
    print("- Strategy underperformed the benchmark in total return over this backtest period.")

if strat_sharpe > bench_sharpe:
    print("- Strategy had higher risk-adjusted returns (Sharpe) than the benchmark.")
else:
    print("- Strategy had lower risk-adjusted returns (Sharpe) than the benchmark.")

print("\nNotes & Caveats:")
print("- This is a simplified backtest. Rebalancing uses initial weights only (no rolling forecast/optimization).")
print("- Transaction costs, slippage, borrowing costs, and taxes are NOT modeled.")
print("- Consider running a more sophisticated backtest with monthly re-optimization, transaction cost model, and walk-forward validation for production use.")
