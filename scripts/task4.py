import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime

# Modeling imports
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from scipy.optimize import minimize

# ---------------------------
# User-Configurable Parameters
# ---------------------------
CSV_PATH = "gmf_assets.csv"     
ASSET_TSLA = "TSLA"
ASSET_SPY = "SPY"
ASSET_BND = "BND"
DATE_COL = "Date"
PRICE_COL = "Close"

TRAIN_END = "2023-12-31"      # train up to this (inclusive)
TEST_START = "2024-01-01"     # test from this date
FORECAST_MONTHS = 6           # months for TSLA forecast horizon (6 or 12)
BUSINESS_DAYS_PER_MONTH = 21  # approx
FORECAST_STEPS = FORECAST_MONTHS * BUSINESS_DAYS_PER_MONTH

RISK_FREE_RATE_ANN = 0.02     # 2% annual risk-free rate (used for Sharpe)
SEQ_LENGTH = 60               # LSTM lookback window
LSTM_EPOCHS = 20
LSTM_BATCH = 32
LSTM_UNITS = 50

# ---------------------------
# Utility functions
# ---------------------------
def load_asset(df, asset):
    """Return DataFrame for given asset with Date index and Close column."""
    sub = df[df['Asset'] == asset].copy()
    sub[DATE_COL] = pd.to_datetime(sub[DATE_COL])
    sub = sub.sort_values(DATE_COL)
    sub = sub[[DATE_COL, PRICE_COL]].dropna()
    sub = sub.set_index(DATE_COL)
    sub = sub.asfreq('B')  # business day freq
    sub[PRICE_COL].fillna(method='ffill', inplace=True)
    return sub

def train_test_split(df, train_end=TRAIN_END):
    train = df.loc[:train_end].copy()
    test = df.loc[TEST_START:].copy()
    return train, test

def evaluate_rmse(true, pred):
    return math.sqrt(mean_squared_error(true, pred))

def get_daily_returns(df):
    return df[PRICE_COL].pct_change().dropna()

# ---------------------------
# TSLA forecasting: ARIMA
# ---------------------------
def forecast_tsla_arima(train_series, steps):
    """Auto_arima to pick order, fit ARIMA, and predict next 'steps' business days."""
    print("Running auto_arima for TSLA (may take some time)...")
    arima_auto = auto_arima(train_series, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
    order = arima_auto.order
    print("Selected ARIMA order:", order)
    model = ARIMA(train_series, order=order)
    fit = model.fit()
    # forecast steps ahead
    fc_res = fit.get_forecast(steps=steps)
    fc = fc_res.predicted_mean
    return fc, fit

# ---------------------------
# TSLA forecasting: LSTM
# ---------------------------
def create_sequences(arr, seq_len=SEQ_LENGTH):
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i, 0])
        y.append(arr[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X, y

def train_lstm(train_series, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, units=LSTM_UNITS):
    """Train LSTM on train_series (pd.Series of prices). Returns model and scaler."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1,1))
    X, y = create_sequences(scaled)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units // 1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0, callbacks=[es])
    return model, scaler

def lstm_iterative_forecast(model, scaler, recent_series, steps, seq_len=SEQ_LENGTH):
    """
    recent_series: pd.Series with latest historical prices (must include the last seq_len days)
    steps: number of future business days to predict
    """
    scaled = scaler.transform(recent_series.values.reshape(-1,1))
    seq = scaled[-seq_len:].tolist()
    preds = []
    for _ in range(steps):
        x = np.array(seq[-seq_len:]).reshape(1, seq_len, 1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        seq.append([p])
    preds = np.array(preds).reshape(-1,1)
    preds_inv = scaler.inverse_transform(preds).flatten()
    return pd.Series(preds_inv)

# ---------------------------
# Portfolio optimization helpers
# ---------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Return portfolio expected annual return, annual volatility."""
    # weights: array of weights summing to 1
    port_return = np.dot(weights, mean_returns)  # mean_returns should be annualized
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

def neg_sharpe(weights, mean_returns, cov_matrix, rf=RISK_FREE_RATE_ANN):
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    # negative Sharpe for minimization
    return - (p_ret - rf) / p_vol

def min_variance(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def generate_random_portfolios(num_portfolios, mean_returns, cov_matrix, rf=RISK_FREE_RATE_ANN):
    np.random.seed(42)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        w = np.random.random(len(mean_returns))
        w /= np.sum(w)
        weights_record.append(w)
        ret, vol = portfolio_performance(w, mean_returns, cov_matrix)
        sharpe = (ret - rf) / vol
        results[0,i] = ret
        results[1,i] = vol
        results[2,i] = sharpe
    return results, weights_record

# ---------------------------
# Main flow
# ---------------------------
def main():
    # Load CSV
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise SystemExit("CSV seems empty or not found. Update CSV_PATH.")
    # Prepare per-asset series
    tsla = load_asset(df, ASSET_TSLA)
    spy = load_asset(df, ASSET_SPY)
    bnd = load_asset(df, ASSET_BND)

    print(f"Loaded assets lengths: TSLA={len(tsla)}, SPY={len(spy)}, BND={len(bnd)}")
    # Align on common date index (intersection)
    common_index = tsla.index.intersection(spy.index).intersection(bnd.index)
    tsla = tsla.loc[common_index]
    spy = spy.loc[common_index]
    bnd = bnd.loc[common_index]

    # Train/test split for TSLA forecasting
    tsla_train, tsla_test = train_test_split(tsla, TRAIN_END)

    # --------------------
    # Forecast TSLA using ARIMA & LSTM, compare on tsla_test, pick best
    # --------------------
    print("\n=== TSLA Forecasting: ARIMA ===")
    arima_fc_future, arima_fit = forecast_tsla_arima(tsla_train[PRICE_COL], steps=FORECAST_STEPS)
    # For test set evaluation, forecast exactly len(tsla_test) ahead (one-shot)
    arima_fc_test_res = arima_fit.get_forecast(steps=len(tsla_test))
    arima_fc_test = pd.Series(arima_fc_test_res.predicted_mean.values, index=tsla_test.index)
    arima_rmse = evaluate_rmse(tsla_test[PRICE_COL].values, arima_fc_test.values)
    print(f"ARIMA RMSE on test set: {arima_rmse:.4f}")

    print("\n=== TSLA Forecasting: LSTM ===")
    lstm_model, lstm_scaler = train_lstm(tsla_train[PRICE_COL], epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, units=LSTM_UNITS)
    # Build test predictions (one-step sliding) to compare with test set
    # Prepare scaled full series (train+test)
    full_series = pd.concat([tsla_train[PRICE_COL], tsla_test[PRICE_COL]])
    scaled_full = lstm_scaler.transform(full_series.values.reshape(-1,1))
    preds_test_scaled = []
    actual_test_scaled = []
    for i in range(len(tsla_train), len(full_series)):
        if i - SEQ_LENGTH < 0:
            continue
        seq = scaled_full[i-SEQ_LENGTH:i]
        x = seq.reshape(1, SEQ_LENGTH, 1)
        p = lstm_model.predict(x, verbose=0)[0,0]
        preds_test_scaled.append(p)
        actual_test_scaled.append(scaled_full[i,0])
    if len(preds_test_scaled) > 0:
        preds_test = lstm_scaler.inverse_transform(np.array(preds_test_scaled).reshape(-1,1)).flatten()
        actual_test = lstm_scaler.inverse_transform(np.array(actual_test_scaled).reshape(-1,1)).flatten()
        # Align indices: predictions correspond to last len(preds_test) dates of tsla_test
        preds_index = tsla_test.index[-len(preds_test):]
        preds_series = pd.Series(preds_test, index=preds_index)
        actual_series = pd.Series(actual_test, index=preds_index)
        lstm_rmse = evaluate_rmse(actual_series.values, preds_series.values)
        print(f"LSTM one-step RMSE on overlapping test portion: {lstm_rmse:.4f}")
    else:
        # fallback: cannot evaluate LSTM on test
        preds_series = pd.Series(dtype=float)
        lstm_rmse = np.inf
        print("Not enough overlap to evaluate LSTM on test set. Setting RMSE to inf.")

    # Choose best model by RMSE on test (lower is better)
    if lstm_rmse < arima_rmse:
        best_model_name = "LSTM"
        print("\nSelected best model: LSTM (lower RMSE).")
        # produce future forecast using iterative LSTM
        recent_series = pd.concat([tsla_train[PRICE_COL], tsla_test[PRICE_COL]])  # use all available history
        tsla_future_forecast = lstm_iterative_forecast(lstm_model, lstm_scaler, recent_series, FORECAST_STEPS)
    else:
        best_model_name = "ARIMA"
        print("\nSelected best model: ARIMA (lower RMSE or LSTM unavailable).")
        tsla_future_forecast = arima_fc_future  # series indexed starting next business day after last date

    # Compute expected return for TSLA from the selected forecast:
    # Approach: compute average daily forecast returns over forecast horizon, then annualize
    # Convert forecast series to daily returns wrt last observed close
    last_price = tsla[PRICE_COL].iloc[-1]
    # If arima_fc_future produced index from after last date, we can compute pct changes
    tsla_fc_prices = tsla_future_forecast.copy()
    # ensure tsla_fc_prices is a pandas Series with consecutive business days
    if isinstance(tsla_fc_prices, pd.Series):
        # compute daily returns: first return relative to last real price
        first_return = (tsla_fc_prices.iloc[0] / last_price) - 1.0
        subsequent_returns = tsla_fc_prices.pct_change().fillna(first_return)
        avg_daily_return = subsequent_returns.mean()
    else:
        # fallback
        raise RuntimeError("Forecast produced invalid type for TSLA future prices.")
    tsla_expected_annual_return = (1 + avg_daily_return) ** 252 - 1  # approximate annualization
    print(f"\nTSLA expected annual return (from {best_model_name} forecast): {tsla_expected_annual_return:.4f} ({tsla_expected_annual_return*100:.2f}%)")

    # --------------------
    # Historical expected returns for SPY and BND (annualized)
    # --------------------
    spy_returns = get_daily_returns(spy)
    bnd_returns = get_daily_returns(bnd)

    spy_mean_daily = spy_returns.mean()
    bnd_mean_daily = bnd_returns.mean()

    spy_annual = (1 + spy_mean_daily) ** 252 - 1
    bnd_annual = (1 + bnd_mean_daily) ** 252 - 1

    print(f"SPY historical annual return: {spy_annual:.4f} ({spy_annual*100:.2f}%)")
    print(f"BND historical annual return: {bnd_annual:.4f} ({bnd_annual*100:.2f}%)")

    # --------------------
    # Build expected returns vector and covariance matrix (annualized)
    # --------------------
    # For covariance, compute daily returns for all three assets aligned on common index
    returns_df = pd.DataFrame({
        'TSLA': get_daily_returns(tsla),
        'SPY': get_daily_returns(spy),
        'BND': get_daily_returns(bnd)
    }).dropna()

    # daily mean returns used (for SPY, BND we used the historical mean; for TSLA, override)
    mean_daily_vector = np.array([avg_daily_return, spy_mean_daily, bnd_mean_daily])
    # Annualized expected returns:
    mean_annual_vector = (1 + mean_daily_vector) ** 252 - 1

    # Covariance matrix (annualized)
    cov_daily = returns_df.cov()
    cov_annual = cov_daily * 252

    print("\nExpected annual returns (TSLA, SPY, BND):")
    print(mean_annual_vector)

    print("\nAnnualized covariance matrix:")
    print(cov_annual)

    # --------------------
    # Efficient frontier: Monte Carlo + optimization
    # --------------------
    assets = ['TSLA', 'SPY', 'BND']
    num_portfolios = 30000
    results, weights = generate_random_portfolios(num_portfolios, mean_annual_vector, cov_annual, rf=RISK_FREE_RATE_ANN)
    rets = results[0]
    vols = results[1]
    sharpes = results[2]

    # Find max Sharpe in random sims
    max_sharpe_idx = np.argmax(sharpes)
    max_sharpe_ret = rets[max_sharpe_idx]
    max_sharpe_vol = vols[max_sharpe_idx]
    max_sharpe_weights = weights[max_sharpe_idx]

    # Find min vol in random sims
    min_vol_idx = np.argmin(vols)
    min_vol_ret = rets[min_vol_idx]
    min_vol_vol = vols[min_vol_idx]
    min_vol_weights = weights[min_vol_idx]

    # Now use optimization to find exact max-Sharpe and min-vol portfolios
    # constraints: sum(weights)=1, 0<=w<=1
    num_assets = len(assets)
    args = (mean_annual_vector, cov_annual)

    # Initial guess
    x0 = np.array(num_assets * [1. / num_assets])

    # Bounds and constraints
    bounds = tuple((0,1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Max Sharpe (min negative Sharpe)
    opt_sharpe = minimize(neg_sharpe, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_sharpe_weights = opt_sharpe.x
    opt_sharpe_ret, opt_sharpe_vol = portfolio_performance(opt_sharpe_weights, mean_annual_vector, cov_annual)
    opt_sharpe_ratio = (opt_sharpe_ret - RISK_FREE_RATE_ANN) / opt_sharpe_vol

    # Min Vol
    opt_minvol = minimize(min_variance, x0, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_minvol_weights = opt_minvol.x
    opt_minvol_ret, opt_minvol_vol = portfolio_performance(opt_minvol_weights, mean_annual_vector, cov_annual)
    opt_minvol_sharpe = (opt_minvol_ret - RISK_FREE_RATE_ANN) / opt_minvol_vol

    # --------------------
    # Print & plot results
    # --------------------
    print("\n=== Recommended Portfolios ===")
    print("\nRandom sim max Sharpe (approx):")
    print(f"Return: {max_sharpe_ret:.4f}, Vol: {max_sharpe_vol:.4f}")
    print("Weights (TSLA, SPY, BND):", np.round(max_sharpe_weights,4))

    print("\nRandom sim min Vol (approx):")
    print(f"Return: {min_vol_ret:.4f}, Vol: {min_vol_vol:.4f}")
    print("Weights (TSLA, SPY, BND):", np.round(min_vol_weights,4))

    print("\nOptimized Max Sharpe (tangency) portfolio:")
    print(f"Return: {opt_sharpe_ret:.4f}, Vol: {opt_sharpe_vol:.4f}, Sharpe: {opt_sharpe_ratio:.4f}")
    print("Weights (TSLA, SPY, BND):", np.round(opt_sharpe_weights,4))

    print("\nOptimized Min Vol portfolio:")
    print(f"Return: {opt_minvol_ret:.4f}, Vol: {opt_minvol_vol:.4f}, Sharpe: {opt_minvol_sharpe:.4f}")
    print("Weights (TSLA, SPY, BND):", np.round(opt_minvol_weights,4))

    # Choose recommendation:
    # Heuristic: if investor wants max risk-adjusted return -> choose tangency; if risk-averse -> choose min-vol
    # For this script, we'll present both and choose tangency as default recommendation.
    recommended_weights = opt_sharpe_weights
    rec_ret, rec_vol = opt_sharpe_ret, opt_sharpe_vol
    rec_sharpe = opt_sharpe_ratio

    print("\nFinal recommended portfolio (default: Max Sharpe):")
    for asset_name, w in zip(assets, recommended_weights):
        print(f"  {asset_name}: {w*100:.2f}%")
    print(f"Expected annual return: {rec_ret:.4f} ({rec_ret*100:.2f}%)")
    print(f"Expected annual volatility: {rec_vol:.4f} ({rec_vol*100:.2f}%)")
    print(f"Sharpe Ratio (rf={RISK_FREE_RATE_ANN}): {rec_sharpe:.4f}")

    # Plot Efficient Frontier
    plt.figure(figsize=(10,6))
    plt.scatter(vols, rets, c=sharpes, cmap='viridis', alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    # plot optimized points
    plt.scatter(opt_sharpe_vol, opt_sharpe_ret, marker='*', color='r', s=200, label='Max Sharpe')
    plt.scatter(opt_minvol_vol, opt_minvol_ret, marker='X', color='black', s=150, label='Min Vol')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title('Efficient Frontier (TSLA, SPY, BND)')
    plt.legend()
    plt.grid(True)
    plt.savefig("efficient_frontier.png", dpi=150)
    plt.show()

    # Save recommended weights and data
    pd.DataFrame({
        'Asset': assets,
        'Recommended_Weight': recommended_weights
    }).to_csv("recommended_portfolio_weights.csv", index=False)

    print("\nSaved 'efficient_frontier.png' and 'recommended_portfolio_weights.csv'")

if __name__ == "__main__":
    main()
