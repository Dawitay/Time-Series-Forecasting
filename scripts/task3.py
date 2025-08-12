import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import timedelta

# Modeling imports
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------
# User parameters
# -------------------------------
CSV_PATH = "gmf_assets.csv"   
ASSET = "TSLA"
DATE_COL = "Date"
CLOSE_COL = "Close"

# Forecast horizon: months to forecast (6 or 12). You can change this.
FORECAST_MONTHS = 6

# LSTM hyperparameters (tunable)
SEQ_LENGTH = 60   # look-back window in days
LSTM_UNITS = 50
LSTM_EPOCHS = 30
LSTM_BATCH = 32

# Random seed for reproducibility
RANDOM_SEED = 42

# -------------------------------
# Utilities
# -------------------------------
def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_and_prepare(csv_path, asset):
    """Load CSV, filter by asset, sort by date, return DataFrame with Date index."""
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    df = df[df['Asset'] == asset].copy()
    df.sort_values(DATE_COL, inplace=True)
    df = df[[DATE_COL, CLOSE_COL]].dropna()
    df.set_index(DATE_COL, inplace=True)
    # Ensure daily frequency by forward-filling missing business days if needed:
    df = df.asfreq('B')  # business day frequency
    df[CLOSE_COL].fillna(method='ffill', inplace=True)
    return df

def chronological_split(df, train_end='2023-12-31'):
    """Split df chronologically into train (<= train_end) and test (> train_end)."""
    train = df.loc[:train_end].copy()
    test = df.loc[train_end + " 00:00:00":].copy() if isinstance(train_end, str) else df.loc[:train_end].copy()
    # If the above slicing returns empty test (e.g., if dates outside), do fallback:
    test = df.loc['2024-01-01':].copy()
    return train, test

def evaluate(true, pred, label="Model"):
    mae = mean_absolute_error(true, pred)
    rmse = math.sqrt(mean_squared_error(true, pred))
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# -------------------------------
# ARIMA: train, forecast, confidence intervals
# -------------------------------
def train_arima(train_series):
    """Use auto_arima to find order, fit ARIMA model and return fitted object."""
    print("Running auto_arima (this may take some time)...")
    arima_auto = auto_arima(train_series, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
    order = arima_auto.order
    print("auto_arima selected order:", order)
    # Fit statsmodels ARIMA using the found order on the entire train
    model = ARIMA(train_series, order=order)
    model_fit = model.fit()
    return model_fit

def arima_forecast_with_ci(model_fit, steps, alpha=0.05):
    """Return forecast (point) and confidence intervals from ARIMA statsmodels object."""
    # Using get_forecast from statsmodels
    forecast_res = model_fit.get_forecast(steps=steps)
    forecast = forecast_res.predicted_mean
    ci = forecast_res.conf_int(alpha=alpha)
    # ci is dataframe with lower and upper columns for the mean
    return forecast, ci

# -------------------------------
# LSTM: prepare data, train, iterative forecast, and approximate CI
# -------------------------------
def create_sequences(data_arr, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in range(seq_length, len(data_arr)):
        X.append(data_arr[i-seq_length:i, 0])
        y.append(data_arr[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X, y

def build_and_train_lstm(X_train, y_train, units=LSTM_UNITS, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH):
    """Builds a small LSTM and trains it."""
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units//1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[es])
    return model

def lstm_iterative_forecast(model, scaler, full_series, train_len, seq_length, steps):
    """
    Create iterative (recursive) forecast using last seq_length points from full_series (scaled),
    predicting step-by-step for 'steps' periods.
    """
    scaled = scaler.transform(full_series.values.reshape(-1,1))  # scaled total series
    # start sequence = last seq_length points of training portion
    start_ix = train_len - seq_length
    seq = scaled[start_ix:train_len].tolist()  # list of seq_length arrays
    preds_scaled = []
    for _ in range(steps):
        x = np.array(seq[-seq_length:]).reshape(1, seq_length, 1)
        pred_scaled = model.predict(x, verbose=0)[0][0]
        preds_scaled.append(pred_scaled)
        seq.append([pred_scaled])  # append to sequence for next step
    preds_scaled = np.array(preds_scaled).reshape(-1,1)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    return preds, preds_scaled.flatten()

# -------------------------------
# Plotting utilities
# -------------------------------
def plot_forecasts(train, test, arima_forecast, arima_ci, lstm_forecast, output_path_prefix="forecast"):
    sns.set(style="darkgrid")
    plt.figure(figsize=(14,6))
    plt.plot(train.index, train[CLOSE_COL], label="Train", color="black", lw=1)
    plt.plot(test.index, test[CLOSE_COL], label="Test (actual)", color="blue", lw=1)

    # ARIMA points
    plt.plot(arima_forecast.index, arima_forecast.values, label="ARIMA forecast", color="orange")
    # ARIMA CI
    plt.fill_between(arima_ci.index,
                     arima_ci.iloc[:,0],
                     arima_ci.iloc[:,1],
                     color="orange", alpha=0.2, label="ARIMA 95% CI")

    # LSTM
    if lstm_forecast is not None:
        # align LSTM with test index (it starts at test.index[0])
        plt.plot(test.index, lstm_forecast, label="LSTM forecast", color="green")
    plt.title(f"Forecasts for {ASSET} - {FORECAST_MONTHS} months ahead")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_combined.png", dpi=150)
    plt.show()

def plot_lstm_ci(test_index, lstm_forecast, lstm_lower, lstm_upper, prefix="lstm"):
    plt.figure(figsize=(14,6))
    plt.plot(test_index, lstm_forecast, label="LSTM forecast", color="green")
    plt.fill_between(test_index, lstm_lower, lstm_upper, color="green", alpha=0.2, label="LSTM approx 95% CI")
    plt.plot(test_index, lstm_forecast, color="green")
    plt.title("LSTM Forecast and Approx. 95% CI")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{prefix}_ci.png", dpi=150)
    plt.show()

# -------------------------------
# Main pipeline
# -------------------------------
def main():
    set_seed()
    print("Loading data...")
    df = load_and_prepare(CSV_PATH, ASSET)
    if df.empty:
        raise SystemExit("No data found for asset. Check CSV_PATH and Asset name.")
    print(f"Data loaded: {len(df)} rows from {df.index.min().date()} to {df.index.max().date()}")

    # Chronological split: train up to 2023-12-31, test 2024 onwards
    train, test = chronological_split(df, train_end='2023-12-31')
    if test.empty:
        print("Warning: test set is empty (no data after 2023-12-31). Adjust your CSV or date split.")
    print(f"Train size: {len(train)}, Test size: {len(test)}")

    # ---------------- ARIMA
    print("\n=== ARIMA Model Training & Forecasting ===")
    arima_fit = train_arima(train[CLOSE_COL])
    # Forecast horizon in business days: convert months -> approx business days
    months = FORECAST_MONTHS
    # approximate business days per month ~21
    steps = months * 21
    print(f"Forecasting {months} months -> approx {steps} business days ahead.")
    arima_forecast, arima_ci = arima_forecast_with_ci(arima_fit, steps=steps, alpha=0.05)
    # Build index for forecast: continue business days from last date in df
    last_date = df.index.max()
    forecast_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)
    arima_forecast.index = forecast_index
    arima_ci.index = forecast_index

    # ---------------- LSTM
    print("\n=== LSTM Training & Forecasting ===")
    # Prepare scaler on full close values (we will simulate forecasting from train & full series)
    scaler = MinMaxScaler()
    scaler.fit(train[CLOSE_COL].values.reshape(-1,1))

    # Prepare train sequences for LSTM using train only
    scaled_train_arr = scaler.transform(train[CLOSE_COL].values.reshape(-1,1))
    X_train, y_train = create_sequences(scaled_train_arr, SEQ_LENGTH)
    # reshape for LSTM: [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # Build & train LSTM
    lstm_model = build_and_train_lstm(X_train, y_train, units=LSTM_UNITS, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH)

    # Iterative forecast using entire series (train + recent history)
    full_series = pd.concat([train, test])  # use available historical up to test end for creating iterative forecasts
    # ensure full_series sorted and uses CLOSE_COL
    full_series = full_series[[CLOSE_COL]]
    # Use the index for forecast mapping: we want forecast index same as ARIMA
    lstm_preds, lstm_preds_scaled = lstm_iterative_forecast(lstm_model, scaler, full_series[CLOSE_COL], len(train), SEQ_LENGTH, steps)
    lstm_forecast_series = pd.Series(lstm_preds, index=forecast_index)

    # ---------------- Confidence intervals for LSTM (approx)
    # Compute residual std on test set to estimate error magnitude for CI
    print("\nEstimating LSTM residuals on test set for approximate CI...")
    # For residuals, predict on test using sliding window approach for the portion that overlaps test
    # Build scaled full array
    scaled_full = scaler.transform(full_series[CLOSE_COL].values.reshape(-1,1))
    # We'll compute one-step-ahead predictions for each point in the test portion that can be predicted from past seq_length
    preds_test = []
    actual_test_vals = []
    for i in range(len(train), len(full_series)):
        if i - SEQ_LENGTH < 0:
            continue
        seq = scaled_full[i-SEQ_LENGTH:i]
        x = seq.reshape(1, SEQ_LENGTH, 1)
        p = lstm_model.predict(x, verbose=0)[0][0]
        preds_test.append(p)
        actual_test_vals.append(scaled_full[i][0])
    if len(preds_test) > 0:
        preds_test = np.array(preds_test).reshape(-1,1)
        actual_test_vals = np.array(actual_test_vals).reshape(-1,1)
        preds_test_inv = scaler.inverse_transform(preds_test).flatten()
        actual_test_inv = scaler.inverse_transform(actual_test_vals).flatten()
        residuals = actual_test_inv - preds_test_inv
        resid_std = residuals.std(ddof=1)
        resid_rmse = math.sqrt(mean_squared_error(actual_test_inv, preds_test_inv))
        print(f"LSTM one-step residual std (approx): {resid_std:.4f}, RMSE: {resid_rmse:.4f}")
    else:
        resid_std = np.std(train[CLOSE_COL].pct_change().dropna()) * train[CLOSE_COL].mean()  # fallback heuristic
        print("No overlap for residual estimation; using heuristic residual std:", resid_std)

    # Build 95% CI using z=1.96
    z = 1.96
    lstm_lower = lstm_forecast_series - z * resid_std
    lstm_upper = lstm_forecast_series + z * resid_std

    # ---------------- Evaluate on available test set (if present)
    print("\n=== Evaluation on historical test set (if available) ===")
    # Align ARIMA forecast to test horizon (if user wants compare on test). But ARIMA forecast was produced for future beyond last_date.
    # We should also compute forecasts covering the test period (2024-2025) to compare. Let's do that:
    # Re-forecast ARIMA for test horizon length:
    test_steps = len(test)
    if test_steps > 0:
        arima_test_forecast_res = arima_fit.get_forecast(steps=test_steps)
        arima_test_forecast = pd.Series(arima_test_forecast_res.predicted_mean.values, index=test.index)
        arima_test_ci = arima_test_forecast_res.conf_int(alpha=0.05)
        # LSTM test predictions (one-step) already computed above if preds_test exists
        if len(preds_test) > 0:
            # preds_test_inv & actual_test_inv from above
            # create series aligned to the portion of test we predicted
            start_idx = test.index[0] if len(preds_test) == len(test) else test.index[len(test) - len(preds_test)]
            lstm_test_pred_series = pd.Series(preds_test_inv.flatten(), index=test.index[-len(preds_test):])
            # Evaluate both on overlap
            print("\nARIMA on test set:")
            evaluate(test.loc[arima_test_forecast.index][CLOSE_COL].values, arima_test_forecast.values, "ARIMA (test)")
            print("\nLSTM one-step on test set:")
            evaluate(actual_test_inv.flatten(), preds_test_inv.flatten(), "LSTM (one-step test)")
        else:
            print("Not enough test data to evaluate LSTM one-step predictions.")
            print("\nARIMA on test set:")
            evaluate(test[CLOSE_COL].values, arima_test_forecast.values, "ARIMA (test)")
    else:
        print("No historical test period in CSV to evaluate against (no data from 2024 onwards).")

    # ---------------- Plot combined forecast results
    print("\nPlotting forecasts and CIs...")
    # For plotting combined (train + test) we show train, test (if exists), and future forecasts (ARIMA & LSTM)
    # Use arima_forecast (future) and arima_ci (future), and lstm_forecast_series + its CI
    plot_forecasts(train, test, arima_forecast, arima_ci, lstm_forecast_series, output_path_prefix=f"{ASSET}_forecast_{months}m")
    plot_lstm_ci(arima_forecast.index, lstm_forecast_series.values, lstm_lower.values, lstm_upper.values, prefix=f"{ASSET}_lstm_{months}m")

    # -------------------------------
    # Forecast analysis & interpretation (printed)
    # -------------------------------
    print("\n=== Forecast Analysis & Interpretation ===")
    # Trend analysis (simple slope on ARIMA forecast)
    arima_slope = (arima_forecast.iloc[-1] - arima_forecast.iloc[0]) / len(arima_forecast)
    lstm_slope = (lstm_forecast_series.iloc[-1] - lstm_forecast_series.iloc[0]) / len(lstm_forecast_series)

    print(f"ARIMA forecast trend slope (avg daily change over horizon): {arima_slope:.6f}")
    print(f"LSTM forecast trend slope (avg daily change over horizon): {lstm_slope:.6f}")

    # Confidence interval width analysis
    arima_ci_width = (arima_ci.iloc[:,1] - arima_ci.iloc[:,0]).values
    lstm_ci_width = (lstm_upper - lstm_lower).values

    # Report initial vs final CI widths (normalized)
    print(f"\nARIMA CI width at start: {arima_ci_width[0]:.4f}, at end: {arima_ci_width[-1]:.4f}")
    print(f"LSTM approx CI width: constant (based on residual std) ≈ {lstm_ci_width[0]:.4f}")

    # Interpretations
    print("\nInterpretation (concise):")
    # Trend:
    if arima_slope > 0 and lstm_slope > 0:
        print("- Both models indicate an overall upward trend over the forecast horizon.")
    elif arima_slope < 0 and lstm_slope < 0:
        print("- Both models indicate an overall downward trend.")
    else:
        print("- Models disagree on short-term trend; inspect plots and CI.")

    # Volatility & CI:
    print("- ARIMA's CI typically widens with horizon (uncertainty increases further into the future).")
    print("- LSTM CI here is approximated from one-step residuals and is roughly constant; this is a rough heuristic.")
    print("  Interpretation: Long-horizon LSTM CI here underestimates structural uncertainty vs ARIMA's model-based CI.")

    # Opportunities & Risks:
    print("\nMarket opportunities & risks (high-level):")
    # Use simple rule: if forecast mean increases > some threshold over horizon -> opportunity
    arima_pct_change = (arima_forecast.iloc[-1] - arima_forecast.iloc[0]) / arima_forecast.iloc[0] * 100
    lstm_pct_change = (lstm_forecast_series.iloc[-1] - lstm_forecast_series.iloc[0]) / lstm_forecast_series.iloc[0] * 100
    print(f"- ARIMA forecast change over horizon: {arima_pct_change:.2f}%")
    print(f"- LSTM forecast change over horizon: {lstm_pct_change:.2f}%")

    if arima_pct_change > 5 or lstm_pct_change > 5:
        print("  Opportunity: Models suggest significant upside over the horizon (>5%). Consider tail-risk protection and position sizing.")
    elif arima_pct_change < -5 or lstm_pct_change < -5:
        print("  Risk: Models suggest significant expected decline (>5%). Consider hedges or reducing exposure.")
    else:
        print("  Modest expected movement: focus on risk management and short-term monitoring.")

    print("\nNote on reliability:")
    print("- Forecasts are model outputs and should be used as one input among many.")
    print("- Confidence intervals widen with horizon indicating increasing uncertainty.")
    print("- LSTM CI method here is heuristic — for rigorous intervals consider quantile regression, bootstrapping, ensemble methods, or Bayesian neural nets.")

    # Save numeric forecasts to CSV
    print("\nSaving numeric forecasts to CSV...")
    out_df = pd.DataFrame({
        "ARIMA_forecast": arima_forecast,
        "ARIMA_lower": arima_ci.iloc[:,0],
        "ARIMA_upper": arima_ci.iloc[:,1],
        "LSTM_forecast": lstm_forecast_series,
        "LSTM_lower": lstm_lower,
        "LSTM_upper": lstm_upper
    })
    out_df.to_csv(f"{ASSET}_future_forecasts_{months}m.csv")
    print(f"Forecasts saved to: {ASSET}_future_forecasts_{months}m.csv")

if __name__ == "__main__":
    main()
