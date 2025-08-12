## 📊 Task 1: Exploratory Data Analysis of Tesla, SPY, and BND

This task involved performing a comprehensive Exploratory Data Analysis (EDA) on financial time-series data consisting of Tesla (TSLA), SPY (S\&P 500 ETF), and BND (Bond ETF). The dataset included columns such as `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, and `Asset`.

### Key Analyses Performed:

* **Time Series Visualization:**
  Plotted closing prices over time to observe long-term trends. TSLA exhibited a more volatile and upward-trending behavior compared to SPY and BND.

* **Daily Returns Analysis:**
  Computed daily percentage change in closing prices. Tesla had significantly higher return volatility than the broader market (SPY) and bonds (BND).

* **Volatility and Rolling Statistics:**
  Applied rolling means and standard deviations (20-day window) to analyze short-term trends and price fluctuations. Tesla again showed the highest variability.

* **Outlier Detection:**
  Identified days with extreme returns to detect anomalies or market shocks.

* **Stationarity Testing (ADF Test):**
  Conducted Augmented Dickey-Fuller test on both prices and returns. Prices were non-stationary, while daily returns were stationary — indicating the need for differencing when modeling price series.

* **Risk Metrics:**

  * **Value at Risk (VaR):** Estimated the maximum expected loss over a 1-day horizon with 95% confidence.
  * **Sharpe Ratio:** Calculated historical risk-adjusted return to assess performance per unit of risk.

### 📌 Insights:

* Tesla's stock price shows strong upward momentum but is accompanied by high volatility.
* Daily returns are noisy and show frequent fluctuations, particularly for TSLA.
* TSLA carries the highest financial risk (highest VaR and standard deviation), but also has the potential for higher returns, as shown by its Sharpe Ratio.

---
