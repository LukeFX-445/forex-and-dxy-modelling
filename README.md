# forex-and-dxy-modelling

This repository contains two complementary models: (i) a cross-sectional engine for currency-pair selection, and (ii) a short-horizon directional model for the U.S. Dollar Index (DXY). Both models are research prototypes that fetch public market data, perform regime-aware volatility and drift estimation, run Monte Carlo simulations, and output diagnostic files. They are meant for transparent analysis and reproducible research, not for live trading.

---

## 1. Forex Pair Selection Model

This model ranks currencies by structural context, volatility regime, and cross-sectional divergence to identify asymmetric opportunities. It combines market-microstructure heuristics with robust statistical filtering.

### 1.1 Data Sources

- Spot FX from Yahoo Finance with `auto_adjust=False` to preserve raw price dynamics (`EURUSD=X`, `GBPUSD=X`, `USDJPY=X`, `USDCHF=X`, `USDCAD=X`, `UUP` for USD).
- Extended notebooks add `AUDUSD=X`, `NZDUSD=X`, and `DX=F` (DXY futures) for an 8-currency basket.
- Optional synthetic USD baskets are constructed from major legs using DXY-style weights for diagnostic comparison.
- Macroeconomic notebooks scrape interest rates, GDP growth, and inflation from TradingEconomics with 403-safe headers, and economic calendars from investpy and ForexFactory.

### 1.2 Volatility Filtering

- Realised measures include True Range ADR(14) and ADR% (ADR divided by level) for scale-free comparisons.
- Additional realised volatility comparators include the Parkinson estimator (10-day) and EWMA (20-day).
- Conditional volatility models include GARCH(1,1), EGARCH, and GJR-GARCH with Student-t innovations and `rescale=False`.
- In enhanced notebooks, the best model is chosen by AIC; if fitting fails, a local realised sigma fallback is used.
- Regime detection via a two-state Markov switching model adapts volatility forecasts.

### 1.3 Drift and Directional Filtering

- Carry drift is derived from interest-rate differentials versus USD and converted into daily fractions.
- Macro drift is derived from GDP and inflation differentials where available.
- Sentiment drift uses VADER on FXStreet RSS feeds and is intentionally scaled down so it remains a mild tilt rather than a dominant driver.
- Technical drift is built from momentum (5-, 20-, and 60-day log returns), RSI(14), Bollinger band position, and an ADX-style proxy.
- Regime awareness means signal weights adapt to volatility conditions, such as ADR expansion or an elevated 10/50-day volatility ratio.

### 1.4 Monte Carlo Simulation and Ranking

- Forward price paths are simulated under a jump-diffusion GBM with fat-tailed returns and rare jumps.
- Simulations incorporate volatility shocks, carry differentials, regime-dependent volatility, and event-week boosts to sigma.
- Enhanced notebooks run simulations across 1-day, 1-week, 2-week, and 1-month horizons.
- For each currency pair, the simulation produces expected returns, VaR(95%), CVaR, skew, kurtosis, percentile bands, and probability of a positive return.
- Risk-adjusted metrics include Sharpe, Sortino, Omega, return-to-VaR, and volatility-adjusted return.
- Pairs are ranked by a composite score combining these metrics.
- CSV outputs such as `pair_rankings_comprehensive.csv` summarise expected returns, volatilities, risk ratios, probability of profitability, regime state, carry, and momentum.

### 1.5 Diagnostics and Tracking

- The engine writes numerous files including volatility forecasts, regime detection, drift components, simulation results by horizon, correlation matrices, PCA loadings, backtest logs, and walk-forward metrics.
- These outputs support reproducibility and help validate whether the models behave as expected over time.

---

## 2. USD Index (DXY) Directional Model

This lightweight model produces a 1-3 day Bullish / Bearish / Neutral call for DXY with an associated confidence and expected range. It merges technical structure, volatility regime, macro events, and text sentiment.

### 2.1 Technical Structure

- Liquidity sweeps identify sessions where only the previous day's high or only the previous day's low is breached, signalling a possible reversal.
- A trend filter evaluates EMA(21/50) spread and slope; alignment with ADR expansion increases conviction.
- Level context measures distance to the prior week's high and low and displacement from key levels.

### 2.2 Volatility Estimation

- Realised measures include TR-ADR(14) and ADR% for intraday range context.
- Parkinson (10-day) and EWMA(20) are used as regime-sensitive realised volatility comparators.
- The model fits a GARCH(1,1) specification to daily percentage returns and forecasts next-day sigma.
- In more advanced variants, GARCH-family models with Student-t innovations and `rescale=False` are selected by AIC, with fallback to local realised sigma.
- A 10/50-day volatility ratio helps detect transitions between compression and expansion regimes.

### 2.3 Event and Sentiment Filters

- High-impact macro events such as NFP, CPI, and FOMC can up-weight volatility and nudge drift.
- Calendars are fetched redundantly via investpy and ForexFactory.
- Interest-rate carry derived from TradingEconomics acts as a proxy drift versus the G5 basket, with a ForexFactory fallback where needed.
- DXY/USD news sentiment is measured using VADER on curated headlines from Google News or FXStreet.
- A Fed-focused subset approximates policy tone and is given greater relevance around policy events.
- A risk-proxy PCA on DXY, Gold, VIX, and SPX provides a risk-on/risk-off vote.

### 2.4 Probability and Scoring

- Each factor contributes a vote in {-1, 0, +1}, including liquidity sweep, trend, PCA vote, sentiment, Monte Carlo probability, ADR expansion, volatility ratio, and level proximity.
- These votes are combined into a final directional score.
- Weights can adapt so that macro-sensitive factors carry more influence when volatility is elevated.
- The final output is mapped into confidence buckets and paired with an expected directional bias.

### 2.5 Outputs

- The script prints the latest ADR values, PCA loadings, liquidity sweep flags, and forecast volatility.
- It ends with a Bullish, Bearish, or Neutral call plus a confidence percentage and a plausible near-term range.

---

## 3. Engineering Notes

- Yahoo Finance is used throughout with `auto_adjust=False` to avoid spurious adjustments in FX series.
- Global currencies are aligned against US trading holidays using forward-fill logic where needed.
- TradingEconomics is scraped with browser-like headers to reduce 403 issues.
- ForexFactory acts as a resilient fallback for both macro calendar data and interest-rate context.
- GARCH-family models use `rescale=False` for stability, and Student-t innovations are used in advanced variants to better handle heavy tails.
- If a model fit fails, realised volatility fallbacks maintain continuity.
- Reproducibility is supported through JSON metadata files, CSV logs, volatility backtests, PCA snapshots, correlation matrices, and ADR exports.
- Safety nets include local realised sigma fallbacks, business-day reindexing, and controlled handling of missing calendar data.

---

## 4. Repository Files

### `pair_selection_forex 6 pairs.py`

Core six-currency pair-selection engine covering USD, EUR, GBP, JPY, CAD, and CHF. Downloads spot FX data, computes realised and forecast volatility, constructs a synthetic USD basket, estimates drift from momentum, sentiment, and carry, and ranks pairs using Monte Carlo simulation outputs and composite risk-adjusted metrics.

### `dxy_directional_model.py`

Baseline DXY directional model. Fetches DXY data, computes ADR and volatility context, performs PCA on correlated macro/risk assets, detects liquidity sweeps, estimates next-day volatility with GARCH, measures news sentiment with VADER, and produces a directional bias with confidence.

### `Macroeconomic 8 pairs 1 month max`

Enhanced eight-currency forecasting notebook adding AUD and NZD. Includes Markov regime switching, broader macro data collection, richer technical indicators, jump-diffusion simulation, pair ranking, forecast tracking, and expanded diagnostics.

### Other notebooks and variants

Other files such as `Macroeconomic Indicators 8 pairs`, `Multi-Asset 2nd Edition`, and similar variants extend the same framework with different asset universes, macro inputs, or forecasting structures.

---

## 5. Limitations

- Public data may lag, break, or change schema, especially on event-heavy days.
- Economic calendar and sentiment features are lightweight heuristics rather than institutional-grade data feeds.
- The models are designed for research and illustrative purposes only.
- This codebase does not constitute investment advice.

---

## 6. Disclaimer

This project is provided strictly for research and educational purposes. It should not be relied upon for live trading or investment decisions.
