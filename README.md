Forex and DXY Modelling

This repository contains a set of Python notebooks and scripts for cross‑sectional currency pair selection and short‑horizon DXY (U.S. Dollar Index) directional forecasting. The codebase aims to be an end‑to‑end research toolkit rather than a ready‑to‑trade system: it fetches public market data, performs regime‑aware volatility and drift estimation, runs Monte‑Carlo simulations, scores currencies and renders diagnostic reports. It is designed for reproducible research – JSON/CSV snapshots are written to disk and important decisions (e.g., choice of volatility model) are logged.

⚠️ Disclaimer: This project is provided for research and educational purposes only. It does not constitute financial advice and should not be used to make live trading decisions.

Repository contents
Script/Notebook	Purpose & key features
pair_selection_forex 6 pairs.py	Implements a currency‑pair selection engine covering six major currencies (USD, EUR, GBP, JPY, CAD, CHF). It downloads spot FX prices from Yahoo Finance, computes realised and forecast volatility (GARCH/EGARCH/GJR‑GARCH with fallback to realised volatility), applies business‑day alignment, constructs a synthetic USD basket, and calculates drift components from momentum, sentiment (VADER on FXStreet RSS), interest‑rate differentials and macro events. An extensive Monte‑Carlo module simulates 1‑day/1‑week/1‑month horizons using regime‑weighted volatility and event boosts. Currencies are ranked on a composite score (Sharpe, Sortino, Omega, return-to‑VaR and volatility‑adjusted return). Numerous CSV outputs – returns_winsorized.csv, volatility_forecasts.csv, sim_results_*.csv, pair_rankings_comprehensive.csv, etc. – enable post‑analysis.
dxy_directional_model.py	Produces a Bullish/Bearish/Neutral call for the U.S. Dollar Index over a 1–3 day horizon. It fetches five years of DXY data, calculates a 14‑day average daily range and a 10‑/50‑day volatility ratio, performs PCA on correlated markets (gold, VIX, S&P 500) to capture a risk‑on/off factor, detects “liquidity sweeps” (single‑leg breaks of the previous day’s high/low), fits a GARCH(1,1) model to forecast next‑day volatility, scrapes Google News headlines for sentiment via VADER, and runs a simple Monte‑Carlo simulation without technical drift. Votes from liquidity sweeps, news sentiment, PCA and Monte‑Carlo tilt the forecast.
Macroeconomic 8 pairs 1 month max	An enhanced FX forecasting notebook covering eight currencies (adds AUD & NZD). It significantly expands the pipeline: it builds a synthetic DXY basket with seven legs, applies Markov‑switching (2‑state) regime detection on log‑returns, performs multi‑model volatility forecasting (GARCH, EGARCH, GJR‑GARCH) and chooses the best model by AIC. Macro data is scraped from TradingEconomics (interest rates, GDP growth, inflation) with 403‑safe fallbacks and from ForexFactory for event calendars. Sentiment is derived from FXStreet RSS, and technical indicators (momentum, RSI, Bollinger position, ADX proxy) are computed. Drifts are assembled from carry, macro fundamentals, sentiment and technical factors with tunable weights. Price paths are simulated via a jump‑diffusion GBM with fat tails and rare jumps. The notebook ranks currency pairs on multiple risk metrics, tracks forecast accuracy, outputs correlation/PCA diagnostics and suggests risk‑management parameters.
Other notebooks (Macroeconomic Indicators 8 pairs, Multi‑Asset 2nd Edition, etc.)	Variants of the enhanced model that experiment with different features (e.g., different macro indicator sets, asset universes or forecast horizons). They follow the same template: download data, perform volatility and regime analysis, compute drift signals, run Monte‑Carlo simulations and produce rankings and reports.
README.md	High‑level overview of the two models, their data sources and methodological components.
Quick start

Because the notebooks/scripts rely on live data and third‑party libraries, running them requires an up‑to‑date Python environment. A typical workflow to reproduce the core models is:

Clone the repository:

git clone https://github.com/LukeFX-445/forex-and-dxy-modelling.git
cd forex-and-dxy-modelling

Install dependencies. The scripts install packages at runtime via pip, but you can pre‑install them for speed:

python -m pip install --upgrade yfinance pandas numpy scikit-learn statsmodels arch investpy feedparser vaderSentiment beautifulsoup4 lxml html5lib fredapi scipy

Run the DXY directional model:

python dxy_directional_model.py

The script prints diagnostic information (latest ADR, PCA loadings, liquidity sweep flags, forecast volatility) and ends with a direction call and confidence percentage.

Run the pair‑selection engine:

python "pair_selection_forex 6 pairs.py"

This script takes several minutes due to data downloads and multiple GARCH fits. Outputs (CSV files) will appear in the working directory. See pair_rankings_comprehensive.csv for a summary table of expected returns, risk metrics and composite scores.

Explore the enhanced macro notebook. For the Macroeconomic 8 pairs 1 month max and related notebooks, we recommend using Jupyter:

jupyter notebook
# open the notebook in your browser

These notebooks contain extensive printouts and generate numerous CSV/JSON files documenting regimes, volatility forecasts, drift components, simulations, rankings and accuracy metrics.

Methodological highlights

Market data: All scripts use Yahoo Finance
 via yfinance (auto_adjust=False) for spot FX pairs, ETFs (UUP), futures (DX=F) and proxies (gold, VIX, S&P 500). Macroeconomic notebooks scrape TradingEconomics and ForexFactory for interest rates, GDP growth, inflation and event calendars. News sentiment is derived from FXStreet or Google News RSS feeds via feedparser and scored with VADER.

Volatility modelling: Realised volatility measures (True Range ADR, Parkinson estimator, EWMA) are combined with multi‑model conditional volatility forecasting using the arch package. The enhanced model tests GARCH, EGARCH and GJR‑GARCH with Student‑t innovations and selects the AIC‑optimal model. Regime detection uses a two‑state Markov switching model from statsmodels.

Drift estimation: Drift signals come from momentum (recent log‑returns), carry trades (interest rate differentials), macro fundamentals (GDP/inflation differentials), sentiment and technical indicators (RSI, Bollinger position, ADX proxy). Weights are tuned to ensure no single component overwhelms the forecast (see inline comments in the code).

Monte‑Carlo simulation: Directional forecasts and pair rankings are based on forward price simulations. The simple DXY script uses a Gaussian GBM, while the macro notebooks implement a jump‑diffusion process with fat‑tailed returns and rare jumps. Event calendars increase volatility multiplicatively and adjust drift additively around high‑impact events.

Diagnostics and reproducibility: Each run writes JSON/CSV logs (e.g., run_meta.json, volatility_forecasts.csv, pair_rankings_comprehensive.csv, walkforward_metrics.csv) so results can be audited. Seeds are set for reproducibility, and warnings/errors are handled gracefully (with fallback calculations) to prevent crashes.

Extending the models

Additional assets: To add more currency pairs or other asset classes (commodities, equities, crypto), modify the tickers dictionary and extend the drift and volatility logic accordingly. Ensure that you adjust the synthetic USD basket weights when adding new legs.

Alternative data: The code is structured so that macro and sentiment inputs come from functions (fetch_interest_rates_enhanced, fetch_economic_indicators, fetch_news_sentiments). You can plug in premium data feeds or more sophisticated NLP models by replacing these functions.

Parameter tuning: Many weights (e.g., sentiment scaling, regime transition thresholds, Monte‑Carlo jump probabilities) are exposed as variables in the notebooks. Experiment with different horizons, trial counts or weighting schemes to stress‑test the robustness of the signals.
