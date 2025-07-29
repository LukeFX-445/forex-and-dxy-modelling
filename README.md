# forex-and-dxy-modelling

This repository contains models for analysing foreign exchange pair selection and for forecasting the short-term direction of the U.S. Dollar Index (DXY). The framework combines volatility modelling, dimensionality reduction, Monte Carlo simulations and sentiment analysis.

## 1. Forex Pair Selection Model

This model ranks currencies based on structure, volatility and divergence to identify asymmetric setups. Its methodology draws on both market microstructure concepts and quantitative statistics.

### 1.1 Data Sources

The model uses price series for the G10 currencies, represented by futures contracts (e.g., 6E, 6B, 6J) or exchange‑traded fund proxies (e.g., UUP).

### 1.2 Volatility Filtering

Volatility regimes are estimated using the average daily range (ADR) and Generalised Autoregressive Conditional Heteroskedasticity (GARCH) models. The ADR measures the average price range of a currency pair over a day. GARCH models estimate the conditional variance as a weighted combination of past variances and squared returns.

### 1.3 Directional Filtering

Price action is evaluated around liquidity zones by identifying sweeps of previous highs or lows, fair value gaps (FVGs) and order blocks (OBs). These filters select currencies showing potential displacement moves relative to recent liquidity.

### 1.4 Dimensional Reduction

Principal Component Analysis (PCA) is applied to daily returns and ADR‑normalised movements. PCA reduces the dimensionality of correlated price series, capturing most of the information in a few orthogonal components. Outlier detection on the component loadings highlights currencies that are misaligned or lagging relative to the cross‑section.

### 1.5 Monte Carlo Simulations

Forward return paths are simulated under a geometric Brownian motion (GBM) assumption. In a GBM framework the change in price is modelled as


ΔS = S × ( μ Δt + σ ε Δt )

where S is the price, μ is the drift, σ is the volatility and ε is a random normal shock. Scenarios are generated under varying volatility shocks and macro 


### 1.6 Sentiment Analysis

The model scrapes news headlines, tags them with tones such as “hawkish” or “dovish” and converts qualitative text into sentiment scores. These scores are cross‑validated against recent price action to confirm or contradict the directional bias.

## 2. USD Index (DXY) Directional Model

This model forecasts the short‑term direction of the U.S. Dollar Index using a combination of technical structure, volatility measures, event filters and sentiment analysis.

### 2.1 Technical Structure

The model monitors structural breaks, fair value gaps, order blocks, liquidity sweeps and displacement moves in the DXY price series to infer shifts in market structure.

### 2.2 Volatility Estimation

ADR bands, standard deviation channels and outputs from GARCH or heteroskedastic models provide a view of volatility regimes. High or low volatility phases influence the weighting of technical signals.

### 2.3 Event Filters

Macroeconomic triggers—including non‑farm payrolls (NFP), consumer price index (CPI) releases, Federal Open Market Committee (FOMC) meetings, Treasury Inflation‑Protected Securities (TIPS) auctions and real yields—are incorporated as exogenous shocks. High‑impact events override technical signals within a dynamic weighting scheme.

### 2.4 Sentiment Filters

News feeds are parsed to detect hawkish or dovish tones and risk‑on or risk‑off shifts. Event‑driven tone shifts, such as surprise central‑bank decisions, receive higher weights.

### 2.5 Probability and Scoring

Each factor is converted into a confidence score on a 0 – 1 scale. The final directional bias is computed as the difference between the probability of a bullish outcome and the probability of a bearish outcome:
Δ_score = P(bullish) - P(bearish)


Scores are adjusted dynamically so that macro events carry more weight when volatility is elevated. A summary output reports the directional bias (bullish, bearish or neutral), the confidence level (low, medium or high) and an expected price range based on ADR‑adjusted bands.

### 2.6 Probability Density Functions

Optional probability density functions over a 1–3 day horizon illustrate the distribution of expected returns and allow the assessment of skewness, fat tails and confidence intervals before trade execution.

---

This repository thus implements two complementary models: one to rank forex pairs and another to forecast DXY direction. Both integrate volatility estimation, dimensional reduction and sentiment analysis to produce probabilistic trade signals.
