# forex-and-dxy-modelling

This repository contains two complementary models: (i) an FX cross-section engine for **currency-pair selection**, and (ii) a **short-horizon directional model** for the U.S. Dollar Index (**DXY**). The framework integrates regime-aware volatility estimation, dimensionality reduction, Monte Carlo simulation, event filters, and sentiment analysis. It is designed for transparent diagnostics and reproducible research.

---

## 1. Forex Pair Selection Model

This model ranks currencies by structural context, volatility regime, and cross-sectional divergence to identify asymmetric opportunities. The methodology combines market-microstructure heuristics with robust statistical filtering.

### 1.1 Data Sources
- Spot FX and proxies via Yahoo Finance (`auto_adjust=False`) to preserve raw price dynamics (e.g., `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`; `DX-Y.NYB` / `UUP` for USD).
- Optional currency futures (e.g., 6E, 6B, 6J) where available.
- A synthetic DXY-like USD basket constructed from major legs for diagnostic comparison.

### 1.2 Volatility Filtering
- **Average Daily Range (ADR)** using **True Range ADR(14)** and **ADR%** (ADR divided by level) for scale-free comparisons.
- **Realised volatility comparators**: **Parkinson** estimator (10-day) and **EWMA** (20-day).
- **Conditional volatility models**: **GARCH(1,1)**, **EGARCH**, and **GJR-GARCH** with **Student-t** innovations (`rescale=False`). Model selection uses **AIC**; if fitting fails, a **local realised σ** fallback maintains continuity.

### 1.3 Directional Filtering
- Microstructure cues: single-sided **liquidity sweeps** (prior high/low taken), displacement moves, and simple **FVG/OB** heuristics.
- Regime awareness: signal weights adapt to volatility state (e.g., ADR expansion and an elevated 10/50-day vol-ratio increase emphasis on trend-following evidence).

### 1.4 Dimensionality Reduction
- **PCA** on standardised daily returns (and ADR-normalised moves) to extract common risk factors.
- Outlier screening on **loadings** flags currencies that are misaligned or lagging relative to the cross-section.
- Sanity checks compare **PC1** to USD proxies (DXY/UUP and the synthetic basket).

### 1.5 Monte Carlo Simulation
Forward paths are simulated under a geometric Brownian motion (GBM) with model-implied drift and volatility. Scenarios incorporate volatility shocks, carry differentials, and event-week boosts to σ. Outputs include expected-move bands, percentiles, VaR(95%), and ES(95%).

### 1.6 Sentiment Analysis
- RSS headlines are parsed with **VADER**; a light central-bank lexicon (Fed, ECB, BoE, BoJ, BoC, SNB) aids context.
- Sentiment scores are cross-validated against recent price action and PCA-derived risk to confirm or challenge directional hypotheses.

---

## 2. USD Index (DXY) Directional Model

This model produces a 1–3 day **Bullish / Bearish / Neutral** call for DXY with an associated **confidence** and **expected range**. It merges technical structure, volatility regime, macro events, and text sentiment.

### 2.1 Technical Structure
- **Liquidity sweeps**: sessions that take out only the previous day’s high (bearish risk) or only the previous day’s low (bullish risk).
- **Trend filter**: EMA(21/50) spread and slope; alignment with ADR expansion increases conviction.
- **Level context**: proximity to the prior week’s extremal range and evidence of displacement from key levels.

### 2.2 Volatility Estimation
- **TR-ADR(14)** and **ADR%** for intraday range context.
- **Parkinson (10-day)** and **EWMA(20)** realised volatility for regime classification.
- **GARCH family** (GARCH/EGARCH/GJR-GARCH, Student-t, `rescale=False`) to forecast **next-day σ**; the **AIC-optimal** model is selected with a robust fallback to local realised σ.
- **10/50-day vol-ratio** to detect transitions between compression and expansion regimes.

### 2.3 Event Filters
- High-impact macro events (e.g., **NFP**, **CPI**, **FOMC**, real yields) **upweight volatility** and may **nudge drift**.
- Calendars fetched with redundancy (investpy plus **ForexFactory JSON/HTML** fallback).
- **Interest-rate carry** derived from TradingEconomics with 403-safe headers; a ForexFactory fallback is provided. Carry acts as a proxy drift versus the G5 basket.

### 2.4 Sentiment Filters
- **DXY/USD news sentiment** via VADER on curated headlines.
- A **Fed-focused** slice approximates policy tone (hawkish/dovish), which receives greater weight around policy events.
- A **risk-proxy PCA** using **[DXY, Gold, VIX, SPX]** (standardised). The sign of PC1 relative to DXY provides a risk-on/risk-off vote.

### 2.5 Probability and Scoring
- Each factor contributes a vote in `{−1, 0, +1}` with tunable weights (e.g., trend, PCA vote, liquidity sweep, sentiment—broad and Fed-specific—Monte-Carlo up-probability, ADR expansion, 10/50 vol-ratio, proximity/level tests).
- The final **directional score** is the weighted sum, mapped to **confidence buckets**. Weights adapt so that macro factors carry greater influence when **volatility is elevated**.

### 2.6 Probability Density Functions
Optional PDFs for 1–3 day horizons describe return distributions (skewness, kurtosis) and provide **p5/p50/p95** intervals alongside **VaR(95%)** and **ES(95%)**.

---

## 3. Engineering Notes
- **Yahoo Finance**: `auto_adjust=False` throughout to avoid spurious adjustments in FX series.
- **403-safe interest-rate retrieval**: TradingEconomics scraped with browser-like headers; **ForexFactory** as resilient fallback.
- **Model stability**: `rescale=False` in the GARCH family; Student-t innovations for heavy tails; an **outlier-winsorised** return snapshot logged for diagnostics.
- **Reproducibility**: JSON run metadata; daily volatility backtest log; PCA/correlation/ADR snapshots exported as CSV.
- **Safety nets**: local realised σ for failed fits; `ffill` on minor calendar gaps; **business-day reindexing**.

---

## 4. Limitations
- Public data may lag, and schema can change; verify key diagnostics on event days.
- Calendar and sentiment features are **lightweight heuristics**; consider extending with richer NLP or professional data feeds.
- This codebase is provided strictly for **research**; it does **not** constitute investment advice.
