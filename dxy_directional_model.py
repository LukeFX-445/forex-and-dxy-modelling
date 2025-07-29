"""
Directional model for the US Dollar Index (DXY).

This script retrieves historical DXY data, calculates average daily range, performs principal component analysis on correlated markets (gold, volatility index, stock indices), checks for liquidity sweeps, forecasts volatility with GARCH, scrapes news sentiment and simulates price paths to determine directional bias (bullish, bearish or neutral).
"""
!pip install yfinance pandas numpy  # install necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np

# Fetch 5 years of DXY (US Dollar Index) data
ticker = "DX-Y.NYB"
dxy_data = yf.download(ticker, period="5y", interval="1d")
# Ensure proper column naming for later use
dxy_data = dxy_data.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume"})
print(f"Fetched {len(dxy_data)} daily data points from {dxy_data.index[0].date()} to {dxy_data.index[-1].date()}.")
display(dxy_data.tail(3))  # display last 3 days for verification

# Compute Average Daily Range (ADR) over a 14-day window
dxy_data['daily_range'] = dxy_data['high'] - dxy_data['low']
dxy_data['ADR_14'] = dxy_data['daily_range'].rolling(14).mean()

# Determine if the latest daily range is in expansion relative to ADR
latest_range = dxy_data['daily_range'].iloc[-1]
adr_value = dxy_data['ADR_14'].iloc[-2]  # use penultimate ADR (last day might not have full window)
expansion = latest_range > adr_value  # True if today's range > recent ADR (volatility expansion)
print(f"Latest daily range = {latest_range:.4f}, 14-day ADR = {adr_value:.4f}, Expansion = {expansion}")

!pip install scikit-learn
from sklearn.decomposition import PCA

# Fetch correlated markets: Gold, VIX, S&P500 for the same period
other_tickers = ["GC=F", "^VIX", "^GSPC"]  # Gold futures, CBOE VIX, S&P 500 index
other_data = yf.download(other_tickers, start=dxy_data.index[0].date(), end=dxy_data.index[-1].date(), interval="1d")['Close']
# Align and compute daily returns for PCA
all_close = pd.concat([dxy_data['close'], other_data], axis=1).dropna()
all_close.columns = ['DXY', 'Gold', 'VIX', 'SPX']
returns = all_close.pct_change().dropna()

# Perform PCA on returns of DXY, Gold, VIX, SPX
pca = PCA(n_components=1)
pca.fit(returns)
pc1_values = pca.transform(returns)[:,0]
latest_pc1 = pc1_values[-1]
print(f"Latest PCA1 value: {latest_pc1:.4f}, PCA1 component weights: {pca.components_[0]}")


# Check if yesterday's high or low was breached (liquidity sweep)
prev_high = dxy_data['high'].iloc[-2]
prev_low = dxy_data['low'].iloc[-2]
today_high = dxy_data['high'].iloc[-1]
today_low = dxy_data['low'].iloc[-1]

swept_high = today_high > prev_high  # took out previous day's high
swept_low = today_low < prev_low    # took out previous day's low

liquidity_bias = 0
if swept_high.any() and not swept_low.any():
    liquidity_bias = -1  # only high swept: took liquidity above (potential bearish reversal)
elif swept_low.any() and not swept_high.any():
    liquidity_bias = 1   # only low swept: took liquidity below (potential bullish reversal)
print(f"Swept High: {str(swept_high)}, Swept Low: {str(swept_low)}, Liquidity Bias = {liquidity_bias}")

!pip install arch
from arch import arch_model

# Prepare returns for volatility modeling
dxy_returns = dxy_data['close'].pct_change().dropna() * 100  # percentage returns
# Fit GARCH(1,1) model (assuming zero mean return for simplicity)
am = arch_model(dxy_returns, vol='GARCH', p=1, q=1, mean='Zero')
res = am.fit(disp='off')
# Forecast next day's volatility
forecast = res.forecast(horizon=1)
next_var = forecast.variance.iloc[-1, 0]
next_vol = float(np.sqrt(next_var))
print(f"Forecasted next-day volatility (stdev of returns) = {next_vol:.3f}%")

# Compute last week's high and low
weekly_data = dxy_data['close'].resample('W').last()  # weekly last price
prev_week_high = dxy_data['high'].iloc[-5:].max()   # high of last 5 trading days (~1 week)
prev_week_low = dxy_data['low'].iloc[-5:].min()     # low of last 5 trading days
close_price = dxy_data['close'].iloc[-1]

# Proximity of current price to last week's extremes (in %)
dist_to_high = (prev_week_high - close_price) / close_price * 100
dist_to_low = (close_price - prev_week_low) / close_price * 100

print(f"Distance to last week's high: {float(dist_to_high.iloc[0]):.2f}%, to low: {float(dist_to_low.iloc[0]):.2f}%")

# Calculate 10-day vs 50-day volatility ratio
short_vol = dxy_returns.iloc[-10:].std()
long_vol = dxy_returns.iloc[-50:].std()
vol_ratio = short_vol / long_vol
print(f"10-day vs 50-day volatility ratio: {vol_ratio.iloc[0]:.2f}")

!pip install feedparser vaderSentiment
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Parse latest news headlines related to DXY or USD
news_feed = feedparser.parse("https://news.google.com/rss/search?q=US+Dollar+Index+DXY")
analyzer = SentimentIntensityAnalyzer()
scores = []
for entry in news_feed.entries[:10]:  # consider up to 10 recent news items
    title = entry.title
    vs = analyzer.polarity_scores(title)
    scores.append(vs['compound'])
avg_sentiment = np.mean(scores) if scores else 0
print(f"Avg news sentiment (compound VADER score) = {avg_sentiment:.2f}")

import math
# Use forecast volatility (next_vol) from GARCH and historical mean return
mean_ret = dxy_returns.mean().iloc[0] # Extract scalar from Series
vol = next_vol  # forecasted vol in %
# Introduce a directional drift based on technical bias (e.g., trend)
# Removed problematic technical trend calculation and drift based on it

import numpy as np

# Assuming mean_ret and vol are already defined in % (e.g., 0.1 for 0.1%)
sim_count = 1000
last_price = dxy_data['close'].iloc[-1].iloc[0] # Extract scalar from Series

ups = 0
for _ in range(sim_count):
    # simulate return (normal dist) without technical trend drift
    simulated_ret = np.random.normal(loc=mean_ret, scale=vol)
    sim_price = last_price * (1 + simulated_ret/100)
    if sim_price > last_price:
        ups += 1

prob_up = ups / sim_count

# Aggregate directional signals
votes = []
# Liquidity sweep vote
votes.append(liquidity_bias)
# News sentiment vote
votes.append(1 if avg_sentiment > 0 else (-1 if avg_sentiment < 0 else 0))
# Policy drift vote
# votes.append(policy_bias) # Removed as policy_bias is not defined
# PCA safe-haven sentiment vote
# Assuming PCA first component positive => risk-off => bullish USD if DXY loading is positive
# Determine sign of DXY's loading in PCA component
dxy_loading = pca.components_[0][0]
pca_vote = 1 if latest_pc1 * dxy_loading > 0 else -1
votes.append(pca_vote)
# Monte Carlo vote
votes.append(1 if prob_up > 0.5 else -1)

# Sum votes and determine majority
score = sum(v for v in votes if v != 0)
direction = "Bullish" if score > 0 else ("Bearish" if score < 0 else "Neutral")

# Confidence as percentage of votes aligned with direction
valid_votes = [v for v in votes if v != 0]
if len(valid_votes) == 0:
    confidence = 50.0
else:
    align = sum(1 for v in valid_votes if (v>0 and score>0) or (v<0 and score<0))
    confidence = align / len(valid_votes) * 100

latest_date = dxy_data.index[-1].date()
print(f"Forecast for {latest_date}: {direction} with {confidence:.0f}% confidence.")
