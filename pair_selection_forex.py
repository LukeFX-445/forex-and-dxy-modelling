"""
Pair Selection Model
---------------------

This script analyses the US Dollar Index (DXY) and major foreign currencies using a combination of market data, principal component analysis, volatility modelling, event‑driven signals and sentiment analysis to form a directional bias.

Functions:
- fetch_dxy_data: downloads DXY OHLCV data and calculates ADR.
- perform_pca: runs PCA on DXY and related markets (Gold, VIX, S&P 500) to infer risk‑on/risk‑off sentiment.
- check_liquidity_sweep: checks whether the prior day's high or low was taken out.
- fit_garch: fits a GARCH(1,1) model to DXY returns and forecasts next‑day volatility.
- compute_volatility_ratio: calculates 10‑day vs 50‑day volatility ratio.
- fetch_news_sentiment: scrapes recent news and scores sentiment using VADER.

At the end of the script the various signals are combined into a simple voting system which outputs a bullish, bearish or neutral bias along with a confidence score.
"""
!pip install yfinance --quiet
import yfinance as yf
import pandas as pd

# Define currency futures tickers (as per Yahoo Finance)
tickers = {
    'EUR': '6E=F',
    'GBP': '6B=F',
    'JPY': '6J=F',
    'CAD': '6C=F',
    'CHF': '6S=F',
    'USD': 'DX-Y.NYB'  # US Dollar Index
}

# Fetch daily OHLCV for the last 2 years
data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, period='2y', interval='1d')
    df['Currency'] = name
    data[name] = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Concatenate all into one DataFrame
combined_df = pd.concat(data.values(), keys=data.keys(), names=['Currency', 'Date'])
combined_df.reset_index(inplace=True)
combined_df.to_csv('currency_futures_data.csv', index=False)


# 1. Install libraries (if not already installed in Colab)
!pip install yfinance arch investpy pandas_datareader -q

# 2. Imports
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from arch import arch_model

# 3. Define the currencies and fetch historical data
# We'll use Yahoo Finance tickers:
# EURUSD=X, GBPUSD=X are quoted as base currency per USD (EUR/USD, GBP/USD).
# USDJPY=X, USDCHF=X, USDCAD=X are quoted as USD per currency (inverted from base perspective).
# We'll also use UUP (USD Index ETF) as a proxy for USD's own strength.
tickers = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "USDCAD=X", "UUP"]

# Define date range for historical data (e.g., last 2 years for volatility calc)
end_date = datetime.utcnow()
start_date = end_date - timedelta(days= 3*365)  # 3 years of data
data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
# 'data' will be a multi-index DataFrame: columns like ('Adj Close','EURUSD=X'), etc.
# We'll use the 'Close' prices for analysis (for currencies, Adj Close == Close as there's no corporate action).
closes = data['Close'].copy()

# Drop any rows with missing values (e.g., holidays)
closes.dropna(inplace=True)

# Invert the USD/X pairs so that all series represent the value of the named currency in USD.
closes['JPYUSD'] = 1 / closes['USDJPY=X']  # JPY value in USD
closes['CHFUSD'] = 1 / closes['USDCHF=X']  # CHF value in USD
closes['CADUSD'] = 1 / closes['USDCAD=X']  # CAD value in USD
# For consistency, rename columns for clarity
closes.rename(columns={
    'EURUSD=X': 'EURUSD',
    'GBPUSD=X': 'GBPUSD',
    'USDJPY=X': 'USDJPY',  # We'll not use USDJPY directly now that we have JPYUSD
    'USDCHF=X': 'USDCHF',
    'USDCAD=X': 'USDCAD',
    'UUP': 'USD_Index'
}, inplace=True)
# Remove original USD/JPY, USD/CHF, USD/CAD columns (to avoid confusion)
closes.drop(columns=['USDJPY','USDCHF','USDCAD'], inplace=True, errors='ignore')

# Now 'closes' has columns: EURUSD, GBPUSD, USD_Index, plus JPYUSD, CHFUSD, CADUSD we added.
currencies = ['EURUSD','GBPUSD','JPYUSD','CADUSD','CHFUSD','USD_Index']

# 4. Compute daily returns for each series (log returns for stability in GARCH)
returns = np.log(closes[currencies] / closes[currencies].shift(1)).dropna()

# 5. Compute Average Daily Range (ADR) for recent period (e.g., 10 days)
highs = data['High'].copy()
lows = data['Low'].copy()
# Compute ADR for each currency: mean of (High-Low) over last N days
N = 10  # period for ADR
adr = {}
for col in currencies:
    if col in ['JPYUSD','CADUSD','CHFUSD']:
        # For inverted pairs, ADR in price terms: use inverted high/low
        # high of base = 1/low of quote pair, low of base = 1/high of quote pair
        base = col[:3]  # 'JPY', 'CAD', 'CHF'
        quote_pair = 'USD'+base+'=X'  # e.g., 'USDJPY=X'
        inv_high = 1 / lows[quote_pair]   # when quote pair is at low, base currency is at high
        inv_low = 1 / highs[quote_pair]   # when quote pair is at high, base is at low
        adr[col] = float((inv_high - inv_low).tail(N).mean())
    elif col == 'USD_Index':
        # For USD_Index (UUP ETF), use its own high-low
        adr[col] = float((highs['UUP'] - lows['UUP']).tail(N).mean())
    else:
        # Direct pair like EURUSD or GBPUSD
        adr[col] = float((highs[col+'=X'] - lows[col+'=X']).tail(N).mean())

# 6. Fit GARCH(1,1) on recent returns to forecast volatility
vol_forecast = {}
for cur in currencies:
    # We annualize returns to percentage for stability in arch (optional)
    series = returns[cur] * 100
    # Fit GARCH(1,1)
    am = arch_model(series, vol='Garch', p=1, q=1, rescale=False)
    res = am.fit(disp='off')
    # One-step ahead forecast (variance)
    fcast = res.forecast(horizon=1)
    sigma_next = np.sqrt(fcast.variance.values[-1, 0])  # predicted vol (in percentage)
    # Convert back to daily std dev in fraction
    vol_forecast[cur] = float(sigma_next / 100)  # as fraction (e.g. 0.005 means 0.5% daily vol)

# 7. Economic calendar integration: check for any high-impact events for each currency in the next horizon.
import datetime as dt
from investpy.news import economic_calendar

today = dt.date.today()
# Set up date windows
tomorrow = today + dt.timedelta(days=1)
week_ahead = today + dt.timedelta(days=7)

# Fetch high-impact events from today through the next 7 days.
try:
    cal = economic_calendar(countries=['united states','euro zone','united kingdom','japan','canada','switzerland'],
                             importances=['high'], from_date=today.strftime('%d/%m/%Y'), to_date=week_ahead.strftime('%d/%m/%Y'))
except Exception as e:
    cal = pd.DataFrame()  # if fetch fails, use empty dataframe

# Determine event-driven volatility or bias adjustments
event_vol_boost = {cur: 1.0 for cur in currencies}  # multiplier for volatility
event_drift_boost = {cur: 0.0 for cur in currencies}  # additive drift (in fraction per day)
if not cal.empty:
    for _, event in cal.iterrows():
        ccy = str(event['currency']).upper() if 'currency' in event else ''
        # Map currency code to our keys
        if ccy == 'USD':
            affected = 'USD_Index'
        elif ccy in ['EUR','EURZONE','EUR ZONE']:
            affected = 'EURUSD'
        elif ccy == 'GBP':
            affected = 'GBPUSD'
        elif ccy == 'JPY':
            affected = 'JPYUSD'
        elif ccy == 'CAD':
            affected = 'CADUSD'
        elif ccy == 'CHF':
            affected = 'CHFUSD'
        else:
            affected = None
        if affected and affected in event_vol_boost:
            # Increase volatility for this currency due to upcoming high-impact event
            event_vol_boost[affected] = 1.3  # e.g., 30% vol increase
            # If the event is a rate decision or Fed meeting, we might also adjust drift:
            title = str(event.get('event', '')).lower()
            if 'interest rate' in title or 'central bank' in title or 'fed' in title:
                # If a hike expected, assume bullish drift for that currency, if cut expected, bearish
                # Here we simplisticly check if "decision" is in title as a sign of a policy meeting.
                # Without actual forecast data, we'll assume a potential hawkish surprise bias.
                event_drift_boost[affected] += 0.001  # small upward drift (0.1%) for potential hike

!pip install feedparser vaderSentiment yfinance arch pandas numpy

import pandas as pd
import numpy as np
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from arch import arch_model

def fetch_news_sentiments():
    # FXStreet RSS feed for latest Forex news
    feed_url = "https://www.fxstreet.com/rss/news"
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:50]  # consider latest 50 news items for broad coverage

    # Define keywords to identify central bank news for each currency
    cb_keywords = {
        'USD': ['Fed', 'Federal Reserve'],
        'EUR': ['ECB', 'European Central Bank'],
        'GBP': ['BoE', 'Bank of England'],
        'JPY': ['BoJ', 'Bank of Japan'],
        'CAD': ['BoC', 'Bank of Canada'],
        'CHF': ['SNB', 'Swiss National Bank']
    }

    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = {curr: 0.0 for curr in cb_keywords}  # default 0 if no news

    # Find the latest news for each central bank and analyze sentiment
    for entry in entries:
        title = entry.title
        summary = entry.summary if 'summary' in entry else ''
        text = title + ". " + summary
        for curr, keywords in cb_keywords.items():
            if any(key in title or key in summary for key in keywords):
                # Compute sentiment score (compound in [-1,1])
                score = analyzer.polarity_scores(text)['compound']
                sentiment_scores[curr] = score
                # Once found the latest relevant news, we can break for that currency
                # (We assume the RSS is sorted newest first)

    return sentiment_scores

def fetch_interest_rates():
    # Scrape the TradingEconomics interest rate table
    url = "https://tradingeconomics.com/country-list/interest-rate"
    try:
        tables = pd.read_html(url)
    except Exception as e:
        print("Error fetching interest rates:", e)
        return {}
    if len(tables) == 0:
        return {}
    df_rates = tables[0]
    # Ensure column names are standard
    df_rates.columns = [col.strip() for col in df_rates.columns]

    # Map currencies to country/region names as listed on TradingEconomics
    country_map = {
        'USD': 'United States',    # Fed Funds Rate
        'EUR': 'Euro Area',        # ECB main rate
        'GBP': 'United Kingdom',   # BoE rate
        'JPY': 'Japan',            # BoJ rate
        'CAD': 'Canada',           # BoC rate
        'CHF': 'Switzerland'       # SNB rate
    }
    rates = {}
    for curr, country in country_map.items():
        # Find row where Country contains the country name (to handle variations)
        row = df_rates[df_rates['Country'].str.contains(country, case=False)]
        if not row.empty:
            # Assume "Last" column has the current interest rate (as float or string)
            try:
                rate_value = float(str(row.iloc[0]['Last']).strip().replace('%',''))
            except:
                rate_value = None
            if rate_value is not None:
                rates[curr] = rate_value
    return rates

def fetch_price_data():
    # Define Yahoo Finance tickers for each currency pair
    # Using USD as base or quote as appropriate
    tickers = {
        'USD': "DX-Y.NYB",       # US Dollar Index (NYBOT) as a proxy for USD strength
        'EUR': "EURUSD=X",       # EUR/USD
        'GBP': "GBPUSD=X",       # GBP/USD
        'JPY': "USDJPY=X",       # USD/JPY
        'CAD': "USDCAD=X",       # USD/CAD
        'CHF': "USDCHF=X"        # USD/CHF
    }
    price_data = {}
    momentum = {}
    for curr, ticker in tickers.items():
        try:
            # Fetch last 10 days of data to compute 5-day momentum
            hist = yf.download(ticker, period="10d", interval="1d", progress=False)
        except Exception as e:
            print(f"Error downloading data for {curr}: {e}")
            continue
        if hist.empty:
            continue
        # Use the last available close price
        last_price = hist['Close'].iloc[-1]
        price_data[curr] = last_price

        # Compute 5-day average return (momentum)
        # Calculate daily returns (percentage change)
        returns = hist['Close'].pct_change().dropna()
        if len(returns) >= 5:
            last5 = returns.iloc[-5:]
            avg_return = last5.mean()
        else:
            avg_return = returns.mean()
        # Adjust sign for currencies where USD is base (reverse relationship)
        if curr in ['JPY', 'CAD', 'CHF']:
            # For USD/JPY, USD/CAD, USD/CHF: if pair rose, USD strengthened, so opposite for JPY,CAD,CHF
            avg_return = -avg_return
        momentum[curr] = avg_return
    return price_data, momentum

# Fetch all required data
sentiments = fetch_news_sentiments()
rates = fetch_interest_rates()
price_data, momentum = fetch_price_data()

# Compute average interest rate for macro bias calculation
avg_rate = np.mean([r for r in rates.values() if r is not None]) if rates else 0

drift_components = {}
volatility = {}

for curr in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'CHF']:
    # Momentum component (as daily fraction)
    mom = momentum.get(curr, 0.0)

    # Sentiment component (scale the VADER compound score)
    sent = sentiments.get(curr, 0.0)
    sentiment_drift = sent * 0.002  # scaling factor: e.g. 0.002 -> up to ±0.2% per day

    # Macro/interest rate component (daily fraction of interest rate difference)
    if curr in rates and rates[curr] is not None:
        # interest rate in percent
        rate = rates[curr]
        # relative to average rate
        rate_diff = (rate - avg_rate)
        macro_drift = (rate_diff / 100.0) / 252.0  # convert percent to fraction and spread over trading days
    else:
        macro_drift = 0.0

    # Event drift boost (e.g., if an event keyword found in news, add a small bias)
    event_drift = 0.0
    # Example: boost if news title contains "hike" or "cut"
    # (Here we illustrate a simple rule: +0.001 for hike, -0.001 for cut if sentiment already captured it)
    # In practice, you might parse the content for explicit event signals.
    if curr in sentiments:
        title_text = ""  # (We did not store titles; this would require modifying fetch_news_sentiments to return titles)
        # if 'hike' in title_text.lower():
        #     event_drift += 0.001
        # elif 'cut' in title_text.lower():
        #     event_drift -= 0.001

    # Total drift for the currency
    total_drift = mom + sentiment_drift + macro_drift + event_drift

    drift_components[curr] = {
        'momentum': mom,
        'sentiment': sentiment_drift,
        'macro': macro_drift,
        'event': event_drift,
        'total': total_drift
    }

    # Fit GARCH(1,1) to recent returns for volatility forecast
    # We use the last 60 days of data for a decent volatility estimate (if available)
    hist_days = 60
    vol = None
    try:
        # For GARCH, obtain a historical series of daily returns
        ticker = None
        if curr == 'USD':
            ticker = "DX-Y.NYB"
        elif curr == 'EUR':
            ticker = "EURUSD=X"
        elif curr == 'GBP':
            ticker = "GBPUSD=X"
        elif curr == 'JPY':
            ticker = "USDJPY=X"
        elif curr == 'CAD':
            ticker = "USDCAD=X"
        elif curr == 'CHF':
            ticker = "USDCHF=X"
        if ticker:
            hist = yf.download(ticker, period=f"{hist_days}d", interval="1d", progress=False)
            returns = hist['Close'].pct_change().dropna()
            # Adjust sign for reverse pairs as before
            if curr in ['JPY', 'CAD', 'CHF']:
                returns = -returns
            if not returns.empty:
                # Fit GARCH(1,1)
                am = arch_model(returns * 100, p=1, q=1)  # multiply by 100 to express in percentage (optional for convergence)
                res = am.fit(disp='off')
                # Forecast volatility for the next day
                forecast = res.forecast(horizon=1)
                # forecast.variance is a DataFrame, get the last forecast var
                var_forecast = forecast.variance.iloc[-1, 0]
                vol = np.sqrt(var_forecast) / 100.0  # convert back to fraction (since we scaled returns by 100)
    except Exception as e:
        print(f"GARCH model failed for {curr}: {e}")
    if vol is None:
        # Fallback: use sample std dev of returns as volatility
        vol = returns.std() if 'returns' in locals() else 0.0
    volatility[curr] = vol

    import math

def simulate_prices(initial_price, drift, vol, days=1, trials=1000):
    """Simulate price paths for given days and trials using GBM with constant drift and vol."""
    results = []
    for _ in range(trials):
        price = initial_price
        for t in range(days):
            # Draw a random shock
            eps = np.random.normal()
            # GBM price update
            price *= math.exp((drift - 0.5 * vol**2) + vol * eps)
        results.append(price)
    return np.array(results)

# Set simulation parameters
trials = 1000
horizons = {'1d': 1, '1w': 5, '1m': 22}  # 1 day, 5 days, 22 days

# Store simulation outcomes
sim_results = {h: {} for h in horizons}
for h, days in horizons.items():
    for curr in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'CHF']:
        if curr not in price_data or price_data[curr] is None:
            continue

        P0 = price_data[curr]
        if isinstance(P0, pd.Series):
            P0 = P0.item()

        mu = drift_components[curr]['total']
        if isinstance(mu, pd.Series):
            mu = mu.item()

        sigma = volatility.get(curr, 0.0)
        # Simulate
        final_prices = simulate_prices(P0, mu, sigma, days=days, trials=trials)
        # Calculate return relative to initial
        returns = final_prices / P0 - 1.0
        sim_results[h][curr] = {
            'final_prices': final_prices,
            'mean_return': returns.mean(),
            'median_return': np.median(returns),
            'std_return': returns.std()
        }

# Determine most bullish/bearish/expansive
if '1d' in sim_results and sim_results['1d']:
    # For 1-day horizon
    day_returns = {curr: sim_results['1d'][curr]['mean_return'] for curr in sim_results['1d']}
    most_bullish_day = max(day_returns, key=day_returns.get)
    most_bearish_day = min(day_returns, key=day_returns.get)
    print("Daily (1-day) horizon: Most Bullish = {} , Most Bearish = {}".format(most_bullish_day, most_bearish_day))

if '1w' in sim_results and sim_results['1w']:
    week_returns = {curr: sim_results['1w'][curr]['mean_return'] for curr in sim_results['1w']}
    most_bullish_week = max(week_returns, key=week_returns.get)
    most_bearish_week = min(week_returns, key=week_returns.get)
    print("Weekly (5-day) horizon: Most Bullish = {} , Most Bearish = {}".format(most_bullish_week, most_bearish_week))

if '1m' in sim_results and sim_results['1m']:
    month_vols = {curr: sim_results['1m'][curr]['std_return'] for curr in sim_results['1m']}
    most_expansive_month = max(month_vols, key=month_vols.get)
    print("Monthly (22-day) horizon: Most Expansive (highest volatility) = {}".format(most_expansive_month))
