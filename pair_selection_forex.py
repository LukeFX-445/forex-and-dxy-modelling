# --- Minimal fixes: (1) auto_adjust=False for yfinance; (2) 403-safe interest-rate fetch; (3) rescale=False in 2nd GARCH; (4) local returns fallback ---
!pip install yfinance --quiet
import yfinance as yf
import pandas as pd
from io import StringIO

# Define currency tickers (spot FX on Yahoo Finance; avoids contract expiry)
tickers = {
    'EUR': 'EURUSD=X',   # EUR/USD spot
    'GBP': 'GBPUSD=X',   # GBP/USD spot
    'JPY': 'USDJPY=X',   # USD/JPY spot (USD base)
    'CAD': 'USDCAD=X',   # USD/CAD spot (USD base)
    'CHF': 'USDCHF=X',   # USD/CHF spot (USD base)
    'USD': 'DX-Y.NYB'    # ICE US Dollar Index (non-expiring index)
}

# Fetch daily OHLCV for the last 2 years
data = {}
for name, ticker in tickers.items():
    df = yf.download(ticker, period='2y', interval='1d', auto_adjust=False, progress=False)
    df['Currency'] = name
    data[name] = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Concatenate all into one DataFrame
combined_df = pd.concat(data.values(), keys=data.keys(), names=['Currency', 'Date'])
combined_df.reset_index(inplace=True)
# Keeping original filename for compatibility; now contains spot data.
combined_df.to_csv('currency_futures_data.csv', index=False)

# ADD: Dual CSV save for clarity (keeps original too)
try:
    combined_df.to_csv('currency_spot_data.csv', index=False)
    combined_df.to_csv('currency_spot_data.csv.gz', index=False, compression='gzip')
except Exception as _e:
    pass


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
data = yf.download(
    tickers,
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval="1d",
    auto_adjust=False,
    progress=False
)
# 'data' will be a multi-index DataFrame: columns like ('Adj Close','EURUSD=X'), etc.
# We'll use the 'Close' prices for analysis (for currencies, Adj Close == Close as there's no corporate action).
closes = data['Close'].copy()

# Drop any rows with missing values (e.g., holidays)
closes.dropna(inplace=True)

# ADD: Business-day alignment & NaN hygiene (non-destructive; creates new frames)
try:
    bindex = pd.bdate_range(closes.index.min(), closes.index.max(), tz=getattr(closes.index, "tz", None))
    closes_bday = closes.reindex(bindex)
    closes_bday_clean = closes_bday.ffill(limit=3)
except Exception as _e:
    closes_bday_clean = closes.copy()

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

# ADD: Synthetic USD basket (5-leg DXY-like approximation), keep existing USD_Index
try:
    r_basket = pd.DataFrame({
        'EURUSD': closes['EURUSD'].pct_change(),
        'USDJPY': (1.0 / closes['JPYUSD']).pct_change(),
        'GBPUSD': closes['GBPUSD'].pct_change(),
        'USDCAD': (1.0 / closes['CADUSD']).pct_change(),
        'USDCHF': (1.0 / closes['CHFUSD']).pct_change()
    }).dropna()
    weights = {'EURUSD': -0.576, 'USDJPY': 0.136, 'GBPUSD': -0.119, 'USDCAD': 0.091, 'USDCHF': 0.036}
    basket_ret = sum(weights[k] * r_basket[k] for k in weights)
    USD_Basket5 = (1.0 + basket_ret).cumprod()
    closes['USD_Basket5'] = USD_Basket5.reindex(closes.index).ffill()
except Exception as _e:
    pass

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

# ADD: Heavier-tail GARCH alternative (Student-t) and choose by AIC; leaves original vol_forecast untouched
vol_forecast_enh = {}
try:
    for cur in currencies:
        series_pct = returns[cur] * 100.0
        am_norm = arch_model(series_pct, vol='Garch', p=1, q=1, dist='normal', rescale=False)
        res_norm = am_norm.fit(disp='off')
        am_t = arch_model(series_pct, vol='Garch', p=1, q=1, dist='t', rescale=False)
        res_t = am_t.fit(disp='off')
        best = res_t if res_t.aic < res_norm.aic else res_norm
        fcast = best.forecast(horizon=1)
        sigma_next = float(np.sqrt(fcast.variance.values[-1, 0]) / 100.0)
        vol_forecast_enh[cur] = sigma_next
except Exception as _e:
    vol_forecast_enh = {}

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
import requests  # added for 403-safe scraping

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
    # Scrape the TradingEconomics interest rate table with a fallback that avoids 403
    url = "https://tradingeconomics.com/country-list/interest-rate"
    tables = []
    try:
        tables = pd.read_html(url)
    except Exception:
        try:
            hdrs = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            }
            html = requests.get(url, headers=hdrs, timeout=10).text
            tables = pd.read_html(StringIO(html))
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
        row = df_rates[df_rates['Country'].str.contains(country, case=False, na=False)]
        if not row.empty:
            # Assume "Last" column has the current interest rate (as float or string)
            try:
                rate_value = float(str(row.iloc[0]['Last']).strip().replace('%',''))
            except Exception:
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
            hist = yf.download(ticker, period="10d", interval="1d", auto_adjust=False, progress=False)
        except Exception as e:
            print(f"Error downloading data for {curr}: {e}")
            continue
        if hist.empty:
            continue
        # Use the last available close price
        last_price = hist['Close'].iloc[-1]
        price_data[curr] = last_price

        # Compute 5-day average return (momentum)
        rets = hist['Close'].pct_change().dropna()
        if len(rets) >= 5:
            last5 = rets.iloc[-5:]
            avg_return = last5.mean()
        else:
            avg_return = rets.mean()
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

# ADD: ForexFactory calendar (secondary source) + interest rate fallback via ForexFactory
!pip install beautifulsoup4 lxml html5lib -q

import json, re, time
import requests
from bs4 import BeautifulSoup

def _norm_ccy_map(ccy_text):
    c = (ccy_text or '').strip().upper()
    if c == 'USD': return 'USD_Index'
    if c == 'EUR': return 'EURUSD'
    if c == 'GBP': return 'GBPUSD'
    if c == 'JPY': return 'JPYUSD'
    if c == 'CAD': return 'CADUSD'
    if c == 'CHF': return 'CHFUSD'
    return None

def fetch_ff_calendar(start_dt, end_dt):
    """
    Try ForexFactory JSON, then HTML calendar as fallback.
    Returns a DataFrame with at least ['date','time','currency','title','impact'].
    """
    sess = requests.Session()
    hdrs = {"User-Agent": "Mozilla/5.0"}
    # Attempt JSON feed for 'this week'
    try:
        url_json = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        r = sess.get(url_json, headers=hdrs, timeout=10)
        if r.ok:
            js = r.json()
            rows = []
            for ev in js:
                try:
                    d = ev.get('date','')
                    t = ev.get('time','')
                    cur = ev.get('currency','')
                    ttl = ev.get('title','')
                    imp = ev.get('impact','')
                    # Parse date to datetime.date if possible
                    dd = pd.to_datetime(d, errors='coerce').date() if d else None
                    if dd is not None and start_dt <= dd <= end_dt:
                        rows.append({'date': dd, 'time': t, 'currency': cur, 'title': ttl, 'impact': imp})
                except Exception:
                    continue
            if rows:
                return pd.DataFrame(rows)
    except Exception:
        pass
    # HTML fallback: week view
    try:
        url_html = "https://www.forexfactory.com/calendar?week=this"
        r = sess.get(url_html, headers=hdrs, timeout=10)
        if not r.ok:
            return pd.DataFrame()
        soup = BeautifulSoup(r.text, "lxml")
        rows = []
        # FF frequently uses rows with 'calendar__row' / 'calendar_row' classes; parse broadly
        for tr in soup.find_all(['tr','div']):
            txt = tr.get_text(" ", strip=True).lower()
            if not txt:
                continue
            # Heuristic grabs:
            ccy = None; impact = None; title = None; date_guess = None; time_guess = None
            cur_el = tr.find(lambda tag: tag.name in ['td','span','div'] and (tag.get('class') and any('calendar__currency' in c for c in tag.get('class'))))
            if cur_el and cur_el.get_text(strip=True):
                ccy = cur_el.get_text(strip=True)
            # Impact often tagged with icons/classes; fallback to text keys
            if 'high impact' in txt or 'high' in txt:
                impact = 'High'
            elif 'medium' in txt:
                impact = 'Medium'
            elif 'low' in txt:
                impact = 'Low'
            # Title
            ttl_el = tr.find(lambda tag: tag.name in ['td','a','span','div'] and (tag.get('class') and any('calendar__event' in c for c in tag.get('class'))))
            if ttl_el and ttl_el.get_text(strip=True):
                title = ttl_el.get_text(strip=True)
            # Date/time heuristics
            # (FF repeats dates; we won't overfit — just store None if uncertain)
            if ccy and title:
                rows.append({'date': date_guess, 'time': time_guess, 'currency': ccy, 'title': title, 'impact': impact or ''})
        if rows:
            df = pd.DataFrame(rows)
            # Filter out empties where possible
            return df
    except Exception:
        pass
    return pd.DataFrame()

def fetch_interest_rates_ff(days_back=365):
    """
    Parse ForexFactory events for 'Interest Rate' actuals over the past year and
    return a dict of latest values per currency in percent.
    """
    start_dt = (dt.date.today() - dt.timedelta(days=days_back))
    end_dt = dt.date.today() + dt.timedelta(days=7)
    try:
        df = fetch_ff_calendar(start_dt, end_dt)
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    df = df.copy()
    df['title_l'] = df['title'].astype(str).str.lower()
    mask = df['title_l'].str.contains('interest rate|policy rate|refinancing rate|bank rate|overnight rate|target rate', regex=True, na=False)
    df_rates = df[mask].copy()
    out = {}
    # Without structured 'actual' in HTML scrape, we can't reliably parse numbers.
    # However, JSON feed sometimes carries 'impact' only. We'll heuristically extract trailing % in title if present.
    for _, r in df_rates.sort_values('date').iterrows():
        ccy_raw = str(r.get('currency','')).strip().upper()
        ccy = ccy_raw.replace('EURO','EUR') if ccy_raw else ccy_raw
        val = None
        m = re.search(r'(-?\d+(\.\d+)?)\s*%$', str(r.get('title','')))
        if m:
            try:
                val = float(m.group(1))
            except Exception:
                val = None
        # Map into our 3-letter set
        key = None
        if ccy in ['USD','EUR','GBP','JPY','CAD','CHF']:
            key = ccy
        if key and val is not None:
            out[key] = val
    return out

# Merge ForexFactory interest-rate fallback
try:
    ff_rates = fetch_interest_rates_ff()
    if ff_rates:
        for k, v in ff_rates.items():
            if k not in rates or rates[k] is None:
                rates[k] = v
except Exception:
    pass

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
            hist = yf.download(ticker, period=f"{hist_days}d", interval="1d", auto_adjust=False, progress=False)
            retns = hist['Close'].pct_change().dropna()
            # Adjust sign for reverse pairs as before
            if curr in ['JPY', 'CAD', 'CHF']:
                retns = -retns
            if not retns.empty:
                # Fit GARCH(1,1)
                am = arch_model(retns * 100, p=1, q=1, rescale=False)
                res = am.fit(disp='off')
                # Forecast volatility for the next day
                forecast = res.forecast(horizon=1)
                var_forecast = forecast.variance.iloc[-1, 0]
                vol = np.sqrt(var_forecast) / 100.0  # convert back to fraction (since we scaled returns by 100)
    except Exception as e:
        print(f"GARCH model failed for {curr}: {e}")
    if vol is None:
        # Fallback: use sample std dev of *local* returns (not global)
        vol = retns.std() if 'retns' in locals() else 0.0
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

# ADD: Outlier-controlled (winsorized) returns snapshot (not used downstream)
try:
    returns_wins = returns.copy()
    for c in returns_wins.columns:
        q1, q99 = returns_wins[c].quantile([0.01, 0.99])
        returns_wins[c] = returns_wins[c].clip(q1, q99)
    returns_wins.to_csv('returns_winsorized.csv')
except Exception as _e:
    pass

# ADD: Realized volatility comparators (Parkinson 10-day, EWMA 20-day)
rv_parkinson_10 = {}
rv_ewma_20 = {}
try:
    window_p = 10
    for col in currencies:
        if col in ['EURUSD', 'GBPUSD']:
            H = highs[col + '=X']
            L = lows[col + '=X']
            hl2 = (np.log(H / L) ** 2).rolling(window_p).mean()
            rv_parkinson_10[col] = float(np.sqrt(hl2 / (4.0 * np.log(2.0))).iloc[-1])
            rv_ewma_20[col] = float(returns[col].ewm(span=20).std().iloc[-1])
        elif col == 'USD_Index':
            H = highs['UUP']
            L = lows['UUP']
            hl2 = (np.log(H / L) ** 2).rolling(window_p).mean()
            rv_parkinson_10[col] = float(np.sqrt(hl2 / (4.0 * np.log(2.0))).iloc[-1])
            rv_ewma_20[col] = float(returns[col].ewm(span=20).std().iloc[-1])
        else:  # inverted quotes: JPYUSD/CADUSD/CHFUSD
            qmap = {'JPYUSD': 'USDJPY=X', 'CADUSD': 'USDCAD=X', 'CHFUSD': 'USDCHF=X'}
            q = qmap.get(col)
            if q is not None:
                inv_high = 1.0 / lows[q]
                inv_low = 1.0 / highs[q]
                hl2 = (np.log(inv_high / inv_low) ** 2).rolling(window_p).mean()
                rv_parkinson_10[col] = float(np.sqrt(hl2 / (4.0 * np.log(2.0))).iloc[-1])
                rv_ewma_20[col] = float(returns[col].ewm(span=20).std().iloc[-1])
except Exception as _e:
    pass

# ADD: ADR enhancements — True Range ADR(14) + percent ADR + alignment to returns index
try:
    def _true_range(H, L, C):
        prevC = C.shift(1)
        tr1 = (H - L).abs()
        tr2 = (H - prevC).abs()
        tr3 = (L - prevC).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    highs_al = highs.reindex(returns.index)
    lows_al  = lows.reindex(returns.index)
    adr_true_14 = {}
    adr_pct_14  = {}
    # Directs
    for k in ['EURUSD','GBPUSD']:
        pair = k + '=X'
        H = highs_al[pair]; L = lows_al[pair]; C = closes[k].reindex(returns.index)
        tr = _true_range(H, L, C).rolling(14).mean()
        adr_true_14[k] = float(tr.iloc[-1])
        adr_pct_14[k]  = float((tr / C).iloc[-1])
    # USD_Index via UUP
    H = highs_al['UUP']; L = lows_al['UUP']; C = closes['USD_Index'].reindex(returns.index)
    tr = _true_range(H, L, C).rolling(14).mean()
    adr_true_14['USD_Index'] = float(tr.iloc[-1]); adr_pct_14['USD_Index'] = float((tr / C).iloc[-1])
    # Inverted: derive H/L/C by inversion
    inv_map = {'JPYUSD': 'USDJPY=X', 'CADUSD': 'USDCAD=X', 'CHFUSD': 'USDCHF=X'}
    for k, pair in inv_map.items():
        Hq = highs_al[pair]; Lq = lows_al[pair]; Cq = closes[k].reindex(returns.index)  # Cq already inverted price
        # Inversion: high of base = 1/low of quote; low of base = 1/high of quote
        H = 1.0 / Lq
        L = 1.0 / Hq
        C = Cq
        tr = _true_range(H, L, C).rolling(14).mean()
        adr_true_14[k] = float(tr.iloc[-1])
        adr_pct_14[k]  = float((tr / C).iloc[-1])
    # Save snapshots
    pd.Series(adr_true_14).to_csv('adr_true_14.csv')
    pd.Series(adr_pct_14).to_csv('adr_pct_14.csv')
except Exception as _e:
    pass

# ADD: USD ADR Unification — prefer DX-Y.NYB H/L if available (fallback to UUP)
try:
    dxy_hl = yf.download("DX-Y.NYB", start=start_date.strftime('%Y-%m-%d'),
                         end=end_date.strftime('%Y-%m-%d'),
                         interval="1d", auto_adjust=False, progress=False)
    if not dxy_hl.empty:
        adr_usd_dxy = float((dxy_hl['High'] - dxy_hl['Low']).tail(N).mean())
        # Keep original adr['USD_Index'] but store unified version too
        adr['USD_Index_unified'] = adr_usd_dxy
        pd.Series(adr).to_csv('adr_last10_with_usd_unified.csv')
except Exception as _e:
    pass

# ADD: Correlation & simple PCA diagnostic (save to disk)
try:
    corr_mat = returns[currencies].corr()
    corr_mat.to_csv('fx_correlation_matrix.csv')
    # PCA via eig on standardized covariance
    R = (returns[currencies] - returns[currencies].mean()) / returns[currencies].std()
    cov = np.cov(R.dropna().T)
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    pc1_loadings = pd.Series(vecs[:, 0], index=currencies)
    pc1_series = R.dot(pc1_loadings)
    # Compare PC1 to USD proxies
    comp = {}
    try:
        comp['corr_PC1_USD_Index'] = float(pc1_series.corr(returns['USD_Index']))
    except Exception:
        pass
    if 'USD_Basket5' in closes.columns:
        usd_basket_ret = np.log(closes['USD_Basket5'] / closes['USD_Basket5'].shift(1)).reindex(pc1_series.index)
        comp['corr_PC1_USD_Basket5'] = float(pc1_series.corr(usd_basket_ret.dropna()))
    pd.Series(comp).to_csv('pca_usd_factor_compare.csv')
    pc1_loadings.to_csv('pca_pc1_loadings.csv')
except Exception as _e:
    pass

# ADD: Best-of (GARCH / EGARCH / GJR-GARCH) with Student-t; fallback to previous
vol_forecast_best = {}
try:
    for cur in currencies:
        sp = returns[cur] * 100.0
        fits = []
        try:
            m1 = arch_model(sp, vol='Garch', p=1, q=1, dist='t', rescale=False).fit(disp='off'); fits.append(m1)
        except Exception: pass
        try:
            m2 = arch_model(sp, vol='EGARCH', p=1, q=1, dist='t', rescale=False).fit(disp='off'); fits.append(m2)
        except Exception: pass
        try:
            m3 = arch_model(sp, vol='Garch', p=1, o=1, q=1, dist='t', rescale=False).fit(disp='off'); fits.append(m3)  # GJR via o=1
        except Exception: pass
        if fits:
            best = min(fits, key=lambda r: r.aic)
            f = best.forecast(horizon=1)
            vol_forecast_best[cur] = float(np.sqrt(f.variance.values[-1,0]) / 100.0)
except Exception:
    pass

# Consolidate volatility choice (best -> enh -> simple)
volatility_best = {}
for cur in currencies:
    volatility_best[cur] = vol_forecast_best.get(cur, vol_forecast_enh.get(cur, volatility.get(cur, 0.0)))

# ADD: Pair-level carry drift (interest differentials) — keeps original drift_components intact
def _carry_drift_for_key(key):
    # convert interest % to daily fraction
    def r(c): 
        return None if (rates.get(c) is None) else (rates[c] / 100.0 / 252.0)
    if key == 'EURUSD': return ( (r('EUR') or 0.0) - (r('USD') or 0.0) )
    if key == 'GBPUSD': return ( (r('GBP') or 0.0) - (r('USD') or 0.0) )
    if key == 'JPYUSD': return ( (r('JPY') or 0.0) - (r('USD') or 0.0) )
    if key == 'CADUSD': return ( (r('CAD') or 0.0) - (r('USD') or 0.0) )
    if key == 'CHFUSD': return ( (r('CHF') or 0.0) - (r('USD') or 0.0) )
    if key == 'USD_Index':
        # approximate vs basket of majors
        others = [c for c in ['EUR','GBP','JPY','CAD','CHF'] if rates.get(c) is not None]
        if not others or rates.get('USD') is None:
            return 0.0
        return ( (rates['USD']/100.0/252.0) - np.mean([rates[o]/100.0/252.0 for o in others]) )
    return 0.0

drift_components_pair = {}
for key in currencies:
    mom = momentum.get(key.replace('USD','').replace('Index','').strip(), 0.0) if key in ['USD_Index'] else momentum.get(key.split('USD')[0], 0.0)
    # Momentum already computed per currency symbol block; keep original approach for consistency:
    mom = momentum.get({'EURUSD':'EUR','GBPUSD':'GBP','JPYUSD':'JPY','CADUSD':'CAD','CHFUSD':'CHF','USD_Index':'USD'}[key], 0.0)
    sent = sentiments.get({'EURUSD':'EUR','GBPUSD':'GBP','JPYUSD':'JPY','CADUSD':'CAD','CHFUSD':'CHF','USD_Index':'USD'}[key], 0.0) * 0.002
    carry = _carry_drift_for_key(key)
    drift_components_pair[key] = {
        'momentum': mom,
        'sentiment': sent,
        'carry': carry,
        'total': mom + sent + carry
    }

# ADD: Merge ForexFactory calendar into event boosts (keeps investpy; uses max boost)
try:
    ff_cal = fetch_ff_calendar(today, week_ahead)
except Exception:
    ff_cal = pd.DataFrame()

if ff_cal is not None and not ff_cal.empty:
    for _, r in ff_cal.iterrows():
        ccy = str(r.get('currency','')).upper()
        impacted_key = _norm_ccy_map(ccy)
        if impacted_key and impacted_key in event_vol_boost:
            # Treat "High" impact as high; otherwise smaller boost
            imp = str(r.get('impact','')).lower()
            boost = 1.3 if 'high' in imp else (1.15 if 'medium' in imp else 1.05)
            event_vol_boost[impacted_key] = max(event_vol_boost[impacted_key], boost)
            title = str(r.get('title','')).lower()
            if any(k in title for k in ['interest rate','policy rate','rate decision','central bank','fed','ecb','boe','boj','boc','snb']):
                event_drift_boost[impacted_key] = event_drift_boost.get(impacted_key, 0.0) + 0.001

# Store simulation outcomes (APPLY event boosts + best vol + pair-level carry)
sim_results = {h: {} for h in horizons}
for h, days in horizons.items():
    for curr in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'CHF']:
        if curr not in price_data or price_data[curr] is None:
            continue
        key_map = {'USD':'USD_Index','EUR':'EURUSD','GBP':'GBPUSD','JPY':'JPYUSD','CAD':'CADUSD','CHF':'CHFUSD'}
        key = key_map[curr]

        P0 = price_data[curr]
        if isinstance(P0, pd.Series):
            P0 = P0.item()

        # Base drift from pair-level components; add event drift boost here
        mu_base = drift_components_pair.get(key, {}).get('total', 0.0)
        mu = mu_base + event_drift_boost.get(key, 0.0)

        # Best sigma, multiplied by event vol boost
        sigma_base = volatility_best.get(key, volatility.get(key, 0.0))
        sigma = sigma_base * event_vol_boost.get(key, 1.0)

        # Simulate
        final_prices = simulate_prices(P0, mu, sigma, days=days, trials=trials)
        r = final_prices / P0 - 1.0
        sim_results[h][curr] = {
            'final_prices': final_prices,
            'mean_return': r.mean(),
            'median_return': np.median(r),
            'std_return': r.std()
        }

# ADD: Reproducible sims + percentiles/VaR/ES + expected-move bands (keeps original sim_results)
np.random.seed(42)
sim_results_plus = {h: {} for h in horizons}
try:
    for h, days in horizons.items():
        for curr in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'CHF']:
            if curr not in price_data or price_data[curr] is None:
                continue
            key_map = {'USD':'USD_Index','EUR':'EURUSD','GBP':'GBPUSD','JPY':'JPYUSD','CAD':'CADUSD','CHF':'CHFUSD'}
            key = key_map[curr]
            P0 = float(price_data[curr]) if not isinstance(price_data[curr], pd.Series) else float(price_data[curr].item())
            mu = float(drift_components_pair.get(key, {}).get('total', 0.0) + event_drift_boost.get(key, 0.0))
            sigma = float(volatility_best.get(key, volatility.get(key, 0.0)) * event_vol_boost.get(key, 1.0))
            finals = simulate_prices(P0, mu, sigma, days=days, trials=trials)
            r = finals / P0 - 1.0
            p5, p25, p50, p75, p95 = np.percentile(r, [5, 25, 50, 75, 95])
            var95 = -np.percentile(r, 5)
            es95 = -r[r <= np.percentile(r, 5)].mean() if np.isfinite(r).all() else np.nan
            mu_d = mu * days
            sigma_d = sigma * np.sqrt(days)
            band1_low = P0 * np.exp(mu_d - sigma_d)
            band1_high = P0 * np.exp(mu_d + sigma_d)
            band2_low = P0 * np.exp(mu_d - 2 * sigma_d)
            band2_high = P0 * np.exp(mu_d + 2 * sigma_d)
            sim_results_plus[h][curr] = {
                'p5': float(p5), 'p25': float(p25), 'p50': float(p50), 'p75': float(p75), 'p95': float(p95),
                'VaR_95': float(var95), 'ES_95': float(es95),
                'band1_low': float(band1_low), 'band1_high': float(band1_high),
                'band2_low': float(band2_low), 'band2_high': float(band2_high)
            }
except Exception as _e:
    pass

# ADD: Simple backtest logger — logs today’s forecast vol and fills yesterday’s realized
try:
    bt_file = 'fx_vol_backtest.csv'
    today_stamp = pd.Timestamp.utcnow().normalize()
    rows = []
    for cur in currencies:
        f = max(vol_forecast_best.get(cur, np.nan), vol_forecast_enh.get(cur, np.nan)) if not np.isnan(vol_forecast_best.get(cur, np.nan)) else vol_forecast_enh.get(cur, np.nan)
        if f is None or (isinstance(f,float) and np.isnan(f)): f = volatility.get(cur, np.nan)
        rows.append({'date': str(today_stamp.date()), 'currency': cur, 'forecast_vol': float(f), 'realized_abs_return': np.nan})
    new_df = pd.DataFrame(rows)
    try:
        old = pd.read_csv(bt_file)
        bt_all = pd.concat([old, new_df], ignore_index=True)
    except Exception:
        bt_all = new_df.copy()
    try:
        last_dt = returns.index[-1]
        realized_row = returns.loc[last_dt, [c for c in currencies if c in returns.columns]].abs().rename('realized_abs_return')
        mask = (bt_all['date'] == str(last_dt.date()))
        for cur in currencies:
            idxs = bt_all.index[mask & (bt_all['currency'] == cur)]
            if len(idxs):
                bt_all.loc[idxs, 'realized_abs_return'] = float(realized_row.get(cur, np.nan))
    except Exception:
        pass
    bt_all.drop_duplicates(subset=['date','currency'], keep='last', inplace=True)
    bt_all.to_csv(bt_file, index=False)
except Exception as _e:
    pass

# ADD: Run metadata snapshot for reproducibility
try:
    import json, sys
    meta = {
        'timestamp_utc': pd.Timestamp.utcnow().isoformat(),
        'python': sys.version.split()[0],
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'yfinance': yf.__version__ if hasattr(yf, "__version__") else "unknown",
        'tickers_spot_block': tickers,
        'currencies_analyzed': currencies,
        'horizons': horizons
    }
    with open('run_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
except Exception as _e:
    pass

# ADD: Walk-forward scoring (last 60 obs): sign hit-rate, RMSE, 1σ coverage
try:
    wf_metrics = []
    lookback_mu = 5
    lookback_sig = 20
    W = 60
    for key in currencies:
        s = returns[key].copy().dropna()
        if len(s) < (lookback_sig + W + 5):
            continue
        mu_pred = s.rolling(lookback_mu).mean().shift(1)
        sig_pred = s.ewm(span=lookback_sig).std().shift(1)
        # Evaluate last W obs
        y = s.iloc[-W:]
        m = mu_pred.reindex(y.index)
        sd = sig_pred.reindex(y.index)
        # Metrics
        hit = float((np.sign(y) == np.sign(m)).mean())
        rmse = float(np.sqrt(((y - m)**2).mean()))
        cover_1sigma = float(((y - m).abs() <= sd).mean())
        cover_95 = float(((y - m).abs() <= 1.96*sd).mean())
        wf_metrics.append({'series': key, 'hit_rate': hit, 'rmse': rmse, 'coverage_1sigma': cover_1sigma, 'coverage_95': cover_95})
    if wf_metrics:
        pd.DataFrame(wf_metrics).to_csv('walkforward_metrics.csv', index=False)
except Exception as _e:
    pass

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

# ADD: Quick summary of walk-forward metrics (prints)
try:
    wf = pd.read_csv('walkforward_metrics.csv')
    best_hit = wf.sort_values('hit_rate', ascending=False).head(1).iloc[0]
    print("Walk-forward: Best hit-rate = {:.1f}% on {}".format(100*best_hit['hit_rate'], best_hit['series']))
    print("Walk-forward: Avg 1σ coverage = {:.1f}%".format(100*wf['coverage_1sigma'].mean()))
except Exception:
    pass
