import yfinance as yf
import numpy as np
import pandas as pd

def get_eurusd_history(period="1y"):
    """Télécharge les données de clôture EUR/USD sur la période choisie ('1y', '6mo', ...)"""
    df = yf.download("EURUSD=X", period=period)
    return df[["Close"]]

def get_historical_vol(symbol="EURUSD=X", days=30):
    """Calcule la volatilité historique annualisée sur X derniers jours pour EUR/USD."""
    data = yf.download(symbol, period=f"{days+1}d")["Close"]
    returns = np.log(data / data.shift(1)).dropna()
    vol_daily = returns.std()
    vol_annualized = vol_daily * np.sqrt(252)
    return float(vol_annualized)


df_fx = get_eurusd_history(period="1y")
current_spot = float(df_fx["Close"].iloc[-1])
print(current_spot)