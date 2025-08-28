"""
rate_curve.py – Lecture du fichier rates.xlsx, construction des courbes zéro USD/EUR
et utilitaires pour obtenir (r_d, r_f) à n’importe quelle date de maturité.
Tout est encapsulé dans ce module : une fois importé, un simple

    from rate_curve import rd_rf, plot_curves

suffit dans le reste de l’application.
"""
import datetime as _dt
from functools import lru_cache
from typing import List, Tuple, Mapping, BinaryIO, Union

import numpy as _np
import pandas as _pd
from scipy.interpolate import PchipInterpolator as _Pchip


# Configuration — adapte ici si le fichier/mapping change

RATES_FILE: str = "rates.xlsx"   # fichier Excel dans le même dossier que app.py
SHEET_NAME: int | str = 0        # onglet à lire

# Colonnes après header=1 (Tenors, Bid, Ask, Tenors.1, Bid.1, Ask.1, …)
USD_COLS: Mapping[str, Union[int, str]] = {"tenor": "Tenors.1", "bid": "Bid.1", "ask": "Ask.1"}
EUR_COLS: Mapping[str, Union[int, str]] = {"tenor": "Tenors.2", "bid": "Bid.2", "ask": "Ask.2"}

_DAYCOUNT_360: float = 360.0
_EPS: float = 1.0e-8


# Helper conversions


def tenor_to_days(tenor: str) -> int:
    t = tenor.strip().upper()
    if t in {"O/N", "ON", "T/N", "TN", "S/N", "SN"}:
        return 1
    qty, unit = int(t[:-1]), t[-1]
    return {"W": 7 * qty, "M": 30 * qty, "Y": 360 * qty}[unit]


def _days_fraction(days, base=_DAYCOUNT_360):
    return _np.asarray(days, dtype=float) / base


def _depo_to_discount(rate_dec, delta):
    return 1.0 / (1.0 + rate_dec * delta)


def _build_zero_interpolator(deltas, zeros):
    zeros = _np.maximum(zeros, 0.0)
    return _Pchip(deltas, zeros, extrapolate=True)


# ZeroCurve

class ZeroCurve:
    def __init__(self, interpolator: _Pchip, daycount: float = _DAYCOUNT_360):
        self._i = interpolator
        self._dc = float(daycount)

    def zero(self, t: float) -> float:
        t = float(max(t, _EPS))
        return float(self._i(t))

    def discount(self, t: float) -> float:
        return _np.exp(-self.zero(t) * t)

    # Helpers dates
    def _yearfrac(self, target: _dt.date, today: _dt.date) -> float:
        return (target - today).days / self._dc

    def zero_date(self, target: _dt.date, today: _dt.date | None = None) -> float:
        today = _dt.date.today() if today is None else today
        return self.zero(self._yearfrac(target, today))


# Construction à partir d’un DataFrame


def _clean_percent(col: _pd.Series) -> _pd.Series:
    return (
        col.astype(str)
           .str.replace("%", "", regex=False)
           .str.replace(",", ".", regex=False)
           .astype(float)
    )


def _label(df: _pd.DataFrame, col: Union[int, str]) -> str:
    return df.columns[col] if isinstance(col, int) else col


def _df_to_quotes(df, tenor_col, bid_col, ask_col):
    tenor_col = _label(df, tenor_col)
    bid_col   = _label(df, bid_col)
    ask_col   = _label(df, ask_col)

    sub = df[[tenor_col, bid_col, ask_col]].copy()
    sub = sub.dropna()                           # enlève lignes vides
    sub[bid_col] = _clean_percent(sub[bid_col])
    sub[ask_col] = _clean_percent(sub[ask_col])
    sub["mid"]   = (sub[bid_col] + sub[ask_col]) / 2.0
    sub["days"]  = sub[tenor_col].apply(tenor_to_days)

    # ---- NEW : tri + suppression duplicates ------------------
    sub = sub.sort_values("days")
    sub = sub.drop_duplicates(subset="days", keep="first")
    # ----------------------------------------------------------

    return list(sub[[tenor_col, "mid", "days"]].itertuples(index=False, name=None))



def _build_curve(quotes):
    _, rates_pct, days = zip(*quotes)
    rates_dec = _np.asarray(rates_pct) / 100.0
    days_arr = _np.asarray(days, dtype=float)
    deltas = _days_fraction(days_arr)
    dfs = _depo_to_discount(rates_dec, deltas)
    zeros = -_np.log(dfs) / _np.maximum(deltas, _EPS)
    return ZeroCurve(_build_zero_interpolator(deltas, zeros))


#  Chargement & cache global


@lru_cache(maxsize=1)
def _load_curves() -> Tuple[ZeroCurve, ZeroCurve]:
    """Lis rates.xlsx, construit et met en cache les deux courbes."""
    df = _pd.read_excel(RATES_FILE, sheet_name=SHEET_NAME, header=1, engine="openpyxl")
    usd_quotes = _df_to_quotes(df, USD_COLS["tenor"], USD_COLS["bid"], USD_COLS["ask"])
    eur_quotes = _df_to_quotes(df, EUR_COLS["tenor"], EUR_COLS["bid"], EUR_COLS["ask"])
    return _build_curve(usd_quotes), _build_curve(eur_quotes)


#  API simples pour l’extérieur


def rd_rf(maturity_date: _dt.date, *, today: _dt.date | None = None) -> Tuple[float, float]:
    """Retourne (r_d, r_f) pour une date de maturité.

    Exemple
    -------
    >>> from rate_curve import rd_rf
    >>> r_d, r_f = rd_rf(date(2025, 12, 15))
    """
    today = _dt.date.today() if today is None else today
    curve_d, curve_f = _load_curves()
    t = (maturity_date - today).days / _DAYCOUNT_360
    return curve_d.zero(t), curve_f.zero(t)


def plot_curves(num_pts: int = 181):
    """Trace les courbes zéro et renvoie la figure matplotlib."""
    import matplotlib.pyplot as _plt
    curve_d, curve_f = _load_curves()
    ts = _np.linspace(1 / 360, 1.0, num_pts)  # jusqu’à 1 an
    _plt.figure(figsize=(7, 4))
    _plt.plot(ts * 360, [curve_d.zero(t) * 100 for t in ts], label="USD zero")
    _plt.plot(ts * 360, [curve_f.zero(t) * 100 for t in ts], label="EUR zero")
    _plt.xlabel("Maturité (jours)")
    _plt.ylabel("Taux zéro (%)")
    _plt.legend()
    _plt.grid(True)
    _plt.tight_layout()
    return _plt.gcf()


#  Public symbols

__all__ = [
    "ZeroCurve",
    "rd_rf",
    "plot_curves",
    "tenor_to_days",
]
