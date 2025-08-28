# price.py — Pricer d’options FX (Streamlit) avec PINN Vanille & PINN Asian (arithmétique)

import datetime as dt
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
import math

#  modules internes (ton repo)
from fx_data import get_historical_vol, get_eurusd_history
from inter_rates import plot_curves
from methods import (
    FXOption,
    AsianOption,
    HestonOption,
    HestonAsianOption,
    # moteurs MC
    MonteCarloGK,
    MonteCarloAsian,
    MonteCarloHeston,
    MonteCarloHestonAsian,
    # formules fermées / alias pratiques
    garman_kohlhagen_price,
    heston_price,
    # PINN Vanille (GK)
    load_pinn_model,
    PINNGKEnginePriceOnly,
    # PINN Asian (arithmétique, GK)
    load_asian_pinn_model,
    PINNGKAsianPriceOnly,
)

st.set_page_config(page_title="Pricer FX Options", layout="wide")
st.title("Pricer d'options FX")

# 1) Données marché : Spot & vol réalisée
df_fx = get_eurusd_history(period="1y")

# Harmonisation colonne 'Close'
if isinstance(df_fx.columns, pd.MultiIndex):
    close_cols = df_fx.columns[df_fx.columns.get_level_values(1) == "Close"]
    if close_cols.size == 0:
        close_cols = df_fx.columns[df_fx.columns.get_level_values(0) == "Close"]
    if close_cols.size:
        df_fx = df_fx[close_cols].copy()
        df_fx.columns = ["Close"]
else:
    if ("Close" not in df_fx.columns) and ("Adj Close" in df_fx.columns):
        df_fx = df_fx.rename(columns={"Adj Close": "Close"})
if ("Close" not in df_fx.columns) and (df_fx.shape[1] == 1):
    df_fx.columns = ["Close"]

if ("Close" in df_fx.columns) and (not df_fx["Close"].empty):
    df_fx = df_fx.dropna(subset=["Close"]).copy()
    df_fx["Close"] = df_fx["Close"].astype(float)

    if not pd.api.types.is_datetime64_any_dtype(df_fx.index):
        df_fx.index = pd.to_datetime(df_fx.index)

    current_spot = float(df_fx["Close"].iloc[-1])

    # Vol annualisée (30 j)
    log_ret = np.log(df_fx["Close"] / df_fx["Close"].shift(1)).dropna()
    rolling_vol = (log_ret.rolling(window=30).std() * np.sqrt(252)).dropna()

    st.subheader("EUR/USD – Prix & Volatilité réalisée (30 j)")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3]
    )
    fig.add_trace(go.Scatter(x=df_fx.index, y=df_fx["Close"],
                             mode="lines", line=dict(color="#F3A006", width=2.5),
                             name="EUR/USD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol,
                             mode="lines", line=dict(color="#00CCFF", width=2),
                             name="Vol 30 j"), row=2, col=1)
    fig.update_layout(plot_bgcolor="#000000", paper_bgcolor="#000000",
                      font_color="#FFFFFF", height=550,
                      margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(showgrid=True, gridcolor="#333333", tickformat="%b %Y")
    fig.update_yaxes(range=[0.9, 1.3], showgrid=True, gridcolor="#333333",
                     tickformat=".4f", row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#333333",
                     tickformat=".2%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"Spot EUR/USD actuel : **{current_spot:.5f}**")
else:
    st.warning("Impossible de récupérer une colonne 'Close' valide.")
    current_spot = 1.10  # défaut

with st.sidebar:
    st.header("Paramètres de l'option FX")

    # Spot & Strike
    spot = current_spot
    st.markdown(f"**Spot EUR/USD** : {spot:.5f}")
    strike = st.number_input("Strike", value=spot, format="%.5f")

    # Notional
    volume = st.number_input("Notional (€ ou USD)", value=1e6, step=1e4, format="%.0f")

    # Maturité (date)
    maturity_date = st.date_input("Date de maturité", value=dt.date.today(), format="DD-MM-YYYY")
    adjusted_date = maturity_date
    if adjusted_date.weekday() >= 5:
        st.warning("Date sur week-end → ajustée au jour ouvrable précédent.")
        while adjusted_date.weekday() >= 5:
            adjusted_date -= dt.timedelta(days=1)
        st.info(f"Date ajustée : {adjusted_date:%d-%m-%Y}")
    maturity = max((adjusted_date - dt.date.today()).days, 1) / 365.0  # clamp à ≥ 1 jour

    # Taux domestique/étranger (fixes ici)
    r_d, r_f = 0.04119, 0.01848
    st.write(f"r_d : {r_d:.4%} | r_f : {r_f:.4%}")

    # Style d'option
    opt_style = st.radio("Style d'option", ("Vanille", "Asian géométrique", "Asian arithmétique"))
    is_call = st.radio("Type", ("Call", "Put")) == "Call"

    # Choix du modèle — CONTRAINTE DEMANDÉE :
    #  - Si Vanille → PINN (GK) possible
    #  - Si Asian arithmétique → PINN Asian possible
    #  - Si Asian géométrique → pas de PINN Asian (pas adapté)
    model_options = ["GK (vol constante)", "Heston GK (vol stochastique)"]
    if opt_style == "Vanille":
        model_options.append("PINN (GK, checkpoint)")
    elif opt_style == "Asian arithmétique":
        model_options.append("PINN Asian (GK, checkpoint)")
    model = st.radio("Modèle de pricing", tuple(model_options))

    # Paramètres Heston si sélectionné
    if model.startswith("Heston GK"):
        st.subheader("Paramètres Heston (calibrés — modifiables)")
        kappa = st.number_input("κ (vitesse de rappel)", value=5.000000, format="%.6f")
        theta = st.number_input("θ (variance long terme)", value=0.007906, format="%.6f")
        sigma = st.number_input("σ (vol de la variance)", value=0.281170, format="%.6f")
        rho   = st.number_input("ρ (corrélation)",       value=-0.325057, format="%.6f")
        v0    = st.number_input("v₀ (var. initiale)",    value=0.010390, format="%.6f")
    else:
        kappa = theta = sigma = rho = v0 = None

    # PINN Vanille (GK, σ-aware) — seulement quand opt_style == "Vanille"
    if model == "PINN (GK, checkpoint)":
        st.subheader("PINN (GK, checkpoint)")
        ckpt_path = st.text_input("Fichier .pt (vanille)", value="fx_pinn_sigma.pt")

        # Constantes d'entraînement (doivent matcher le checkpoint)
        T_train   = 1.3
        r_d_train = 0.041190
        r_f_train = 0.01848
        K_ref     = strike
        sigma_ref = 0.14
        with st.expander("Infos (constants d'entraînement — Vanille)", expanded=False):
            st.write({"T_train": T_train, "r_d_train": r_d_train, "r_f_train": r_f_train,
                      "K_ref": K_ref, "sigma_ref": sigma_ref})
            st.caption("Ces constantes doivent rester identiques à l'entraînement du PINN (vanille).")

        # Vol implicite utilisée par le PINN Vanille
        vol_pinn = st.number_input("Vol implicite PINN (Vanille) (%)", value=8.00, format="%.4f") / 100

        @st.cache_resource
        def _load_pinn_cached(path: str, K_ref: float, T_train: float, sigma_ref: float):
            return load_pinn_model(path, K_ref=K_ref, T_train=T_train, sigma_ref=sigma_ref, device="cpu")

        try:
            pinn_model = _load_pinn_cached(ckpt_path, K_ref, T_train, sigma_ref)
            st.success("PINN (vanille) chargé.")
        except Exception as e:
            pinn_model = None
            st.error(f"Chargement PINN (vanille) impossible : {e}")
    else:
        pinn_model = None
        vol_pinn   = None

    # PINN Asian (arithmétique, CALL) — seulement quand opt_style == "Asian arithmétique"
    if model == "PINN Asian (GK, checkpoint)":
        st.subheader("PINN Asian (GK, checkpoint)")
        ckpt_asian_path = st.text_input("Fichier .pt (Asian)", value="fx_asian_pinn.pt")

        # Constantes d'entraînement Asian (matcher ton script)
        r_d_train_asian = 0.04124
        with st.expander("Infos (constants d'entraînement — Asian)", expanded=False):
            st.write({
                "r_d_train": r_d_train_asian,
                "note": "Modèle entraîné pour CALL arithmétique uniquement (pas de PUT ni géométrique).",
            })

        # Vol implicite utilisée par le PINN Asian (σ-aware)
        vol_pinn_asian = st.number_input("Vol implicite PINN (Asian) (%)", value=8.00, format="%.4f") / 100

        @st.cache_resource
        def _load_asian_pinn_cached(path: str, hidden: int = 128, depth: int = 6):
            return load_asian_pinn_model(path, hidden=hidden, depth=depth, device="cpu")

        try:
            pinn_asian_model = _load_asian_pinn_cached(ckpt_asian_path)
            st.success("PINN Asian chargé.")
        except Exception as e:
            pinn_asian_model = None
            st.error(f"Chargement PINN Asian impossible : {e}")
    else:
        pinn_asian_model = None
        vol_pinn_asian   = None

    # Monte-Carlo (GK/Heston/Asian)
    n_paths = int(st.number_input("Nombre de trajectoires MC", value=100_000, step=10_000))
    n_steps_per_fix = 1  # tweakable si besoin (Asian Heston)

# 2) Volatilité (affichage selon modèle)
if model.startswith("GK"):
    st.subheader("Volatilité (GK)")
    vol_choice = st.radio("Volatilité manuelle ou historique ?", ("Manuelle", "Historique"))
    if vol_choice == "Manuelle":
        vol = st.number_input("Vol implicite (%)", value=7.50, format="%.4f") / 100
    else:
        days = st.selectbox("Période vol historique (jours)", [10, 15, 21, 30, 60, 90, 180, 252], index=3)
        if "vol_histo" not in st.session_state:
            st.session_state["vol_histo"] = None
        if st.button("Calculer vol. historique"):
            try:
                st.session_state["vol_histo"] = get_historical_vol(days=days)
                st.success(f"Vol historique : {st.session_state['vol_histo']:.2%}")
            except Exception as e:
                st.error(f"Erreur téléchargement : {e}")
        vol = st.session_state["vol_histo"]
else:
    vol = None  # non utilisé directement par Heston / PINN (hors champs dédiés ci-dessus)

# 3) Courbes de taux (visuel)
st.subheader("EUR & USD – Taux d'interêt")
fig = plot_curves()
st.plotly_chart(fig, use_container_width=True)

# 4) Récapitulatif des paramètres
st.write("### Paramètres choisis")
st.json({
    "spot": spot,
    "strike": strike,
    "maturity (ans)": maturity,
    "vol (GK)": vol if model.startswith("GK") else None,
    "vol (PINN vanille)": vol_pinn if model == "PINN (GK, checkpoint)" else None,
    "vol (PINN Asian)": vol_pinn_asian if model == "PINN Asian (GK, checkpoint)" else None,
    "r_d": r_d,
    "r_f": r_f,
    "style": opt_style,
    "call": is_call,
    "modèle": model,
    "n_paths": n_paths,
})

# 5) Pricing — bouton
can_price = (
    (model.startswith("GK") and (vol is not None))
    or model.startswith("Heston")
    or (model == "PINN (GK, checkpoint)" and (pinn_model is not None) and (vol_pinn is not None) and (vol_pinn > 0))
    or (model == "PINN Asian (GK, checkpoint)" and (pinn_asian_model is not None) and (vol_pinn_asian is not None) and (vol_pinn_asian > 0))
)
if st.button("Lancer Pricing", disabled=not can_price):

    # ==== GK (const-vol) ====
    if model.startswith("GK"):
        if opt_style == "Vanille":
            opt = FXOption(
                spot=spot, strike=strike, maturity=maturity,
                domestic_rate=r_d, foreign_rate=r_f,
                vol=vol, is_call=is_call, n_paths=n_paths,
            )
            pricer = MonteCarloGK(opt, n_paths=n_paths, seed=42)
            price_mc, se_mc = pricer.price()
            delta, _ = pricer.delta()
            price_analytic = garman_kohlhagen_price(opt)

            montant_mc = price_mc * volume
            st.subheader("Résultats (GK)")
            st.markdown(f"- **Maturité** : {adjusted_date:%d-%m-%Y}")
            st.markdown(f"- **Prix MC** : {price_mc:.6f} ± {1.96*se_mc:.6f} (IC 95 %)")
            st.markdown(f"- **Prix analytique** : {price_analytic:.6f}")
            st.markdown(f"- **Écart absolu** : {abs(price_mc - price_analytic):.6g}")
            st.markdown(f"- **Delta** : {delta:.6f}")
            st.markdown(f"- **Notionnel MC** : {montant_mc:,.2f}")
            st.markdown(f"- **Notionnel analytique** : {price_analytic * volume:,.2f}")
            st.markdown(f"- **Prime / Notionnel (MC)** : {price_mc*100:.4f}%")
            st.markdown(f"- **Prime / Notionnel (Analytique)** : {price_analytic*100:.4f}%")
            st.markdown(f"- **IC 95% (en % du notionnel)** : ± {1.96*se_mc*100:.4f}%")

        else:
            avg_type = "geometric" if opt_style.endswith("géométrique") else "arithmetic"
            opt = AsianOption(
                spot=spot, strike=strike, maturity=maturity,
                domestic_rate=r_d, foreign_rate=r_f,
                vol=vol, is_call=is_call, n_paths=n_paths,
                n_fixings=12, average_type=avg_type,
            )
            pricer = MonteCarloAsian(opt, n_paths=n_paths, seed=42)
            price_mc, se_mc = pricer.price()

            montant_mc = price_mc * volume
            st.subheader("Résultats (GK Asian)")
            st.markdown(f"- **Maturité** : {adjusted_date:%d-%m-%Y}")
            st.markdown(f"- **Prix MC** : {price_mc:.6f} ± {1.96*se_mc:.6f} (IC 95 %)")
            st.markdown(f"- **Notionnel MC** : {montant_mc:,.2f}")
            st.markdown(f"- **Prime / Notionnel (MC)** : {price_mc*100:.4f}%")
            st.markdown(f"- **IC 95% (en % du notionnel)** : ± {1.96*se_mc*100:.4f}%")

    # ==== Heston (vol-stoch) ====
    elif model.startswith("Heston"):
        if opt_style == "Vanille":
            opt = HestonOption(
                spot=spot, strike=strike, maturity=maturity,
                domestic_rate=r_d, foreign_rate=r_f,
                is_call=is_call, n_paths=n_paths,
                v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho,
            )
            price_cf = heston_price(opt, method="cf")
            pricer   = MonteCarloHeston(opt, n_paths=n_paths, n_steps=252, seed=42)
            price_mc, se_mc = pricer.price()
            delta, _ = pricer.delta()

            montant_mc = price_mc * volume
            st.subheader("Résultats (Heston)")
            st.markdown(f"- **Maturité** : {adjusted_date:%d-%m-%Y}")
            st.markdown(f"- **Prix MC** : {price_mc:.6f} ± {1.96*se_mc:.6f} (IC 95 %)")
            st.markdown(f"- **Prix semi-analytique** : {price_cf:.6f}")
            st.markdown(f"- **Écart absolu** : {abs(price_mc - price_cf):.6g}")
            st.markdown(f"- **Delta (approx)** : {delta:.6f}")
            st.markdown(f"- **Notionnel MC** : {montant_mc:,.2f}")
            st.markdown(f"- **Notionnel semi-analytique** : {price_cf * volume:,.2f}")
            st.markdown(f"- **Prime / Notionnel (MC)** : {price_mc*100:.4f}%")
            st.markdown(f"- **Prime / Notionnel (Semi-analytique)** : {price_cf*100:.4f}%")
            st.markdown(f"- **IC 95% (en % du notionnel)** : ± {1.96*se_mc*100:.4f}%")
        else:
            avg_type = "geometric" if "géométrique" in opt_style else "arithmetic"
            opt = HestonAsianOption(
                spot=spot, strike=strike, maturity=maturity,
                domestic_rate=r_d, foreign_rate=r_f,
                is_call=is_call, n_paths=n_paths,
                v0=v0, kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                n_fixings=12, average_type=avg_type,
            )
            pricer = MonteCarloHestonAsian(opt, n_paths=n_paths, n_steps_per_fix=n_steps_per_fix, seed=42)
            price_mc, se_mc = pricer.price()

            montant_mc = price_mc * volume
            st.subheader("Résultats (Heston Asian)")
            st.markdown(f"- **Maturité** : {adjusted_date:%d-%m-%Y}")
            st.markdown(f"- **Prix MC** : {price_mc:.6f} ± {1.96*se_mc:.6f} (IC 95 %)")
            st.markdown(f"- **Notionnel MC** : {montant_mc:,.2f}")
            st.markdown(f"- **Prime / Notionnel (MC)** : {price_mc*100:.4f}%")
            st.markdown(f"- **IC 95% (en % du notionnel)** : ± {1.96*se_mc*100:.4f}%")

    # ==== PINN Vanille (GK, σ-aware) — seulement pour Vanille ====
    elif model == "PINN (GK, checkpoint)":
        if opt_style != "Vanille":
            st.error("PINN (vanille) n'est disponible que pour le style 'Vanille'.")
        elif pinn_model is None:
            st.error("Charge d’abord le modèle PINN (vanille).")
        elif maturity > T_train + 1e-12:
            st.error(f"Maturité {maturity:.4f} > T_train {T_train:.4f}. "
                     "Réduis la maturité ou entraîne un checkpoint avec T plus grand.")
        elif vol_pinn is None or vol_pinn <= 0:
            st.error("Renseigne une volatilité positive pour le PINN (vanille).")
        else:
            opt = FXOption(
                spot=spot, strike=strike, maturity=maturity,
                domestic_rate=r_d, foreign_rate=r_f,
                vol=vol_pinn, is_call=is_call, n_paths=n_paths,
            )
            engine = PINNGKEnginePriceOnly(
                option=opt, model=pinn_model,
                r_d_train=r_d_train, r_f_train=r_f_train, T_train=T_train
            )
            price_pinn, _ = engine.price()
            montant_pinn = price_pinn * volume

            st.subheader("Résultats (PINN – Vanille)")
            st.markdown(f"- **Maturité** : {adjusted_date:%d-%m-%Y}")
            st.markdown(f"- **Vol PINN (vanille)** : {vol_pinn:.2%}")
            st.markdown(f"- **Prix PINN** : {price_pinn:.6f}")
            st.markdown(f"- **Notionnel PINN** : {montant_pinn:,.2f}")
            st.markdown(f"- **Prime / Notionnel** : {price_pinn*100:.4f}%")

    # ==== PINN Asian (GK, σ-aware) — seulement pour Asian arithmétique ====
    elif model == "PINN Asian (GK, checkpoint)":
        if opt_style != "Asian arithmétique":
            st.error("PINN Asian n'est disponible que pour le style 'Asian arithmétique'.")
        elif not is_call:
            st.error("Le checkpoint Asian est entraîné pour **CALL** uniquement (pas de PUT).")
        elif pinn_asian_model is None:
            st.error("Charge d’abord le modèle PINN Asian.")
        elif vol_pinn_asian is None or vol_pinn_asian <= 0:
            st.error("Renseigne une volatilité positive pour le PINN Asian.")
        else:
            opt = AsianOption(
                spot=spot, strike=strike, maturity=maturity,
                domestic_rate=r_d, foreign_rate=r_f,
                vol=vol_pinn_asian, is_call=is_call, n_paths=n_paths,
                n_fixings=12, average_type="arithmetic",
            )
            engine = PINNGKAsianPriceOnly(
                option=opt, model=pinn_asian_model, r_d_train=r_d_train_asian
            )
            price_pinn_asian, _ = engine.price()
            montant_pinn_asian = price_pinn_asian * volume

            st.subheader("Résultats (PINN – Asian arithmétique)")
            st.markdown(f"- **Maturité** : {adjusted_date:%d-%m-%Y}")
            st.markdown(f"- **Vol PINN (Asian)** : {vol_pinn_asian:.2%}")
            st.markdown(f"- **Prix PINN Asian** : {price_pinn_asian:.6f}")
            st.markdown(f"- **Notionnel PINN Asian** : {montant_pinn_asian:,.2f}")
            st.markdown(f"- **Prime / Notionnel** : {price_pinn_asian*100:.4f}%")

    else:
        st.error("Combinaison modèle/style non supportée.")
