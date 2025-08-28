import numpy as np
from scipy.stats import norm
from typing import Tuple
import math

# 1 |  Vectorised closed‑form Garman–Kohlhagen price


def gk_price(opt_type: str, S0: float, K: float, T: float,
             r_d: float, r_f: float, sigma: float) -> float:
    """Closed‑form GK price for a 'call' or 'put' FX option (settlement in domestic currency)."""
    d1 = (np.log(S0 / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    df_d, df_f = np.exp(-r_d * T), np.exp(-r_f * T)
    if opt_type == "call":
        return S0 * df_f * norm.cdf(d1) - K * df_d * norm.cdf(d2)
    elif opt_type == "put":
        return K * df_d * norm.cdf(-d2) - S0 * df_f * norm.cdf(-d1)
    raise ValueError("opt_type must be 'call' or 'put'")


# 2 |  Plain Monte‑Carlo pricer (European only)


def mc_price(opt_type: str, S0: float, K: float, T: float,
             r_d: float, r_f: float, sigma: float,
             n_sim: int, seed: int = 42) -> Tuple[float, float]:
    """Return (price, standard‑error) for a European FX option."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sim)
    ST = S0 * np.exp((r_d - r_f - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0.0) if opt_type == "call" else np.maximum(K - ST, 0.0)
    disc = np.exp(-r_d * T)
    discounted = disc * payoffs
    price  = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_sim)
    return float(price), float(stderr)


# 3 |  Monte‑Carlo pricer for Asian options (arithmetic & geometric)


def mc_asian_price(opt_type: str,
                   S0: float, K: float, T: float,
                   r_d: float, r_f: float, sigma: float,
                   n_fix: int,
                   n_sim: int,
                   avg_type: str = "arithmetic",
                   seed: int = 42) -> Tuple[float, float]:
    """Return (price, standard‑error) of an Asian FX option under GK.

    Parameters
    ----------
    opt_type   : 'call' or 'put'
    avg_type   : 'arithmetic' or 'geometric'
    n_fix      : number of equally‑spaced fixings (≥ 1)
    """
    if avg_type not in ("arithmetic", "geometric"):
        raise ValueError("avg_type must be 'arithmetic' or 'geometric'")

    rng = np.random.default_rng(seed)
    dt     = T / n_fix
    drift  = (r_d - r_f - 0.5 * sigma**2) * dt
    vol    = sigma * np.sqrt(dt)

    Z = rng.standard_normal((n_sim, n_fix))
    log_paths = np.log(S0) + np.cumsum(drift + vol * Z, axis=1)
    S_paths   = np.exp(log_paths)

    if avg_type == "arithmetic":
        avg = S_paths.mean(axis=1)
    else:
        avg = np.exp(np.mean(np.log(S_paths), axis=1)) # on utilise x^a=exp(log(x) * a)

    payoffs = np.maximum(avg - K, 0.0) if opt_type == "call" else np.maximum(K - avg, 0.0)
    discounted = np.exp(-r_d * T) * payoffs

    price  = discounted.mean()
    stderr = discounted.std(ddof=1) / np.sqrt(n_sim)
    return float(price), float(stderr)


# 4 |  Closed‑form (continuous) geometric Asian option under GK


def geom_asian_price(opt_type: str, S0: float, K: float, T: float,
                      r_d: float, r_f: float, sigma: float) -> float:
    """Continuous‑averaging geometric Asian FX option (GK).

    The pricing formula assumes continuous monitoring of the geometric mean.
    It is exact and much faster than Monte‑Carlo.  For discrete fixings
    with a large number of dates it still provides a very good benchmark.
    """
    # Effective volatility & drift of the geometric average (continuous‑time result)
    sigma_g = sigma / np.sqrt(3.0)
    mu_g    = 0.5 * (r_d - r_f - 0.5 * sigma**2)

    d1 = (np.log(S0 / K) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))
    d2 = d1 - sigma_g * np.sqrt(T)

    df_d, df_f = np.exp(-r_d * T), np.exp(-r_f * T)
    if opt_type == "call":
        return S0 * df_f * norm.cdf(d1) - K * df_d * norm.cdf(d2)
    elif opt_type == "put":
        return K * df_d * norm.cdf(-d2) - S0 * df_f * norm.cdf(-d1)
    raise ValueError("opt_type must be 'call' or 'put'")

################ HESTON ######################

def _heston_cf(u: np.ndarray | float,
               S0: float,
               v0: float,
               kappa: float,
               theta: float,
               sigma: float,
               rho: float,
               r_d: float,
               r_f: float,
               T: float) -> np.ndarray:
    """ϕ(u) = E[exp(iu ln S_T)] sous Heston (domestic numéraire)."""
    u = np.asarray(u, dtype=np.complex128)
    x, a, b = math.log(S0), kappa * theta, kappa
    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - b) ** 2 + sigma**2 * (iu + u**2))
    g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)

    C = (r_d - r_f) * iu * T + (a / sigma**2) * (
        (b - rho * sigma * iu - d) * T
        - 2.0 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = (b - rho * sigma * iu - d) / sigma**2 * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )
    return np.exp(C + D * v0 + iu * x)


# ---- 8·2 |  Semi-analytique (Heston 1993) -----------------------------------


def heston_price_cf(opt_type: str,
                    S0: float, K: float, T: float,
                    r_d: float, r_f: float,
                    v0: float, kappa: float, theta: float,
                    sigma: float, rho: float,
                    integ_lim: float = 100.0,
                    n: int = 20_000) -> float:
    """Prix Heston semi-analytique par intégration numérique (Carr-Madan)."""
    is_call = opt_type == "call"
    lnK = math.log(K)
    u = np.linspace(1e-5, integ_lim, n)

    phi      = _heston_cf(u,      S0, v0, kappa, theta, sigma, rho, r_d, r_f, T)
    phi_u1   = _heston_cf(u - 1j, S0, v0, kappa, theta, sigma, rho, r_d, r_f, T)
    phi_mi   = _heston_cf(-1j,    S0, v0, kappa, theta, sigma, rho, r_d, r_f, T)

    P1 = 0.5 + (1 / math.pi) * np.trapz(
        np.real(np.exp(-1j * u * lnK) * phi_u1 / (1j * u * phi_mi)), u
    )
    P2 = 0.5 + (1 / math.pi) * np.trapz(
        np.real(np.exp(-1j * u * lnK) * phi / (1j * u)), u
    )

    call = S0 * math.exp(-r_f * T) * P1 - K * math.exp(-r_d * T) * P2
    price = call if is_call else (call - S0 * math.exp(-r_f * T) + K * math.exp(-r_d * T))
    return float(price.real)


# ---- 8·3 |  Simulation de trajectoires Heston -------------------------------


def _simulate_heston_paths(S0: float,
                           v0: float,
                           kappa: float,
                           theta: float,
                           sigma: float,
                           rho: float,
                           r_d: float,
                           r_f: float,
                           T: float,
                           n_steps: int,
                           n_sim: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Full-truncation Euler, variance ≥ 0 garant-i."""
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    S = np.full(n_sim, S0)
    v = np.full(n_sim, v0)

    for _ in range(n_steps):
        z1 = rng.standard_normal(n_sim)
        z2 = rng.standard_normal(n_sim)
        dW_S = z1 * sqrt_dt
        dW_v = (rho * z1 + math.sqrt(1 - rho**2) * z2) * sqrt_dt

        v_pos = np.maximum(v, 0.0)
        v += kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * dW_v
        v = np.maximum(v, 0.0)              # full truncation
        S *= np.exp((r_d - r_f - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_S)

    return S


def mc_heston_price(opt_type: str,
                    S0: float, K: float, T: float,
                    r_d: float, r_f: float,
                    v0: float, kappa: float, theta: float,
                    sigma: float, rho: float,
                    n_sim: int, n_steps: int = 252,
                    seed: int = 42) -> Tuple[float, float]:
    """Monte-Carlo Heston (prix, erreur-type)."""
    rng = np.random.default_rng(seed)
    ST = _simulate_heston_paths(S0, v0, kappa, theta, sigma, rho,
                                r_d, r_f, T, n_steps, n_sim, rng)
    payoffs = np.maximum(ST - K, 0.0) if opt_type == "call" else np.maximum(K - ST, 0.0)
    disc = math.exp(-r_d * T)
    discounted = disc * payoffs
    price  = discounted.mean()
    stderr = discounted.std(ddof=1) / math.sqrt(n_sim)
    return float(price), float(stderr)


def mc_heston_asian_price(opt_type: str,
                          S0: float, K: float, T: float,
                          r_d: float, r_f: float,
                          v0: float, kappa: float, theta: float,
                          sigma: float, rho: float,
                          n_fix: int,
                          n_sim: int,
                          avg_type: str = "arithmetic",
                          n_steps_per_fix: int = 1,
                          seed: int = 42) -> Tuple[float, float]:
    """
    Monte-Carlo pricing of an Asian FX option under Heston dynamics.

    Parameters
    ----------
    opt_type        : 'call' or 'put'
    avg_type        : 'arithmetic' or 'geometric'
    n_fix           : # equally-spaced fixings (≥ 1)
    n_steps_per_fix : Euler steps between two consecutive fixings (≥ 1)
                      → mettre 4-8 si T est long ou vol élevé
    """
    if avg_type not in ("arithmetic", "geometric"):
        raise ValueError("avg_type must be 'arithmetic' or 'geometric'")

    rng = np.random.default_rng(seed)
    n_steps = n_fix * n_steps_per_fix
    dt      = T / n_steps
    sqrt_dt = math.sqrt(dt)

    # state vectors
    S = np.full(n_sim, S0)
    v = np.full(n_sim, v0)

    # accumulators for the average
    sum_S     = np.zeros(n_sim)    # arithmetic
    sum_logS  = np.zeros(n_sim)    # geometric (store log to avoid overflow)
    fix_count = 0

    for step in range(n_steps):
        # 1 | correlated Brownian increments
        z1 = rng.standard_normal(n_sim)
        z2 = rng.standard_normal(n_sim)
        dW_S = z1 * sqrt_dt
        dW_v = (rho * z1 + math.sqrt(1.0 - rho**2) * z2) * sqrt_dt

        # 2 | full-truncation Euler for v_t ≥ 0
        v_pos = np.maximum(v, 0.0)
        v += kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos) * dW_v
        v = np.maximum(v, 0.0)

        # 3 | spot update
        S *= np.exp((r_d - r_f - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW_S)

        # 4 | if we just hit a fixing date, accumulate
        if (step + 1) % n_steps_per_fix == 0:
            sum_S    += S
            sum_logS += np.log(S)
            fix_count += 1

    if fix_count != n_fix:                     # should never happen
        raise RuntimeError("Fixing counter mismatch")

    # 5 | compute the average
    if avg_type == "arithmetic":
        avg = sum_S / n_fix
    else:
        avg = np.exp(sum_logS / n_fix)

    # 6 | discounted payoff
    payoff = np.maximum(avg - K, 0.0) if opt_type == "call" else np.maximum(K - avg, 0.0)
    disc   = math.exp(-r_d * T)
    discounted = disc * payoff

    price  = discounted.mean()
    stderr = discounted.std(ddof=1) / math.sqrt(n_sim)
    return float(price), float(stderr)





# 5 |  Data containers


class FXOption:
    """Holds the option data so the rest of your UI doesn’t change."""
    def __init__(self, *, spot: float, strike: float, maturity: float,
                 domestic_rate: float, foreign_rate: float,
                 vol: float | None = None,   # par défault none, 
                 is_call: bool,
                 n_paths: int = 10000) -> None:
        self.spot = spot
        self.strike = strike
        self.maturity = maturity
        self.domestic_rate = domestic_rate
        self.foreign_rate = foreign_rate
        self.vol = vol
        self.is_call = is_call
        self.n_paths = n_paths

class AsianOption(FXOption):
    """Extends FXOption with Asian‑specific metadata."""
    def __init__(self, *, n_fixings: int = 12, average_type: str = "arithmetic", **kwargs) -> None:
        super().__init__(**kwargs)
        if average_type not in ("arithmetic", "geometric"):
            raise ValueError("average_type must be 'arithmetic' or 'geometric'")
        self.n_fixings = n_fixings
        self.average_type = average_type


# 6 |  Monte‑Carlo engines


class MonteCarloGK:
    """Wraps mc_price and adds an (analytic) delta so your UI keeps working."""
    def __init__(self, option: FXOption, n_paths: int = 10000, seed: int = 42) -> None:
        self.option = option
        self.n_paths = n_paths
        self.seed = seed

    # ---- pricing ----
    def price(self) -> Tuple[float, float]:
        typ = "call" if self.option.is_call else "put"
        return mc_price(typ, self.option.spot, self.option.strike,
                        self.option.maturity, self.option.domestic_rate,
                        self.option.foreign_rate, self.option.vol,
                        self.n_paths, self.seed)

    # ---- delta (closed‑form GK, cheaper & less noisy than MC finite diff) ----
    def delta(self) -> Tuple[float, float]:
        S0, K, T = self.option.spot, self.option.strike, self.option.maturity
        r_d, r_f, sigma = self.option.domestic_rate, self.option.foreign_rate, self.option.vol
        d1 = (np.log(S0 / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        df_f = np.exp(-r_f * T)
        delta = df_f * norm.cdf(d1) if self.option.is_call else df_f * (norm.cdf(d1) - 1)
        return float(delta), 0.0



######## HESTON:

class HestonOption(FXOption):
    """Ajoute les 5 paramètres vol-stoch à FXOption."""
    def __init__(self,
                 *,
                 v0: float, kappa: float, theta: float,
                 sigma: float, rho: float,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho


class MonteCarloHeston:
    """Pricer MC + delta GK de référence (optionnel)."""
    def __init__(self,
                 option: HestonOption,
                 n_paths: int | None = None,
                 n_steps: int = 252,
                 seed: int = 42) -> None:
        self.opt = option
        self.n_paths = n_paths or option.n_paths
        self.n_steps = n_steps
        self.seed = seed

    # ---- pricing ----
    def price(self) -> Tuple[float, float]:
        typ = "call" if self.opt.is_call else "put"
        return mc_heston_price(typ,
                               self.opt.spot, self.opt.strike, self.opt.maturity,
                               self.opt.domestic_rate, self.opt.foreign_rate,
                               self.opt.v0, self.opt.kappa, self.opt.theta,
                               self.opt.sigma, self.opt.rho,
                               self.n_paths, self.n_steps, self.seed)

    # ---- delta (approx : sensi num via bump-&-reprice GK) ----
    def delta(self, bump: float = 1e-4) -> Tuple[float, float]:
        """Delta ≈ (P(S0+ε)-P(S0-ε))/2ε — bruité mais rapide à coder."""
        S0 = self.opt.spot
        self.opt.spot = S0 + bump
        up, _  = self.price()
        self.opt.spot = S0 - bump
        dn, _  = self.price()
        self.opt.spot = S0          # restore
        delta = (up - dn) / (2 * bump)
        # très bruité : pas de stderr sérieux ici
        return float(delta), float("nan")

class HestonAsianOption(HestonOption):
    """Heston + metadata for Asian averaging."""
    def __init__(self,
                 *,
                 n_fixings: int = 12,
                 average_type: str = "arithmetic",
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if average_type not in ("arithmetic", "geometric"):
            raise ValueError("average_type must be 'arithmetic' or 'geometric'")
        self.n_fixings   = n_fixings
        self.average_type = average_type



######## engine

class MonteCarloHestonAsian:
    """Wrapper object to keep the same OO feel as the rest of your code."""
    def __init__(self,
                 option: HestonAsianOption,
                 n_paths: int | None = None,
                 n_steps_per_fix: int = 1,
                 seed: int = 42) -> None:
        self.opt = option
        self.n_paths = n_paths or option.n_paths
        self.n_steps_per_fix = n_steps_per_fix
        self.seed = seed

    def price(self) -> Tuple[float, float]:
        typ = "call" if self.opt.is_call else "put"
        return mc_heston_asian_price(
            typ,
            self.opt.spot, self.opt.strike, self.opt.maturity,
            self.opt.domestic_rate, self.opt.foreign_rate,
            self.opt.v0, self.opt.kappa, self.opt.theta,
            self.opt.sigma, self.opt.rho,
            self.opt.n_fixings,
            self.n_paths,
            self.opt.average_type,
            self.n_steps_per_fix,
            self.seed,
        )




class MonteCarloAsian:
    """Monte‑Carlo pricer for Asian FX options (arithmetic or geometric)."""
    def __init__(self, option: AsianOption, n_paths: int | None = None, seed: int = 42) -> None:
        self.option = option
        self.n_paths = n_paths or option.n_paths
        self.seed = seed

    def price(self) -> Tuple[float, float]:
        typ = "call" if self.option.is_call else "put"
        return mc_asian_price(typ,
                              self.option.spot, self.option.strike, self.option.maturity,
                              self.option.domestic_rate, self.option.foreign_rate, self.option.vol,
                              self.option.n_fixings,
                              self.n_paths,
                              self.option.average_type,
                              self.seed)


# 7 |  Convenience aliases

def garman_kohlhagen_price(option: FXOption) -> float:
    typ = "call" if option.is_call else "put"
    return gk_price(typ, option.spot, option.strike,
                    option.maturity, option.domestic_rate,
                    option.foreign_rate, option.vol)


def heston_price(option: HestonOption,
                 method: str = "cf",
                 **mc_kwargs) -> float | Tuple[float, float]:
    """Retourne le prix Heston (closed-form ou MC)."""
    typ = "call" if option.is_call else "put"
    if method.lower() == "cf":
        return heston_price_cf(typ,
                               option.spot, option.strike, option.maturity,
                               option.domestic_rate, option.foreign_rate,
                               option.v0, option.kappa, option.theta,
                               option.sigma, option.rho,
                               **mc_kwargs)   # ici: integ_lim / n éventuellement
    elif method.lower() == "mc":
        return mc_heston_price(typ,
                               option.spot, option.strike, option.maturity,
                               option.domestic_rate, option.foreign_rate,
                               option.v0, option.kappa, option.theta,
                               option.sigma, option.rho,
                               mc_kwargs.get("n_sim", option.n_paths),
                               mc_kwargs.get("n_steps", 252),
                               mc_kwargs.get("seed", 42))
    raise ValueError("method must be 'cf' or 'mc'")


def heston_asian_price(option: HestonAsianOption,
                       **mc_kwargs) -> Tuple[float, float]:
    """
    Returns (price, stderr) — Monte-Carlo only (no closed-form under Heston).
    mc_kwargs → n_sim, n_steps_per_fix, seed
    """
    typ = "call" if option.is_call else "put"
    return mc_heston_asian_price(
        typ,
        option.spot, option.strike, option.maturity,
        option.domestic_rate, option.foreign_rate,
        option.v0, option.kappa, option.theta,
        option.sigma, option.rho,
        option.n_fixings,
        mc_kwargs.get("n_sim", option.n_paths),
        option.average_type,
        mc_kwargs.get("n_steps_per_fix", 1),
        mc_kwargs.get("seed", 42),
    )



# ===== PINN (GK, σ-aware) – load + price only ================================
import torch
import torch.nn as nn
import numpy as np

class WPINNVol(nn.Module):
    """
    w(S,τ,K,σ) = ReLU(S-K) + τ * softplus( NN([S,τ,K,σ,log(S/K)]) )
    """
    def __init__(self, K_ref, T, sigma_ref=0.14, hidden=128, layers=6):
        super().__init__()
        self.K_ref = float(K_ref)
        self.T = float(T)
        self.sigma_ref = float(sigma_ref)
        h = hidden
        net = [nn.Linear(5, h), nn.Tanh()]
        for _ in range(layers - 1):
            net += [nn.Linear(h, h), nn.Tanh()]
        net += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*net)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x):
        # x = [S, τ, K, σ]
        S, tau, K, sig = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        Sn   = (S - self.K_ref) / (self.K_ref + 1e-8)
        tn   = tau / (self.T + 1e-12)
        Kn   = (K - self.K_ref) / (self.K_ref + 1e-8)
        sign = sig / (self.sigma_ref + 1e-12)
        mn   = torch.log(torch.clamp(S, 1e-12) / torch.clamp(K, 1e-12))
        n = self.net(torch.cat([Sn, tn, Kn, sign, mn], dim=1))
        payoff = torch.relu(S - K)
        return payoff + tau * self.softplus(n)

def load_pinn_model(path: str,
                    K_ref: float,
                    T_train: float,
                    sigma_ref: float = 0.14,
                    hidden: int = 128,
                    layers: int = 6,
                    device: str | None = None) -> WPINNVol:
    """
    Load the trained PINN weights. Must match the architecture used in training.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = WPINNVol(K_ref=K_ref, T=T_train, sigma_ref=sigma_ref,
                     hidden=hidden, layers=layers).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def pinn_price_call(model: WPINNVol,
                    r_d_train: float,
                    T_train: float,
                    S: float, K: float, tau: float, sigma: float) -> float:
    """
    Call price from PINN: u = e^{-r_d τ} · w(S,τ,K,σ).
    NOTE: Use the same r_d and T that were used to train the checkpoint.
    """
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor([[S, tau, K, sigma]], dtype=torch.float32, device=dev)
        w = model(x).item()
    return float(np.exp(-r_d_train * tau) * w)



class PINNGKEnginePriceOnly:
    """
    Price-only engine to fit your UI like MonteCarloGK.
    Uses call PINN + parity for puts.
    """
    def __init__(self,
                 option: FXOption,
                 model: WPINNVol,
                 r_d_train: float,
                 r_f_train: float,
                 T_train: float):
        self.opt = option
        self.model = model
        self.r_d_train = float(r_d_train)
        self.r_f_train = float(r_f_train)
        self.T_train   = float(T_train)

    def price(self) -> tuple[float, float]:
        tau = float(self.opt.maturity)
        if tau > self.T_train + 1e-12 or tau < -1e-12:
            raise ValueError(f"PINN trained with T={self.T_train}. Got maturity={tau}.")
        if self.opt.is_call:
            u = pinn_price_call(self.model, self.r_d_train, self.T_train,
                                self.opt.spot, self.opt.strike, tau, self.opt.vol)
            return float(u), 0.0
        # put via parity (GK)
        call = pinn_price_call(self.model, self.r_d_train, self.T_train,
                               self.opt.spot, self.opt.strike, tau, self.opt.vol)
        disc_d = np.exp(-self.opt.domestic_rate * tau)
        disc_f = np.exp(-self.opt.foreign_rate * tau)
        put = call - self.opt.spot * disc_f + self.opt.strike * disc_d
        return float(put), 0.0



class AsianPINN(nn.Module):
    """
    w(S,I,τ,K,T,σ) = max(I/T − K, 0) + τ * softplus( NN([S, A=I/T, τ, K, T, σ, m=log(S/K), it=A-K]) )
    Retourne w en espace actualisé; le prix est V = e^{−r_d τ} · w.
    """
    def __init__(self, hidden: int = 128, depth: int = 6):
        super().__init__()
        h = hidden
        layers = [nn.Linear(8, h), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=1.0)
        # init Xavier + biais à 0 (comme à l'entraînement)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S, I, tau, K, T, sigma):
        A  = I / torch.clamp(T, min=1e-8)                  # moyenne courante / T
        m  = torch.log(torch.clamp(S, 1e-12) / torch.clamp(K, 1e-12))  # log-moneyness
        it = (A - K)                                       # (A − K)
        x  = torch.cat([S, A, tau, K, T, sigma, m, it], dim=1)
        core   = self.net(x)
        payoff = torch.clamp(A - K, min=0.0)
        return payoff + tau * self.softplus(core)


def load_asian_pinn_model(path: str,
                          hidden: int = 128,
                          depth: int = 6,
                          device: str | None = None) -> AsianPINN:
    """
    Charge le checkpoint PINN Asian (état dict ou dict{'state_dict': ...}).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AsianPINN(hidden=hidden, depth=depth).to(device)
    ckpt  = torch.load(path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def pinn_asian_fixed_call_price(model: AsianPINN,
                                r_d_train: float,
                                S0: float,
                                K: float,
                                T: float,
                                sigma: float) -> float:
    """
    Prix Asian **arithmétique CALL** à t=0 avec I(0)=0 et τ=T :
      w = model(S=S0, I=0, τ=T, K, T, σ)  →  V = e^{−r_d T} · w
    """
    dev = next(model.parameters()).device
    dtp = next(model.parameters()).dtype   # ← aligne le dtype sur celui du modèle

    with torch.no_grad():
        S   = torch.tensor([[S0]],   dtype=dtp, device=dev)
        I   = torch.tensor([[0.0]],  dtype=dtp, device=dev)
        tau = torch.tensor([[T]],    dtype=dtp, device=dev)
        K_  = torch.tensor([[K]],    dtype=dtp, device=dev)
        T_  = torch.tensor([[T]],    dtype=dtp, device=dev)
        sig = torch.tensor([[sigma]],dtype=dtp, device=dev)

        w = model(S, I, tau, K_, T_, sig).item()
    return float(np.exp(-r_d_train * T) * w)






class PINNGKAsianPriceOnly:
    """
    Moteur 'price-only' pour Asian arithmétique **CALL** via PINN (GK, σ-aware).
    Interface homogène: .price() -> (prix, 0.0)
    """
    def __init__(self,
                 option: AsianOption,
                 model: AsianPINN,
                 r_d_train: float):
        self.opt = option
        self.model = model
        self.r_d_train = float(r_d_train)

        # Garde-fous conforme à l'entraînement
        if self.opt.average_type != "arithmetic":
            raise ValueError("Asian PINN supporte uniquement l'average **arithmétique**.")
        if not self.opt.is_call:
            raise ValueError("Asian PINN entraîné pour **CALL** uniquement (pas de PUT).")

    def price(self) -> tuple[float, float]:
        v = pinn_asian_fixed_call_price(self.model,
                                        r_d_train=self.r_d_train,
                                        S0=self.opt.spot,
                                        K=self.opt.strike,
                                        T=float(self.opt.maturity),
                                        sigma=self.opt.vol)
        return float(v), 0.0