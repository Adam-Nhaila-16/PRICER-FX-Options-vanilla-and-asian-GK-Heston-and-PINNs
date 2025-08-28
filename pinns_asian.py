"""
Discounted, σ‑aware PINN for FX **Asian (fixed‑strike) calls** under
Garman–Kohlhagen with **curriculum learning**, **MC anchor grid**,
**LR scheduling**, and optional **LBFGS refine**. Notebook‑friendly CLI.

Why this version?
- Your gap grew after the last patch. That hinted at sampling/BC pressure.
- This build adds:
  1) **Physically consistent I bounds**: I ∈ [0, S_MAX·(T−τ)].
  2) **Curriculum** (first 60% epochs focus ATM & mid maturities, then widen).
  3) **MC anchor grid** (several K,T pairs per epoch) with stronger weight.
  4) **ReduceLROnPlateau** + optional **LBFGS refine** (post‑Adam polish).
  5) Larger PDE batch & more epochs by default.

Quick start in a notebook
-------------------------
from Pinn_Fx_Asian_Garman_Kohlhagen import run_cli
run_cli(["--train"])                                  # train & save weights
run_cli(["--quote","--K","1.10","--T","0.75"])        # instant quote + MC

"""
from __future__ import annotations
import math, time, argparse, warnings
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # Market
    r_d: float = 0.04124
    r_f: float = 0.01847
    sigma: float | None = None            # if None, estimate from 60d realized

    # Spot/vol data
    ticker: str = "EURUSD=X"
    hist_period_days: int = 60
    use_hist_vol: bool = True

    # Parameterized quoting/training ranges
    T_min: float = 1/12                   # 1m
    T_max: float = 2.0                    # 2y
    K_mult_low: float = 0.6               # global K ∈ [0.6 S0, 1.4 S0]
    K_mult_high: float = 1.4

    # Curriculum (phase 1 focus near ATM & mid maturities)
    curriculum_frac: float = 0.6
    focus_K_band: float = 0.12            # ±12% around ATM in phase 1
    focus_T_min: float = 0.25             # 3m
    focus_T_max: float = 1.0              # 1y

    # Domain and losses
    S_max_mult: float = 2.5
    N_pde: int = 20_000                   # larger PDE batch
    N_bc: int = 1_200
    N_tc: int = 1_000
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_tc: float = 0.05                    # small; payoff hard‑wired
    w_mono: float = 0.05                  # soft monotonicity (∂w/∂S, ∂w/∂I ≥ 0)
    w_anchor: float = 1.2                 # stronger anchors (price grid)

    # MC anchor grid per epoch (t=0)
    anchor_K_mults: tuple = (0.9, 1.0, 1.1, 1.2)
    anchor_T_vals: tuple = (0.25, 0.5, 1.0)
    anchor_paths: int = 6_000
    anchor_steps: int = 126

    # Network / training
    hidden: int = 128
    depth: int = 6
    epochs: int = 3000
    lr: float = 1e-3
    weight_decay: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # LR schedule
    plateau_factor: float = 0.5
    plateau_patience: int = 200

    # Monte Carlo for reporting
    mc_paths: int = 150_000
    mc_steps: int = 252

    # Optional LBFGS refine at end
    refine_lbfgs: bool = True
    refine_pde: int = 8_000
    refine_bc: int = 800
    refine_max_iter: int = 200

    # I/O
    weights_path: str = "fx_asian_pinn.pt"

CFG = Config()

# --------------------------------------------------------------------------- #
# Market data                                                                 #
# --------------------------------------------------------------------------- #
S0_FALLBACK = 1.10
SIGMA_FALLBACK = 0.12

def get_spot_and_sigma() -> Tuple[float, float]:
    S0 = S0_FALLBACK
    sig = SIGMA_FALLBACK if CFG.sigma is None else CFG.sigma
    try:
        import yfinance as yf
        df = yf.download(CFG.ticker, period=f"{CFG.hist_period_days}d", progress=False)
        if df is not None and len(df) > 0:
            px = df.get("Adj Close", df["Close"]).astype(float)
            S0 = float(px.iloc[-1])
            if CFG.sigma is None and CFG.use_hist_vol and len(px) > 3:
                lr = np.log(px/px.shift(1)).dropna()
                sig = float(lr.std() * math.sqrt(252))
    except Exception as e:
        print(f"[warn] yfinance unavailable, fallbacks used: {e}")
    return S0, sig

S0, SIGMA = get_spot_and_sigma()
S_MAX = CFG.S_max_mult * S0
EPS_TIME = 1e-6  # for stable handling of very small (T - tau)

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

def U(n, lo, hi):
    return lo + (hi-lo) * torch.rand(n,1, device=CFG.device, dtype=torch.float64)

def fx_int_coeff(rd_minus_rf: float, tau: torch.Tensor) -> torch.Tensor:
    """(e^{(r_d−r_f)τ} − 1)/(r_d−r_f), with r_d≈r_f limit -> τ."""
    if abs(rd_minus_rf) < 1e-12:
        return tau
    return (torch.exp(rd_minus_rf * tau) - 1.0) / rd_minus_rf

# --------------------------------------------------------------------------- #
# Model: discounted w with hard terminal payoff                               #
# --------------------------------------------------------------------------- #
class AsianPINN(nn.Module):
    """w(S,I,τ,K,T,σ) = max(I/T − K, 0) + τ * softplus( NN(features) )."""
    def __init__(self, hidden=128, depth=6):
        super().__init__()
        h = hidden
        layers = [nn.Linear(8, h), nn.Tanh()]
        for _ in range(depth-2):
            layers += [nn.Linear(h, h), nn.Tanh()]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus(beta=1.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S, I, tau, K, T, sigma):
        A = I / torch.clamp(T, min=1e-8)                             # running avg per T
        m = torch.log(torch.clamp(S,1e-12)/torch.clamp(K,1e-12))     # log‑moneyness
        it = (A - K)                                                 # ITMness of avg
        x = torch.cat([S, A, tau, K, T, sigma, m, it], dim=1)
        core = self.net(x)
        payoff = torch.clamp(A - K, min=0.0)
        return payoff + tau * self.softplus(core)

model = AsianPINN(CFG.hidden, CFG.depth).to(CFG.device)
opt = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=CFG.plateau_factor,
                                             patience=CFG.plateau_patience, verbose=False)

# --------------------------------------------------------------------------- #
# Sampling (with curriculum)                                                  #
# --------------------------------------------------------------------------- #

def sample_KT(n: int, focus: bool):
    if focus:
        T = U(n, CFG.focus_T_min, CFG.focus_T_max)
        K = U(n, (1-CFG.focus_K_band)*S0, (1+CFG.focus_K_band)*S0)
    else:
        T = U(n, CFG.T_min, CFG.T_max)
        K = U(n, CFG.K_mult_low*S0, CFG.K_mult_high*S0)
    return K, T

def sample_pde(n: int, focus: bool):
    K, T = sample_KT(n, focus)
    tau = U(n, 0.0, 1.0) * T
    S = U(n, 0.0, S_MAX)
    elapsed = torch.clamp(T - tau, min=EPS_TIME)
    I = torch.rand(n,1, device=CFG.device, dtype=torch.float64) * (S_MAX * elapsed)
    sigma = U(n, 0.08, 0.20)
    return S.requires_grad_(True), I.requires_grad_(True), tau.requires_grad_(True), K, T, sigma

def sample_bc_S0(n: int, focus: bool):
    K, T = sample_KT(n, focus)
    tau = U(n, 0.0, 1.0) * T
    S = torch.zeros_like(T)
    elapsed = torch.clamp(T - tau, min=EPS_TIME)
    I = torch.rand(n,1, device=CFG.device, dtype=torch.float64) * (S_MAX * elapsed)
    sigma = U(n, 0.08, 0.20)
    target = torch.clamp(I/T - K, min=0.0)
    return S, I, tau, K, T, sigma, target

def sample_bc_Smax(n: int, focus: bool):
    K, T = sample_KT(n, focus)
    tau = U(n, 0.0, 1.0) * T
    S = torch.full_like(T, S_MAX)
    elapsed = torch.clamp(T - tau, min=EPS_TIME)
    I = torch.rand(n,1, device=CFG.device, dtype=torch.float64) * (S_MAX * elapsed)
    sigma = U(n, 0.08, 0.20)
    coeff = fx_int_coeff(CFG.r_d - CFG.r_f, tau) / T
    target = torch.clamp(I/T - K + S_MAX*coeff, min=0.0)
    return S, I, tau, K, T, sigma, target

def sample_bc_Imax(n: int, focus: bool):
    K, T = sample_KT(n, focus)
    tau = U(n, 0.0, 1.0) * T
    S = U(n, 0.0, S_MAX)
    elapsed = torch.clamp(T - tau, min=EPS_TIME)
    I = S_MAX * elapsed
    sigma = U(n, 0.08, 0.20)
    coeff = fx_int_coeff(CFG.r_d - CFG.r_f, tau) / T
    target = (I/T - K) + S*coeff
    return S, I, tau, K, T, sigma, target

# --------------------------------------------------------------------------- #
# PDE residual in discounted space                                            #
# --------------------------------------------------------------------------- #

def pde_residual(S, I, tau, K, T, sigma):
    w = model(S, I, tau, K, T, sigma)
    ones = torch.ones_like(w)
    w_S, w_I, w_tau = torch.autograd.grad(w, (S, I, tau), grad_outputs=ones, create_graph=True)
    w_SS = torch.autograd.grad(w_S, S, grad_outputs=torch.ones_like(w_S), create_graph=True)[0]
    resid = w_tau - (0.5 * sigma**2 * S**2 * w_SS + (CFG.r_d - CFG.r_f) * S * w_S + S * w_I)
    # Normalize residual to reduce scale bias across domain
    return resid / (1.0 + torch.abs(w) + S**2)

# --------------------------------------------------------------------------- #
# Monte Carlo (for comparison / anchors)                                      #
# --------------------------------------------------------------------------- #

def mc_price_asian_fixed_call(S0: float, K: float, T: float, r_d: float, r_f: float, sigma: float,
                              paths: int, steps: int) -> Tuple[float, float]:
    dt = T / steps
    nudt = (r_d - r_f - 0.5 * sigma * sigma) * dt
    sdt = sigma * math.sqrt(dt)
    S = np.full(paths, S0, dtype=np.float64)
    I = np.zeros(paths, dtype=np.float64)
    for _ in range(steps):
        Z = np.random.normal(size=paths)
        S *= np.exp(nudt + sdt * Z)
        I += S * dt
    A = I / T
    payoff = np.maximum(A - K, 0.0)
    disc = math.exp(-r_d * T)
    price = disc * payoff.mean()
    se = disc * payoff.std(ddof=1) / math.sqrt(paths)
    return price, se

# --------------------------------------------------------------------------- #
# Anchors                                                                     #
# --------------------------------------------------------------------------- #

def mc_anchor_grid(sig: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute small grid of MC anchors at t=0 and map to discounted w targets.
       Returns (X_list, w_targets) where X_list packs (S,I,tau,K,T,sigma) tensors.
    """
    Ks = [m*S0 for m in CFG.anchor_K_mults]
    Ts = list(CFG.anchor_T_vals)
    Xs: List[torch.Tensor] = []
    ws: List[float] = []
    for K in Ks:
        for T in Ts:
            v_mc, _ = mc_price_asian_fixed_call(S0, K, T, CFG.r_d, CFG.r_f, sig,
                                                paths=CFG.anchor_paths, steps=CFG.anchor_steps)
            w_tgt = math.exp(CFG.r_d * T) * v_mc  # w = e^{r_d T} V at t=0
            Xs.append(torch.tensor([[S0, 0.0, T, K, T, sig]], dtype=torch.float64, device=CFG.device))
            ws.append(w_tgt)
    X = torch.cat(Xs, dim=0)
    w_targets = torch.tensor(ws, dtype=torch.float64, device=CFG.device).view(-1,1)
    # Unpack X into model input shapes in forward pass order: S,I,tau,K,T,sigma
    S = X[:,0:1]; I = X[:,1:2]; tau = X[:,2:3]; K = X[:,3:4]; T = X[:,4:5]; sigma = X[:,5:6]
    return (S,I,tau,K,T,sigma), w_targets

# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #

def train():
    mse = nn.MSELoss()
    for ep in range(1, CFG.epochs+1):
        focus = (ep <= int(CFG.curriculum_frac * CFG.epochs))
        opt.zero_grad()

        # PDE interior
        S,I,tau,K,T,sig = sample_pde(CFG.N_pde, focus)
        L_pde = torch.mean(pde_residual(S,I,tau,K,T,sig)**2)

        # Boundaries
        Sb0, Ib0, tb0, Kb0, Tb0, sigb0, yb0 = sample_bc_S0(CFG.N_bc, focus)
        L_b0 = mse(model(Sb0, Ib0, tb0, Kb0, Tb0, sigb0), yb0)

        Sbm, Ibm, tbm, Kbm, Tbm, sigbm, ybm = sample_bc_Smax(CFG.N_bc, focus)
        L_bm = mse(model(Sbm, Ibm, tbm, Kbm, Tbm, sigbm), ybm)

        Sbi, Ibi, tbi, Kbi, Tbi, sigbi, ybi = sample_bc_Imax(CFG.N_bc, focus)
        L_bi = mse(model(Sbi, Ibi, tbi, Kbi, Tbi, sigbi), ybi)
        L_bc = L_b0 + L_bm + L_bi

        # Terminal (τ=0) check (should be near zero once learned)
        Kt, Tt = sample_KT(CFG.N_tc, focus)
        St = U(CFG.N_tc, 0.0, S_MAX)
        It = torch.rand(CFG.N_tc,1, device=CFG.device, dtype=torch.float64) * (S_MAX*torch.ones_like(Tt)) * 0.1
        taut = torch.zeros_like(Tt)
        sigt = U(CFG.N_tc, 0.08, 0.20)
        y_term = torch.clamp(It/Tt - Kt, min=0.0)
        L_tc = mse(model(St, It, taut, Kt, Tt, sigt), y_term)

        # Soft monotonicity: ∂w/∂S ≥0, ∂w/∂I ≥0
        w = model(S,I,tau,K,T,sig)
        dS, dI, _ = torch.autograd.grad(w, (S,I,tau), grad_outputs=torch.ones_like(w), create_graph=True)
        L_mono = torch.mean(torch.relu(-dS)) + torch.mean(torch.relu(-dI))

        # MC anchor grid (t=0) for several (K,T)
        (Sa,Ia,ta,Ka,Ta,siga), w_tgts = mc_anchor_grid(float(SIGMA))
        w_hat = model(Sa, Ia, ta, Ka, Ta, siga)
        L_anchor = mse(w_hat, w_tgts)

        # Total loss
        loss = (CFG.w_pde * L_pde + CFG.w_bc * L_bc + CFG.w_tc * L_tc
                + CFG.w_mono * L_mono + CFG.w_anchor * L_anchor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss)

        if ep % 100 == 0 or ep == 1:
            print(f"[ep {ep:4d}] total={loss.item():.6e}  pde={L_pde.item():.6e}  "
                  f"bc={L_bc.item():.6e}  tc={L_tc.item():.6e}  mono={L_mono.item():.6e}  anch={L_anchor.item():.6e}")

    # Optional LBFGS refine on a fixed mini‑domain for extra PDE tightness
    if CFG.refine_lbfgs:
        refine_lbfgs()

    torch.save({'state_dict': model.state_dict()}, CFG.weights_path)
    print(f"Saved weights -> {CFG.weights_path}")

# --------------------------------------------------------------------------- #
# LBFGS refine                                                                #
# --------------------------------------------------------------------------- #

def refine_lbfgs():
    print("Starting LBFGS refine…")
    mse = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8, max_iter=CFG.refine_max_iter,
                             tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn='strong_wolfe')
    focus = True
    # Freeze a modest batch for deterministic refine
    S,I,tau,K,T,sig = sample_pde(CFG.refine_pde, focus)
    Sb0, Ib0, tb0, Kb0, Tb0, sigb0, yb0 = sample_bc_S0(CFG.refine_bc, focus)
    Sbm, Ibm, tbm, Kbm, Tbm, sigbm, ybm = sample_bc_Smax(CFG.refine_bc, focus)
    Sbi, Ibi, tbi, Kbi, Tbi, sigbi, ybi = sample_bc_Imax(CFG.refine_bc, focus)
    (Sa,Ia,ta,Ka,Ta,siga), w_tgts = mc_anchor_grid(float(SIGMA))

    def closure():
        optimizer.zero_grad()
        L_pde = torch.mean(pde_residual(S,I,tau,K,T,sig)**2)
        L_b0 = mse(model(Sb0, Ib0, tb0, Kb0, Tb0, sigb0), yb0)
        L_bm = mse(model(Sbm, Ibm, tbm, Kbm, Tbm, sigbm), ybm)
        L_bi = mse(model(Sbi, Ibi, tbi, Kbi, Tbi, sigbi), ybi)
        L_bc = L_b0 + L_bm + L_bi
        L_tc = torch.zeros((), dtype=torch.float64, device=CFG.device)
        w = model(S,I,tau,K,T,sig)
        dS, dI, _ = torch.autograd.grad(w, (S,I,tau), grad_outputs=torch.ones_like(w), create_graph=True)
        L_mono = torch.mean(torch.relu(-dS)) + torch.mean(torch.relu(-dI))
        w_hat = model(Sa, Ia, ta, Ka, Ta, siga)
        L_anchor = mse(w_hat, w_tgts)
        loss = (CFG.w_pde * L_pde + CFG.w_bc * L_bc + CFG.w_tc * L_tc
                + CFG.w_mono * L_mono + CFG.w_anchor * L_anchor)
        loss.backward()
        return loss

    loss0 = optimizer.step(closure)
    print(f"LBFGS done. Final refine loss={float(loss0):.6e}")

# --------------------------------------------------------------------------- #
# Quoting & comparison                                                         #
# --------------------------------------------------------------------------- #

def _ensure_loaded(path: str | None = None):
    path = path or CFG.weights_path
    try:
        ckpt = torch.load(path, map_location=CFG.device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return True
    except Exception as e:
        print(f"[warn] could not load weights: {e}")
        return False


def quote(K: float, T: float, sigma: float | None = None, S_init: float | None = None) -> float:
    """Instant price at t=0, I(0)=0, S=S0 by default."""
    if not _ensure_loaded():
        model.eval()
    S = S_init if S_init is not None else S0
    sig = sigma if sigma is not None else SIGMA
    with torch.no_grad():
        w = model(torch.tensor([[S]],dtype=torch.float64,device=CFG.device),
                  torch.tensor([[0.0]],dtype=torch.float64,device=CFG.device),
                  torch.tensor([[T]],dtype=torch.float64,device=CFG.device),
                  torch.tensor([[K]],dtype=torch.float64,device=CFG.device),
                  torch.tensor([[T]],dtype=torch.float64,device=CFG.device),
                  torch.tensor([[sig]],dtype=torch.float64,device=CFG.device)).item()
        V = math.exp(-CFG.r_d * T) * w
        return float(V)


def compare_with_mc(K: float, T: float, sigma: float | None = None):
    sig = sigma if sigma is not None else SIGMA
    v_pinn = quote(K, T, sigma=sig)
    v_mc, se = mc_price_asian_fixed_call(S0, K, T, CFG.r_d, CFG.r_f, sig, CFG.mc_paths, CFG.mc_steps)
    diff = v_pinn - v_mc
    rel = 100.0 * (diff / v_mc) if v_mc > 1e-12 else float('nan')
    print(f"PINN={v_pinn:.6f} | MC={v_mc:.6f} ± {1.96*se:.6f} | abs diff={diff:+.6f} ({rel:.2f}% )  [ASIAN]")
    return v_pinn, v_mc, se

# --------------------------------------------------------------------------- #
# CLI helpers (Notebook‑friendly)                                             #
# --------------------------------------------------------------------------- #

def run_cli(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Discounted PINN for FX Asian calls (GK)")
    parser.add_argument("--train", action="store_true", help="Train the PINN and save weights.")
    parser.add_argument("--quote", action="store_true", help="Quote instantly for given K and T (loads weights).")
    parser.add_argument("--K", type=float, default=None, help="Strike for quoting.")
    parser.add_argument("--T", type=float, default=None, help="Maturity (years) for quoting.")
    parser.add_argument("--sigma", type=float, default=None, help="Override volatility (otherwise 60d realized).")

    args, _unknown = parser.parse_known_args(argv)

    global SIGMA
    if args.sigma is not None:
        SIGMA = args.sigma

    if args.train:
        print(f"Training on {CFG.device} with S0={S0:.6f}, σ={SIGMA:.4%}, r_d={CFG.r_d}, r_f={CFG.r_f}")
        t0 = time.time()
        train()
        print(f"Training done in {time.time()-t0:.1f}s")

    if args.quote:
        if args.K is None or args.T is None:
            raise SystemExit("--quote requires --K and --T")
        v = quote(args.K, args.T, sigma=args.sigma)
        print(f"PINN quote: {v:.6f}")
        compare_with_mc(args.K, args.T, sigma=args.sigma)

if __name__ == "__main__":
    run_cli()


# Save current model weights to a .pt file
torch.save({'state_dict': model.state_dict()}, "fx_asian_pinn.pt")
