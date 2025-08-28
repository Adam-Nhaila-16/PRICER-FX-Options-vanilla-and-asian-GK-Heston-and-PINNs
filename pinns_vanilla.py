# ======================= σ-aware GK PINN (annotated) =======================
"""
A fully-commented PyTorch implementation of a Physics-Informed Neural Network (PINN)
for FX European call options under the Garman–Kohlhagen (GK) model.

Core idea
--
We learn the discounted option value `w(S, τ, K, σ)` that satisfies the GK PDE in
**discounted domestic numeraire**. We parameterize `w` as:

    w(S, τ, K, σ) = ReLU(S - K) + τ * softplus( NN([S, τ, K, σ, log(S/K)]) )

This guarantees:
- Correct terminal payoff at τ=0 (since the additive term vanishes).
- Nonnegative time value via `softplus`.
- Awareness of volatility `σ` as an input, so the network can learn a σ-conditional map.

We train with a weighted loss composed of:
- PDE residual loss in (S, τ, K, σ) space.
- Boundary conditions at S=0 and S=S_max (in discounted form).
- Terminal condition at τ=0.
- A soft lower-bound penalty consistent with no-arbitrage in discounted space.
- "Anchors": supervised targets from the GK closed-form price and its delta to help
  the network converge quickly and stably.

Finally, predicted undiscounted price is recovered as `u = e^{-r_d τ} * w`.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time, warnings
warnings.filterwarnings("ignore")

# GK closed form (internal) for anchor targets 

def _gk_call_torch(S, t, K, r_d, r_f, sigma, T, device="cpu"):
    """Compute the Garman–Kohlhagen European call price (undiscounted u) in Torch.

    Parameters
    ---
    S : Tensor-like
        Spot FX rate (domestic per unit of foreign), S > 0.
    t : Tensor-like
        Current time (in years). Used together with T to form τ = T - t.
    K : Tensor-like
        Strike.
    r_d, r_f : float
        Domestic and foreign continuously compounded risk-free rates.
    sigma : Tensor-like
        Volatility.
    T : float
        Maturity time.
    device : str
        Torch device.

    Returns
    
    price : torch.Tensor
        Undiscounted call price u(t,S).
    """
    # Robust casting/clamping to avoid inf/nan in logs/divisions
    S = torch.as_tensor(S, dtype=torch.float32, device=device)
    t = torch.as_tensor(t, dtype=torch.float32, device=device)
    K = torch.as_tensor(K, dtype=torch.float32, device=device)
    tau = torch.clamp(torch.tensor(T, dtype=torch.float32, device=device) - t, min=1e-12)
    Sd  = torch.clamp(S, min=1e-12)
    sig = torch.as_tensor(sigma, dtype=torch.float32, device=device)

    vol = sig * torch.sqrt(tau)
    d1 = (torch.log(Sd / K) + (r_d - r_f + 0.5*sig*sig)*tau) / vol
    d2 = d1 - vol
    Phi = lambda z: 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))  # normal CDF via erf

    # GK formula (undiscounted): u = S e^{-r_f τ} Φ(d1) - K e^{-r_d τ} Φ(d2)
    price = Sd * torch.exp(-r_f*tau) * Phi(d1) - K * torch.exp(-r_d*tau) * Phi(d2)
    return torch.clamp(price, min=0.0)

def _gk_delta_torch(S, t, K, r_d, r_f, sigma, T, device="cpu"):
    """
    Delta (∂u/∂S) du call GK (non actualisé u), en Torch.
    u = S e^{-r_f τ} Φ(d1) - K e^{-r_d τ} Φ(d2)
    ⇒ Δ_u = e^{-r_f τ} Φ(d1)
    """
    S = torch.as_tensor(S, dtype=torch.float32, device=device)
    t = torch.as_tensor(t, dtype=torch.float32, device=device)
    K = torch.as_tensor(K, dtype=torch.float32, device=device)
    tau = torch.clamp(torch.tensor(T, dtype=torch.float32, device=device) - t, min=1e-12)
    Sd  = torch.clamp(S, min=1e-12)
    sig = torch.as_tensor(sigma, dtype=torch.float32, device=device)

    vol = torch.clamp(sig * torch.sqrt(tau), min=1e-8)  # stabilité
    d1  = (torch.log(Sd / K) + (r_d - r_f + 0.5*sig*sig)*tau) / vol
    Phi = lambda z: 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
    delta_u = torch.exp(-r_f * tau) * Phi(d1)
    return delta_u



# - Model 
class WPINNVol(nn.Module):
    """
    Neural network for discounted value `w(S, τ, K, σ)` with hard-wired payoff.

    Definition
    ---
    w(S, τ, K, σ) = ReLU(S - K) + τ * softplus( NN([S, τ, K, σ, log(S/K)]) )

    Notes
    -----
    - The `ReLU(S-K)` term enforces the exact terminal payoff at τ=0.
    - Multiplying the learned residual by τ forces the time value to vanish at expiry.
    - `softplus` keeps the learned time value nonnegative (stability / no-arb hint).
    - We input `log(S/K)` to help the net learn moneyness symmetry.

    Normalizations
    
    Inputs are lightly normalized around K_ref, T, and σ_ref to aid optimization.
    """
    def __init__(self, K_ref, T, sigma_ref=0.12, hidden=128, layers=6):
        super().__init__()
        self.K_ref = float(K_ref)
        self.T = float(T)
        self.sigma_ref = float(sigma_ref)

        # Simple MLP backbone with Tanh; ends with linear to produce one scalar.
        h = hidden
        net = [nn.Linear(5, h), nn.Tanh()]
        for _ in range(layers-1):
            net += [nn.Linear(h, h), nn.Tanh()]
        net += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*net)
        self.softplus = nn.Softplus(beta=1.0)

        # Xavier init typically works well with Tanh activations.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Parameters
        ---
        x : torch.Tensor of shape (N, 4)
            Columns are [S, τ, K, σ].
        """
        # Unpack raw inputs
        S   = x[:,0:1]
        tau = x[:,1:2]
        K   = x[:,2:3]
        sig = x[:,3:4]

        # Light normalizations around references to ease training dynamics
        Sn   = (S - self.K_ref)/(self.K_ref + 1e-8)
        tn   = tau / (self.T + 1e-12)
        Kn   = (K - self.K_ref)/(self.K_ref + 1e-8)
        sign = sig/(self.sigma_ref + 1e-12)
        mn   = torch.log(torch.clamp(S, min=1e-12)/torch.clamp(K, min=1e-12))  # log-moneyness

        # Network predicts (pre-softplus) time-value density
        n = self.net(torch.cat([Sn, tn, Kn, sign, mn], dim=1))

        # Hard terminal payoff + positive time value
        payoff = torch.relu(S - K)
        w = payoff + tau * self.softplus(n)
        return w


# - Autodiff 

def _w_derivs(model, x):
    """Compute w and its derivatives needed by the PDE.

    Returns
    
    w      : w(S,τ,K,σ)
    w_tau  : ∂w/∂τ
    w_S    : ∂w/∂S
    w_SS   : ∂²w/∂S²
    """
    if not x.requires_grad:
        x = x.clone().detach().requires_grad_(True)

    w = model(x)

    # First-order grads wrt [S, τ, K, σ]
    grads = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w),
                                create_graph=True, retain_graph=True)[0]
    w_S   = grads[:,0:1]
    w_tau = grads[:,1:2]

    # Second derivative w.r.t S
    w_SS  = torch.autograd.grad(w_S, x, grad_outputs=torch.ones_like(w_S),
                                create_graph=True, retain_graph=True)[0][:,0:1]
    return w, w_tau, w_S, w_SS


# Loss 
class GKLossWVol:
    """
    Composite loss enforcing the GK PDE, boundary/terminal conditions,
    a soft lower bound, and anchor supervision.

    PDE (discounted form)
    -
    ∂w/∂τ = 0.5 σ² S² ∂²w/∂S² + (r_d - r_f) S ∂w/∂S

    where w = e^{r_d τ} u and u is the undiscounted price.
    """
    def __init__(self, r_d, r_f, T, S_max,
                 K_focus=(1.18,1.20), focus_sigmaK=0.012,
                 anchor_w=1.0, anchor_delta_w=0.4):
        self.r_d, self.r_f, self.T = float(r_d), float(r_f), float(T)
        self.S_max = float(S_max)

        # For importance sampling / weighting around strikes of interest
        self.K_focus = (float(K_focus[0]), float(K_focus[1]))
        self.K_mid   = 0.5*(self.K_focus[0]+self.K_focus[1])
        self.focus_sigmaK = float(focus_sigmaK)

        # Relative weights for anchor price and anchor delta terms
        self.anchor_w = float(anchor_w)
        self.anchor_delta_w = float(anchor_delta_w)

    def pde(self, model, x_pde):
        """Weighted PDE residual loss over interior points.

        We upweight near-the-money, mid-maturity, and near-focus-K regions
        to help the network nail the surface where it matters most.
        """
        # x = [S, τ, K, σ]
        S   = x_pde[:,0:1]
        tau = x_pde[:,1:2]
        K   = x_pde[:,2:3]
        sig = x_pde[:,3:4]

        w, w_tau, w_S, w_SS = _w_derivs(model, x_pde)
        resid = w_tau - (0.5*(sig**2)*(S**2)*w_SS + (self.r_d - self.r_f)*S*w_S)

        # Heuristic weights: ATM (via log-moneyness), time, and strike focus
        mn   = torch.log(torch.clamp(S,1e-12)/torch.clamp(K,1e-12))
        w_atm= torch.exp(-(mn**2)/(2*(0.12**2)))
        w_t  = torch.clamp(tau/self.T, min=0.0)
        wK   = torch.exp(-((K-self.K_mid)**2)/(2*(self.focus_sigmaK**2)))
        wght = 0.25 + 0.75*(w_atm*w_t) + 1.25*wK

        # Normalize by (1+|w|) to reduce scale sensitivity across the domain
        return torch.mean(wght * (resid / (1.0 + torch.abs(w))).pow(2))

    def bc(self, model, x0, xSmax):
        """Boundary conditions in discounted space.

        - At S=0    : w ≈ 0
        - At S=Smax : w ≈ S e^{(r_d - r_f) τ} - K
        """
        L0 = torch.mean(model(x0).pow(2))

        Smax = xSmax[:,0:1]
        tau  = xSmax[:,1:2]
        K    = xSmax[:,2:3]
        target = Smax*torch.exp((self.r_d - self.r_f)*tau) - K
        Lmax = torch.mean((model(xSmax) - target).pow(2)) / 2.0
        return L0 + Lmax

    def term(self, model, xT0):
        """Terminal condition at τ=0: w = max(S-K, 0)."""
        S = xT0[:,0:1]
        K = xT0[:,2:3]
        return torch.mean((model(xT0) - torch.relu(S-K)).pow(2))

    def lb_penalty(self, model, x):
        """Soft lower-bound penalty: w ≥ S e^{(r_d - r_f) τ} - K.

        This mirrors the no-arbitrage bound in discounted space and discourages
        underpricing. (We apply a small coefficient to keep it gentle.)
        """
        S   = x[:,0:1]
        tau = x[:,1:2]
        K   = x[:,2:3]
        lower = S*torch.exp((self.r_d - self.r_f)*tau) - K
        return 0.02*torch.mean(torch.relu(lower - model(x)))

    def anchors(self, model, xA, wA, dAw):
        """Anchor supervision using GK closed-form price and delta.

        Encourages fast/stable convergence and preserves correct shape.
        """
        w_pred = model(xA)
        Lw = torch.mean((w_pred - wA).pow(2))

        # Delta of w wrt S (∂w/∂S) should match the discounted delta target
        _, _, w_S, _ = _w_derivs(model, xA)
        Ld = torch.mean((w_S - dAw).pow(2))
        return self.anchor_w*Lw + self.anchor_delta_w*Ld

    def total(self, model, x_pde, x0, xSmax, xT0, xA, wA, dAw):
        """Compute total loss and return component breakdown."""
        Lp = self.pde(model, x_pde)
        Lb = self.bc(model, x0, xSmax)
        Lt = self.term(model, xT0)
        Ll = self.lb_penalty(model, x_pde)
        La = self.anchors(model, xA, wA, dAw)
        return Lp + Lb + 0.1*Lt + Ll + La, (Lp, Lb, Lt, Ll, La)


#  Trainer 
class TrainerVol:
    """
    Training harness for the volatility-aware PINN.

    Responsibilities
    -
    - Sample interior points (PDE), boundaries, terminal, and anchors.
    - Emphasize near-moneyness and focus strikes via sampling/weights.
    - Run the optimization loop with gradient clipping and lr scheduling.
    """
    def __init__(self, model, loss, device="cpu",
                 S0=1.11, K_bounds=(0.9,1.35), K_focus=(1.18,1.20),
                 vol_bounds=(0.08,0.20), anchor_sigmas=(0.10,0.12,0.14)):
        self.model, self.loss, self.device = model.to(device), loss, device
        self.S0 = float(S0)
        self.K_bounds = (float(K_bounds[0]), float(K_bounds[1]))
        self.K_focus  = (float(K_focus[0]),  float(K_focus[1]))
        self.vol_bounds = (float(vol_bounds[0]), float(vol_bounds[1]))
        self.anchor_sigmas = [float(v) for v in anchor_sigmas]

    # --- Sampling utilities --
    def _sample_K(self, n, frac_focus=0.6):
        """Sample strikes with some mass around a focus band for fidelity."""
        nf = int(frac_focus*n); nb = n-nf
        Kf = torch.rand(nf,1,device=self.device)*(self.K_focus[1]-self.K_focus[0])+self.K_focus[0]
        Kb = torch.rand(nb,1,device=self.device)*(self.K_bounds[1]-self.K_bounds[0])+self.K_bounds[0]
        return torch.cat([Kf,Kb], dim=0)

    def _sample_sigma(self, n, frac_focus=0.5):
        """Sample σ with a portion drawn from preferred anchor sigmas."""
        nf = int(frac_focus*n); nb = n-nf
        sig_f = torch.tensor(np.random.choice(self.anchor_sigmas, size=nf),
                             dtype=torch.float32, device=self.device).view(-1,1)
        a,b = self.vol_bounds
        sig_b = torch.rand(nb,1,device=self.device)*(b-a)+a
        return torch.cat([sig_f, sig_b], dim=0)

    def _around(self, center, pct, n):
        """Sample n values in [(1-pct)*center, (1+pct)*center]."""
        lo = max(center*(1-pct), 1e-6)
        hi = center*(1+pct)
        return torch.rand(n,1,device=self.device)*(hi-lo)+lo

    def _anchors(self, T):
        """Build a grid of anchor points and their GK targets (price & delta).

        We compute u via GK, then convert to discounted w = e^{r_d τ} u.
        We also convert delta_u to delta in w-space: ∂w/∂S = e^{(r_d - r_f) τ} ∂u/∂S.
        """
        # Small grid around focus region for K, τ, S, and specific σ values
        Ks   = torch.tensor([self.K_focus[0], 0.5*(self.K_focus[0]+self.K_focus[1]), self.K_focus[1]],
                            dtype=torch.float32, device=self.device).view(-1,1)
        taus = torch.tensor([0.2*T, 0.5*T, 0.8*T], dtype=torch.float32, device=self.device).view(-1,1)
        Ss   = torch.tensor([self.S0*0.98, self.S0, self.S0*1.02],
                            dtype=torch.float32, device=self.device).view(-1,1)
        sigs = torch.tensor(self.anchor_sigmas, dtype=torch.float32, device=self.device).view(-1,1)

        xs=[]
        for K in Ks:
            for tau in taus:
                for S in Ss:
                    for sig in sigs:
                        xs.append(torch.stack([S.squeeze(), tau.squeeze(), K.squeeze(), sig.squeeze()]))
        xA = torch.stack(xs, dim=0).view(-1,4).requires_grad_(True)

        # Targets in w-space (discounted form)
        tA = (torch.tensor(self.loss.T, device=self.device) - xA[:,1:2])
        uA = _gk_call_torch(xA[:,0:1], tA, xA[:,2:3],
                            r_d=self.loss.r_d, r_f=self.loss.r_f, sigma=xA[:,3:4],
                            T=self.loss.T, device=self.device)
        wA = torch.exp(self.loss.r_d * xA[:,1:2]) * uA

        delta_u = _gk_delta_torch(xA[:,0:1], tA, xA[:,2:3],
                                  r_d=self.loss.r_d, r_f=self.loss.r_f, sigma=xA[:,3:4],
                                  T=self.loss.T, device=self.device)
        dAw = torch.exp((self.loss.r_d - self.loss.r_f)*xA[:,1:2]) * delta_u
        return xA, wA.detach(), dAw.detach()

    def _batches(self, n_pde=4096, n_b=256, n_T=512, T=1.0, S_max=3.0):
        """Draw one training batch of all point types (PDE, BC, terminal, anchors)."""
        # Interior (PDE) points: bias toward near-moneyness and around S0
        Kp   = self._sample_K(n_pde, frac_focus=0.6)
        sigp = self._sample_sigma(n_pde, frac_focus=0.5)

        n_near = int(0.5*n_pde)
        S_nearK = self._around(1.0, 0.10, n_near) * Kp[:n_near]  # ~S≈K
        S_nearS = self._around(self.S0, 0.10, n_pde-n_near)      # ~S≈S0
        S_pde   = torch.clamp(torch.cat([S_nearK, S_nearS], dim=0), 1e-6, S_max)
        tau_pde = torch.rand(n_pde,1,device=self.device)*(0.8*T - 0.2*T)+0.2*T
        x_pde   = torch.cat([S_pde, tau_pde, Kp, sigp], dim=1).requires_grad_(True)

        # Boundary points (S=0 and S=S_max) across τ, K, σ
        tau_b = torch.rand(n_b,1,device=self.device)*T
        Kb    = self._sample_K(n_b, frac_focus=0.6)
        sigb  = self._sample_sigma(n_b, frac_focus=0.5)
        x0    = torch.cat([torch.zeros_like(tau_b), tau_b, Kb, sigb], dim=1).requires_grad_(True)
        xSmax = torch.cat([torch.full_like(tau_b, S_max), tau_b, Kb, sigb], dim=1).requires_grad_(True)

        # Terminal points at τ=0 across a diverse (S,K,σ)
        Kt   = self._sample_K(n_T, frac_focus=0.6)
        sigt = self._sample_sigma(n_T, frac_focus=0.5)
        S_T  = torch.where(torch.rand(n_T,1,device=self.device)<0.7,
                           self._around(1.0,0.25,n_T)*Kt,
                           torch.rand(n_T,1,device=self.device)*S_max)
        xT0  = torch.cat([torch.clamp(S_T,1e-6,S_max), torch.zeros_like(S_T), Kt, sigt], dim=1).requires_grad_(True)

        # Anchor grid and targets
        xA, wA, dAw = self._anchors(T)
        return x_pde, x0, xSmax, xT0, xA, wA, dAw

    def train(self, epochs=1000, lr=1e-3, print_every=200):
        """Main training loop with Adam, ReduceLROnPlateau, and grad clipping."""
        opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-8)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.7, patience=200, verbose=False)

        t0 = time.time(); best=float("inf"); best_state=None
        for ep in range(1, epochs+1):
            opt.zero_grad()

            # Freshly resample each epoch for PINN-style stochastic training
            x_pde,x0,xSmax,xT0,xA,wA,dAw = self._batches(T=self.loss.T, S_max=self.loss.S_max)
            total,(Lp,Lb,Lt,Ll,La) = self.loss.total(self.model, x_pde,x0,xSmax,xT0, xA,wA,dAw)

            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step(); sched.step(total)

            # Track best state for a simple early-best checkpoint
            if total.item()<best:
                best=total.item(); best_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}

            if ep % print_every==0:
                print(f"Epoch {ep:5d}  Total:{total.item():.6f}  PDE:{Lp.item():.6f}  BC:{Lb.item():.6f}  "
                      f"T:{Lt.item():.6f}  LB:{Ll.item():.6f}  AN:{La.item():.6f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Loaded best state. Best loss={best:.6f}. Train time {time.time()-t0:.1f}s")


# --- Public training wrapper 

def run_fx_pinn_sigma(
    S0=1.11, K_ref=1.10, r_d=0.04, r_f=0.03, T=1.0,
    K_focus=(1.18,1.20), K_bounds=(0.90,1.35),
    vol_bounds=(0.08,0.20), anchor_sigmas=(0.10,0.12,0.14),
    S_max_factor=2.5, epochs=1000, device=None
):
    """Train a σ-aware GK PINN and return the trained model.

    Parameters
    ---
    S0 : float
        Reference spot level used for sampling near-moneyness.
    K_ref : float
        Reference strike for input normalization.
    r_d, r_f : float
        Domestic and foreign rates.
    T : float
        Maturity horizon used for τ sampling.
    K_focus, K_bounds : tuple
        Strike focus band and global bounds for sampling.
    vol_bounds, anchor_sigmas : tuple, tuple/list
        σ sampling range and a few preferred anchor σ values.
    S_max_factor : float
        Sets S_max = S_max_factor * max strike bound.
    epochs : int
        Training epochs.
    device : Optional[str]
        Torch device; auto-detects CUDA if None.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    S_max = S_max_factor * K_bounds[1]

    # Reproducibility for demos (feel free to remove in production)
    torch.manual_seed(0); np.random.seed(0)

    model = WPINNVol(K_ref=K_ref, T=T, sigma_ref=float(np.mean(vol_bounds)), hidden=128, layers=6)
    loss  = GKLossWVol(r_d=r_d, r_f=r_f, T=T, S_max=S_max,
                       K_focus=K_focus, focus_sigmaK=0.012,
                       anchor_w=1.0, anchor_delta_w=0.4)
    trainer = TrainerVol(model, loss, device=device, S0=S0,
                         K_bounds=K_bounds, K_focus=K_focus,
                         vol_bounds=vol_bounds, anchor_sigmas=anchor_sigmas)

    print(f"Training on {device}. Vol range={vol_bounds}, anchors={anchor_sigmas}")
    trainer.train(epochs=epochs, lr=1e-3, print_every=200)
    return model


# --- Public prediction helper ------

def pinn_price(model, r_d, T, S, K, tau, sigma):
    """Return the **undiscounted** call price `u` from a trained PINN.

    Notes
    -----
    - The network outputs `w`; we convert to `u = e^{-r_d τ} * w`.
    - `T` is accepted for API symmetry with other utilities but is not
      used here (τ is passed explicitly).
    """
    dev = next(model.parameters()).device
    with torch.no_grad():
        x = torch.tensor([[S, tau, K, sigma]], dtype=torch.float32, device=dev)
        w = model(x).item()
        return float(w * np.exp(-r_d * tau))


# ------ (Optional) save/load 

def save_model(model, path="fx_pinn_sigma.pt"):
    """Save model parameters to a .pt file."""
    torch.save(model.state_dict(), path)


def load_model(path, K_ref, T, sigma_ref, hidden=128, layers=6, device=None):
    """Load model parameters and build the matching architecture.

    Returns a ready-to-eval model placed on `device`.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = WPINNVol(K_ref=K_ref, T=T, sigma_ref=sigma_ref, hidden=hidden, layers=layers).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# Example usage ---
if __name__ == "__main__":
    # Example hyperparameters (feel free to tweak)
    S0, r_d, r_f, T = 1.16604, 0.041190, 0.01848, 1.3

    # Train (e.g., 3000 epochs). Anchors include a few σ values to guide training.
    model = run_fx_pinn_sigma(
        S0=S0, K_ref=1.10, r_d=r_d, r_f=r_f, T=T,
        K_focus=(1.18,1.20), K_bounds=(0.90,1.35),
        vol_bounds=(0.08,0.20), anchor_sigmas=(0.10,0.12,0.09),
        S_max_factor=2.5, epochs=3000
    )

    # Predict anywhere in (S, K, τ, σ):
    price = pinn_price(model, r_d=r_d, T=T, S=1.16, K=1.19, tau=1, sigma=0.08)
    print(f"PINN price: {price:.6f}")

    # Save weights (optional)
    save_model(model, "fx_pinn_sigma.pt")
