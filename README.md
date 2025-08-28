# PRICER-FX-Options-vanilla-and-asian-GK-Heston-and-PINNs
Pricer of fx options ( vanilla and asian) via classical approach, heston model and pinns

## Aperçu
Pricer option FX vanille et asiatique.  
Une **appli Streamlit** affiche le spot EUR/USD, la vol historique, les courbes de taux, et calcule les prix d’options **vanilles** et **asiatiques** avec :
- **Garman–Kohlhagen (GK)** (formule fermée)
- **Monte‑Carlo**
- **PINN (réseaux de neurones informés par la physique)**  

## Contenu
- `pricer.py` — interface **Streamlit** (lance l’appli)
- `methods.py` — formules/pricers : **GK**, **Monte‑Carlo**, **asiatiques** (géométrique + MC)
- `fx_data.py` — données EUR/USD via **yfinance** (historique + vol 30j)
- `inter_rates.py` — lecture de **rates.xlsx**, construction des **courbes zéro USD/EUR** + tracés
- `pinns_vanilla.py` — PINN pour vanilles GK (σ‑aware)
- `pinns_asian.py` — PINN pour options asiatiques
- `fx_pinn_sigma.pt`, `fx_asian_pinn.pt` — poids de modèles PINN (facultatif)
- `rates.xlsx` - taux eur dol 

>  Fichier requis : **`rates.xlsx`** (au même niveau que `pricer.py`).

## Prérequis
- **Python 3.10+**
- packages: yfinance 
- Internet pour télécharger `EURUSD=X` la première fois (via yfinance).

### Installer les paquets
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install numpy pandas scipy matplotlib plotly streamlit yfinance torch openpyxl
```

##  Lancer l’appli
Depuis le dossier du projet :
```bash
streamlit run pricer.py
```
Ouvrir le lien local proposé par Streamlit dans le navigateur.

## Utilisation rapide
1. Vérifier que **`rates.xlsx`** est présent (courbes USD/EUR).
2. Lancer l’appli. La **barre latérale** permet de choisir : spot/strike, maturité, notional, méthode (GK/MC/Asiatique/PINN).
3. La page montre :
   - **Courbe EUR/USD** et **volatilité 30 jours** (calculée).
   - **Courbes de taux** domestique/étranger à partir de `rates.xlsx`.
   - Le **prix** et des **graphiques** selon la méthode choisie.

>  Les modèles **PINN** ne sont utilisés que si les fichiers `.pt` sont disponibles dans le dossier.


