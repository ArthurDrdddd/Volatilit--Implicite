import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Fonction Black-Scholes pour le prix d'un call
def black_scholes_call(S, K, T, r, sigma):
    """
    Calcule le prix d'un Call selon le modèle de Black-Scholes.

    Paramètres:
    S : float - Prix actuel de l'actif sous-jacent
    K : float - Prix d'exercice de l'option
    T : float - Temps restant jusqu'à l'échéance (en années)
    r : float - Taux sans risque
    sigma : float - Volatilité du sous-jacent (écart-type)

    Retourne:
    float : Prix du Call
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Fonction pour calculer la volatilité implicite
def implied_volatility(S, K, T_days, r, market_price, tol=1e-5, days_in_year=365):
    """
    Calcule la volatilité implicite via une recherche numérique.

    Paramètres:
    S : float - Prix actuel de l'actif sous-jacent
    K : float - Prix d'exercice de l'option
    T_days : int - Temps restant jusqu'à l'échéance (en jours)
    r : float - Taux sans risque
    market_price : float - Prix observé du Call sur le marché
    tol : float - Tolérance pour la précision du résultat
    days_in_year : int - Nombre de jours dans une année (252 pour jours ouvrés)

    Retourne:
    float : Volatilité implicite
    """
    # Convertir T de jours à années
    T = T_days / days_in_year

    def difference(sigma):
        return black_scholes_call(S, K, T, r, sigma) - market_price

    # Recherche de la volatilité entre 0.001 et 500%
    try:
        return brentq(difference, 1e-5, 5.0, xtol=tol)
    except ValueError:
        return None

# Exemple d'utilisation
if __name__ == "__main__":
    # Paramètres de l'option
    S0 = 57.09       # Prix actuel de l'actif sous-jacent
    K = 60        # Prix d'exercice
    T_days = 28    # Temps jusqu'à l'échéance (en jours)
    r = 0.05       # Taux sans risque
    C_market = 0.40  # Prix observé de l'option call sur le marché

    # Calcul de la volatilité implicite
    vol_implicite = implied_volatility(S0, K, T_days, r, C_market)

    if vol_implicite is not None:
        print(f"Volatilité implicite: {vol_implicite:.4f} (ou {vol_implicite * 100:.2f}%)")
    else:
        print("La volatilité implicite n'a pas pu être calculée avec les paramètres fournis.")

