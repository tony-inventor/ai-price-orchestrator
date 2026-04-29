from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def treinar_modelo(df: pd.DataFrame) -> tuple[float, float]:
    # Filter out invalid data: prices and quantities must be positive
    df_valid = df[(df["preco"] > 0) & (df["quantidade"] > 0)]
    
    if df_valid.empty:
        raise ValueError("No valid data for training: all prices or quantities are non-positive.")
    
    X = np.log(df_valid["preco"].values)
    y = np.log(df_valid["quantidade"].values)
    b, a = np.polyfit(X, y, 1)
    return a, b


def prever_demanda(preco: float, a: float, b: float) -> float:
    if preco <= 0:
        raise ValueError(f"Price must be positive for demand prediction, got {preco}")
    return float(np.exp(a + b * np.log(preco)))


def otimizar_preco(
    custo: float, a: float, b: float, preco_concorrente: float, max_preco: int = 120
) -> tuple[int, float]:
    """
    Optimize price considering cost, demand elasticity, and competitive pricing.
    Uses continuous optimization for better accuracy.
    Applies a penalty for prices too far from competitors to maintain market realism.
    """
    def profit(price: float) -> float:
        if price <= 0:
            return -float('inf')
        demanda = prever_demanda(price, a, b)
        lucro_bruto = (price - custo) * demanda
        
        # Apply competitive penalty
        diferenca_preco = abs(price - preco_concorrente)
        penalidade_competitiva = 0
        
        if price > preco_concorrente:
            diferenca_permitida = max(5, preco_concorrente * 0.15)
            if diferenca_preco > diferenca_permitida:
                penalidade_competitiva = demanda * (diferenca_preco / preco_concorrente) * 0.5
        
        return lucro_bruto - penalidade_competitiva

    # Use analytical optimum as initial guess if possible
    if b != -1:  # Avoid division by zero
        price_guess = (custo * b) / (b + 1)
        if price_guess <= 0 or price_guess > max_preco:
            price_guess = (custo + max_preco) / 2
    else:
        price_guess = (custo + max_preco) / 2

    # Optimize using scipy
    result = minimize_scalar(lambda p: -profit(p), bounds=(custo, max_preco), method='bounded')
    
    melhor_preco = int(round(result.x))
    melhor_lucro = profit(melhor_preco)
    
    return melhor_preco, melhor_lucro


def carregar_vendas(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
