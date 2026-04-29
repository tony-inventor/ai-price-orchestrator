from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression


def carregar_vendas(path: Path) -> pd.DataFrame:
    """Carrega os dados de vendas do arquivo CSV."""
    return pd.read_csv(path)


def treinar_modelo(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Treina um modelo de regressão múltipla para capturar a elasticidade própria
    e a elasticidade cruzada com a concorrência.
    Retorna: (intercepto, b1_elasticidade_propria, b2_elasticidade_cruzada)
    """
    df_valid = df[
        (df["preco"] > 0) & (df["quantidade"] > 0) & (df["preco_concorrente"] > 0)
    ].copy()

    if df_valid.empty:
        # Fallback para regressão simples se não houver dados de concorrência
        X = np.log(df["preco"].values.reshape(-1, 1))
        y = np.log(df["quantidade"].values)
        model = LinearRegression().fit(X, y)
        return model.intercept_, model.coef_[0], 0.0

    log_q = np.log(df_valid["quantidade"].values)
    log_p_proprio = np.log(df_valid["preco"].values)
    log_p_concorrente = np.log(df_valid["preco_concorrente"].values)

    X = np.column_stack((log_p_proprio, log_p_concorrente))
    model = LinearRegression().fit(X, log_q)

    return model.intercept_, model.coef_[0], model.coef_[1]


def prever_demanda(
    preco: float, a: float, b1: float, b2: float, preco_concorrente: float
) -> float:
    """Previsão de demanda considerando o impacto do preço próprio e do concorrente."""
    if preco <= 0 or preco_concorrente <= 0:
        return 0.0
    log_q = a + (b1 * np.log(preco)) + (b2 * np.log(preco_concorrente))
    return float(np.exp(log_q))


def otimizar_preco(
    custo: float,
    a: float,
    b1: float,
    b2: float,
    preco_concorrente: float,
    max_preco: int = 200,
) -> tuple[int, float]:
    """Otimiza o preço para maximizar o lucro considerando a concorrência."""

    def objetivo_lucro(p: float) -> float:
        demanda = prever_demanda(p, a, b1, b2, preco_concorrente)
        lucro = (p - custo) * demanda
        return -lucro

    res = minimize_scalar(objetivo_lucro, bounds=(custo, max_preco), method="bounded")
    melhor_preco = int(round(res.x))
    lucro_maximo = -res.fun
    return melhor_preco, lucro_maximo
