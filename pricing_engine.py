from pathlib import Path

import numpy as np
import pandas as pd


def treinar_modelo(df: pd.DataFrame) -> tuple[float, float]:
    X = np.log(df["preco"].values)
    y = np.log(df["quantidade"].values)
    b, a = np.polyfit(X, y, 1)
    return a, b


def prever_demanda(preco: float, a: float, b: float) -> float:
    return float(np.exp(a + b * np.log(preco)))


def otimizar_preco(custo: float, a: float, b: float, max_preco: int = 120) -> tuple[int, float]:
    melhor_preco = int(custo)
    melhor_lucro = -float("inf")

    for preco in range(int(custo), max_preco):
        demanda = prever_demanda(preco, a, b)
        lucro = (preco - custo) * demanda

        if lucro > melhor_lucro:
            melhor_preco = preco
            melhor_lucro = lucro

    return melhor_preco, melhor_lucro


def carregar_vendas(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
