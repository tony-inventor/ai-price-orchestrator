from pricing_engine import prever_demanda


def simular(preco: float, a: float, b: float) -> tuple[float, float]:
    """
    Simulate revenue and profit using the trained demand model.
    Returns (revenue, demand) based on the elasticity model.
    """
    demanda_estimada = prever_demanda(preco, a, b)
    receita = preco * demanda_estimada
    return round(receita, 2), demanda_estimada
