from pricing_engine import prever_demanda


def simular(
    preco: float, a: float, b1: float, b2: float, preco_concorrente: float
) -> tuple[float, float]:
    """
    Simula a receita e demanda usando o modelo de elasticidade próprio e cruzado.
    Retorna (receita, demanda).
    """
    demanda_estimada = prever_demanda(preco, a, b1, b2, preco_concorrente)
    receita = preco * demanda_estimada
    return round(receita, 2), demanda_estimada
