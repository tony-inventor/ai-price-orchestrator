def simular(preco: float) -> float:
    demanda_estimada = max(0.0, 120.0 - preco)
    return round(preco * demanda_estimada, 2)
