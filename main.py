from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ai_module import sugerir_preco_com_ia
from pricing_engine import carregar_vendas, otimizar_preco, treinar_modelo
from simulation import simular


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "sales.csv"
DECISIONS_FILE = DATA_DIR / "decisions.csv"


def salvar_decisao(produto: str, preco: int, receita: float, lucro: float) -> None:
    if DECISIONS_FILE.exists():
        df = pd.read_csv(DECISIONS_FILE)
    else:
        df = pd.DataFrame(
            columns=["data", "produto", "preco_sugerido", "preco_aplicado", "receita", "lucro"]
        )

    nova_linha = {
        "data": datetime.now().strftime("%Y-%m-%d"),
        "produto": produto,
        "preco_sugerido": preco,
        "preco_aplicado": preco,
        "receita": receita,
        "lucro": lucro,
    }

    df = pd.concat([df, pd.DataFrame([nova_linha])], ignore_index=True)
    df.to_csv(DECISIONS_FILE, index=False)


def main() -> None:
    df_all = carregar_vendas(DATA_FILE)
    custo = 50.0

    for produto in df_all["produto"].unique():
        produto_df = df_all[df_all["produto"] == produto]
        a, b = treinar_modelo(produto_df)

        melhor_preco, melhor_lucro = otimizar_preco(custo, a, b)
        preco_ia = sugerir_preco_com_ia(
            custo,
            float(produto_df["preco_concorrente"].mean()),
            float(produto_df["quantidade"].mean()),
        )
        receita_estimad = simular(melhor_preco)

        print("=== AI PRICE ORCHESTRATOR ===")
        print(f"Produto: {produto}")
        print(f"Elasticidade estimada: {b:.2f}")
        print(f"Melhor preço: {melhor_preco}")
        print(f"Preço IA sugerido: {preco_ia}")
        print(f"Receita estimada: {receita_estimad:.2f}")
        print(f"Lucro esperado: {melhor_lucro:.2f}\n")

        salvar_decisao(produto, melhor_preco, receita_estimad, melhor_lucro)


if __name__ == "__main__":
    main()
