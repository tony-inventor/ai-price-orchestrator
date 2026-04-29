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
RESULTS_FILE = ROOT / "results.csv"


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
    resultados: list[dict[str, object]] = []

    for produto in df_all["produto"].unique():
        produto_df = df_all[df_all["produto"] == produto]
        a, b = treinar_modelo(produto_df)
        preco_concorrente = float(produto_df["preco_concorrente"].mean())
        demanda_media = float(produto_df["quantidade"].mean())

        melhor_preco, melhor_lucro = otimizar_preco(custo, a, b, preco_concorrente)
        preco_ia = sugerir_preco_com_ia(
            custo,
            preco_concorrente,
            demanda_media,
        )
        receita_estimad, demanda_estimada = simular(melhor_preco, a, b)
        lucro_consistente = receita_estimad - (custo * demanda_estimada)

        # Simular preço sugerido pela IA
        receita_ia, demanda_ia = simular(preco_ia, a, b)
        lucro_ia = receita_ia - (custo * demanda_ia)

        # Escolher o preço com maior lucro
        if lucro_ia > lucro_consistente:
            preco_final = preco_ia
            receita_final = receita_ia
            demanda_final = demanda_ia
            lucro_final = lucro_ia
            metodo = "IA"
        else:
            preco_final = melhor_preco
            receita_final = receita_estimad
            demanda_final = demanda_estimada
            lucro_final = lucro_consistente
            metodo = "Otimização"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        resultados.append(
            {
                "data_hora": timestamp,
                "produto": produto,
                "elasticidade-estimada": round(b, 2),
                "melhor-preço-otimização": melhor_preco,
                "lucro-otimizacao": round(lucro_consistente, 2),
                "preço-ia-sugerido": preco_ia,
                "lucro-ia-sugerido": round(lucro_ia, 2),
                "preço-aplicado-otimização": preco_final,
                "demanda-estimada": round(demanda_final, 2),
                "receita-estimada": round(receita_final, 2),
                "lucro-esperado": round(lucro_final, 2),
            }
        )

        print("=== AI PRICE ORCHESTRATOR ===")
        print(f"Produto: {produto}")
        print(f"Elasticidade estimada: {b:.2f}")
        print(f"Melhor preço (otimização): {melhor_preco} (lucro: {lucro_consistente:.2f})")
        print(f"Preço IA sugerido: {preco_ia} (lucro: {lucro_ia:.2f})")
        print(f"Preço aplicado ({metodo}): {preco_final}")
        print(f"Demanda estimada: {demanda_final:.2f}")
        print(f"Receita estimada: {receita_final:.2f}")
        print(f"Lucro esperado: {lucro_final:.2f}\n")

        salvar_decisao(produto, preco_final, receita_final, lucro_final)

    pd.DataFrame(resultados).to_csv(RESULTS_FILE, index=False)


if __name__ == "__main__":
    main()
