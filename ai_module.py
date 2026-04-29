import os
from typing import Optional

import openai


def construir_prompt(custo: float, preco_concorrente: float, demanda: float) -> str:
    return (
        "Você é um especialista em precificação.\n"
        f"Custo: {custo}\n"
        f"Preço concorrente: {preco_concorrente}\n"
        f"Demanda: {demanda}\n"
        "Sugira o melhor preço considerando: maximizar margem e não perder competitividade.\n"
        "Responda apenas com um número."
    )


def sugerir_preco_com_ia(custo: float, preco_concorrente: float, demanda: float) -> float:
    """
    Uses AI to suggest optimal pricing based on cost, competitor price, and demand.
    Falls back to rule-based logic if AI is unavailable.
    """
    # Try AI first
    ai_price = _get_ai_price_suggestion(custo, preco_concorrente, demanda)
    if ai_price is not None:
        return ai_price

    # Fallback to rule-based logic
    print("AI unavailable, using rule-based pricing...")
    preco_base = max(custo * 1.2, custo + 1)

    if preco_concorrente > preco_base:
        return round(preco_concorrente - 1, 2)

    return round(preco_base, 2)


def _get_ai_price_suggestion(custo: float, preco_concorrente: float, demanda: float) -> Optional[float]:
    """
    Calls OpenAI API to get pricing suggestion.
    Returns None if API call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        return None

    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = construir_prompt(custo, preco_concorrente, demanda)

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using cost-effective model
            messages=[
                {"role": "system", "content": "Você é um especialista em precificação econômica."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3  # Lower temperature for more consistent pricing
        )

        # Extract the price from the response
        content = response.choices[0].message.content.strip()

        # Try to parse as float, handle various formats
        try:
            # Remove currency symbols and extra text, keep only numbers
            import re
            numbers = re.findall(r'\d+\.?\d*', content)
            if numbers:
                return round(float(numbers[0]), 2)
        except (ValueError, IndexError):
            pass

        print(f"Could not parse AI response: {content}")
        return None

    except Exception as e:
        print(f"AI API error: {e}")
        return None
