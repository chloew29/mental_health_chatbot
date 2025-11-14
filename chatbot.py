# chatbot_litellm.py
# Hybrid chatbot using LiteLLM for inference (API or local Ollama)

import os
from litellm import completion
from crisis_policy import crisis_response, supportive_footer, low_risk_openers


# === CONFIGURATION ===
# (per your request, keep API key inline)
os.environ["OPENAI_API_KEY"] = "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"

# === Configurable Models ===
RISK_MODEL = os.getenv("RISK_MODEL", "gpt-5-mini")       # for risk classification
RESP_MODEL = os.getenv("RESP_MODEL", "gpt-5-mini")       # for empathetic responses
LOCALE = os.getenv("USER_LOCALE", "US")

# === Risk detection prompt ===
RISK_PROMPT = """
You are a classifier that determines if a user message suggests high mental health risk.
Output only one word: HIGH or NORMAL.

User message:
"{text}"
"""

# === Empathetic response prompt ===
RESP_PROMPT = """
You are a supportive, empathetic counselor assistant.
Respond kindly and calmly to the user's message. Avoid giving medical advice.
Be warm, validating, and concise.

User message:
"{text}"
"""

def detect_risk(user_text: str) -> str:
    """Uses LiteLLM to classify risk level via language model."""
    response = completion(
        model=RISK_MODEL,
        messages=[
            {"role": "system", "content": "You are a text classifier."},
            {"role": "user", "content": RISK_PROMPT.format(text=user_text)}
        ],
        temperature=1,
    )
    content = response["choices"][0]["message"]["content"].strip().upper()
    if "HIGH" in content:
        return "HIGH"
    return "NORMAL"

def generate_response(user_text: str) -> str:
    """Generates an empathetic response using LiteLLM."""
    response = completion(
        model=RESP_MODEL,
        messages=[
            {"role": "system", "content": "You are a kind and supportive counselor assistant."},
            {"role": "user", "content": RESP_PROMPT.format(text=user_text)}
        ],
        temperature=0.7,
        max_tokens=200,
    )
    reply = response["choices"][0]["message"]["content"].strip()
    return reply + supportive_footer()

def chatbot(user_input: str):
    risk = detect_risk(user_input)
    if risk == "HIGH":
        return crisis_response(locale=LOCALE)
    else:
        opener = low_risk_openers()[0]
        reply = generate_response(user_input)
        return f"{opener}\n\n{reply}"

# === Simple CLI ===
if __name__ == "__main__":
    print("Mental Health Chatbot (LiteLLM version)\nType 'quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit"}:
            print("Bot: Take care. You are not alone ❤️")
            break
        if not user:
            continue
        bot_reply = chatbot(user)
        print(f"Bot: {bot_reply}\n")
