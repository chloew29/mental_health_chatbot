# crisis_policy.py
# Centralized safety / crisis-handling messages

from dataclasses import dataclass

@dataclass(frozen=True)
class Helpline:
    immediate: str
    textline: str | None = None
    note: str | None = None

HELPLINES: dict[str, Helpline] = {
    "US": Helpline(
        immediate="Call or text **988** (Suicide & Crisis Lifeline), or dial **911** for immediate danger.",
        textline="Text **HOME** to **741741** for Crisis Text Line."
    ),
    "CA": Helpline(
        immediate="Call or text **988** (Suicide Crisis Helpline), or dial **911** for immediate danger.",
        textline=None
    ),
    "UK": Helpline(
        immediate="Call **116 123** (Samaritans), or **999** for immediate danger.",
        textline=None
    ),
    "HK": Helpline(
        immediate="Call **2896 0000** (Suicide Prevention Services), or **999** for immediate danger.",
        textline=None
    ),
    "DEFAULT": Helpline(
        immediate="Please reach your local emergency number. If available, contact your national crisis line.",
        textline=None,
        note="Try searching “suicide crisis line <your country>”."
    ),
}

def _helpline(locale: str) -> Helpline:
    return HELPLINES.get(locale.upper(), HELPLINES["DEFAULT"])

def crisis_response(locale: str = "US") -> str:
    h = _helpline(locale)
    parts = [
        "I'm really sorry you're going through this. You deserve care and support.",
        "I can stay with you here, but I’m not a substitute for professional help.",
        h.immediate
    ]
    if h.textline:
        parts.append(h.textline)
    if h.note:
        parts.append(h.note)
    parts.append(
        "If it helps, we can try a brief grounding exercise together (e.g., name 5 things you see, "
        "4 you can touch, 3 you hear, 2 you smell, 1 you taste). Would you like that?"
    )
    return " ".join(parts)

def supportive_footer() -> str:
    return (
        "\n\n*I can offer supportive conversation, but I’m not a medical professional. "
        "If you might hurt yourself or someone else, please contact emergency services or a local crisis line.*"
    )

def low_risk_openers() -> list[str]:
    return [
        "Thanks for sharing that with me. It sounds really tough.",
        "I hear you. Many people feel this way sometimes, and you're not alone.",
        "I’m here with you. Would you like to share a bit more about what’s been hardest?"
    ]
