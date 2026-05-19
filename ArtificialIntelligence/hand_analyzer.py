from __future__ import annotations

from shared.events import HandEvent


_SYSTEM_PROMPT = (
    "Schreibe eine kurze mystische Handlese-Wahrsagung auf Deutsch.\n"
    "Antwort nur mit 2 Saetzen, ohne Einleitung und ohne Aufzaehlung.\n"
    "Schreibe natuerlich gesprochen, mit kurzen vollstaendigen Saetzen.\n"
    "Kein Markdown, keine Klammern, keine Emojis.\n"
    "Ton: weise, leicht dunkel, konkret.\n"
)


def build_prompt(hand_event: HandEvent | None) -> str:
    if not hand_event:
        return _SYSTEM_PROMPT + "\nKeine Handdaten verfuegbar."

    trigger = hand_event.trigger.value if hand_event.trigger else "unknown"
    hand = hand_event.hand.value if hand_event.hand else "unknown"
    lengths = hand_event.lengths
    vector = hand_event.vector

    data_desc = []
    if lengths:
        fingers = ", ".join(f"{k}: {v:.1f}" for k, v in lengths.items() if v)
        if fingers:
            data_desc.append(f"Fingerlaengen: {fingers}")

    if vector:
        vector_summary = f"Merkmalsvektor mit {len(vector)} Werten"
        data_desc.append(vector_summary)

    data_str = " ".join(data_desc) if data_desc else "Keine messbaren Daten"

    return f"{_SYSTEM_PROMPT}\nTrigger: {trigger}. Hand: {hand}. {data_str}"