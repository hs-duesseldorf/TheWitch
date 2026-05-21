from __future__ import annotations

from shared.events import HandEvent


_SYSTEM_PROMPT = (
    "/no_think\n"
    "Antworte ausschliesslich auf Deutsch.\n"
    "Gib nur die finale gesprochene Antwort aus.\n"
    "Beginne sofort mit der Vorhersage, ohne Analyse oder Vorrede.\n"
    "Schreibe eine mystische Zukunftsvorhersage wie eine Hexe oder Wahrsagerin.\n"
    "Deute die menschliche Hand anhand von Proportionen, Handflaeche, Fingern und Linien.\n"
    "Sprich ueber kommende Ereignisse, Entscheidungen, Warnungen oder verborgene Chancen.\n"
    "Antworte mit 5 bis 6 vollstaendigen Saetzen.\n"
    "Gib eine detaillierte Deutung, aber ohne Wortzaehlungen oder formale Hinweise.\n"
    "Es geht immer um eine menschliche Hand, niemals um ein Handtuch.\n"
    "Verwende nie die Woerter Handtuch, Tuch oder Stoff.\n"
    "Benutze konkrete Handlese-Bilder wie Lebenslinie, Schicksalslinie, Finger oder Handflaeche.\n"
    "Erfinde keine Messwerte und erwaehne keine technischen Begriffe.\n"
    "Kein Markdown, keine Klammern, keine Emojis.\n"
    "Ton: weise, leicht dunkel, konkret.\n"
    "/no_think\n"
)


def build_prompt(hand_event: HandEvent | None) -> str:
    if not hand_event:
        return _SYSTEM_PROMPT + "\nKeine Handdaten verfuegbar."

    trigger = hand_event.trigger.value if hand_event.trigger else "unknown"
    hand = _hand_label(hand_event)
    lengths = hand_event.lengths
    vector = hand_event.vector

    data_desc = []
    if lengths:
        fingers = ", ".join(f"{k}: {v:.1f}" for k, v in lengths.items() if v)
        if fingers:
            data_desc.append(f"Fingerlaengen: {fingers}")

    if vector:
        data_desc.append("Linien- und Handflaechenmuster sind erkannt")

    data_str = " ".join(data_desc) if data_desc else "Keine messbaren Daten"

    return f"{_SYSTEM_PROMPT}\nBeobachtung: {trigger}. Gesehene Hand: {hand}. {data_str}"


def _hand_label(hand_event: HandEvent) -> str:
    if not hand_event.hand:
        return "unbekannte Hand"
    if hand_event.hand.value == "left":
        return "linke Hand"
    if hand_event.hand.value == "right":
        return "rechte Hand"
    return "unbekannte Hand"
