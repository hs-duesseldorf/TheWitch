from __future__ import annotations

import json
import random
from pathlib import Path

from shared.events import HandEvent

from . import hand_analysis


_SYSTEM_PROMPT = (
 "/no_think\n"
    "Antworte ausschliesslich auf Deutsch.\n"
    "Gib nur die finale gesprochene Antwort aus.\n"
    "Beginne sofort mit der Vorhersage, ohne Analyse oder Vorrede.\n"
    "STRIKTE REGEL: Du bist ein reiner Text-Transformator. Erfinde KEINE eigenen Geschichten, Linien oder Metaphern.\n"
    "Nimm die bereitgestellten Basistexte und übersetze sie VOLLSTÄNDIG und OHNE Sinnveränderung in den Tonfall einer weisen, düsteren Wahrsagerin.\n"
    "Das in den Basistexten genannte dominante Element MUSS namentlich und prominent in der Antwort vorkommen.\n"
    "Es geht immer um eine menschliche Hand, niemals um ein Handtuch.\n"
    "Verwende nie die Woerter Handtuch, Tuch oder Stoff.\n"
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
        base_texts = _get_base_texts(trigger, hand_event, lengths)
        if base_texts:
            data_desc.append(base_texts)

    if vector:
        data_desc.append("Linien- und Handflaechenmuster sind erkannt")

    data_str = " ".join(data_desc) if data_desc else "Keine messbaren Daten"

    return f"{_SYSTEM_PROMPT}\nBeobachtung: {trigger}. Gesehene Hand: {hand}. {data_str}"


def _get_base_texts(trigger: str, hand_event: HandEvent, lengths: dict[str, float]) -> str:
    payload = {
        "type": "hand",
        "trigger": trigger,
        "hand": hand_event.hand.value if hand_event.hand else None,
        "lengths": lengths,
    }

    try:
        result = hand_analysis.build_result(payload)
        lines = hand_analysis.GetLines(result)
    except Exception:
        return ""

    dominant = result.get("dominant_element") if result else None

    package_texts = _load_package_texts()
    if not lines and not package_texts:
        return ""

    storyline = _build_storyline(lines, package_texts, dominant)
    if not storyline:
        return ""

    joined = " ".join(line for line in storyline if line)

    return (
        "Hier sind deine verbindlichen Basistexte. Übersetze sie fließend in deinen "
        "mystischen Stil, aber behalte JEDE Information und JEDEN Satzbau-Sinn bei. "
        "Es darf absolut kein Fakt oder Inhalt weggelassen oder verkürzt werden: "
        f"{joined}"
    )


def _load_package_texts() -> dict[str, list[str]]:
    try:
        package_path = Path(__file__).resolve().parent / "package.json"
        with open(package_path, "r") as json_file:
            return json.loads(json_file.read())
    except Exception:
        return {}



def _build_storyline(
    element_lines: list[str],
    package_texts: dict[str, list[str]],
    dominant_element: str | None = None,
) -> list[str]:
    storyline: list[str] = []

    for line in element_lines:
        if line:
            storyline.append(line)

    return storyline




def _hand_label(hand_event: HandEvent) -> str:
    if not hand_event.hand:
        return "unbekannte Hand"
    if hand_event.hand.value == "left":
        return "linke Hand"
    if hand_event.hand.value == "right":
        return "rechte Hand"
    return "unbekannte Hand"
