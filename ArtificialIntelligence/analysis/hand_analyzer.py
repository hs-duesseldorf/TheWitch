from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from shared.events import HandEvent, Scene

logger = logging.getLogger("ai")

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]

PACKAGE_PATH = Path(__file__).resolve().parent.parent / "package.json"

_HAND_ANALYSIS_SYSTEM_PROMPT = (
    "/no_think\n"
    "Antworte ausschließlich auf Deutsch.\n"
    "Gib nur die finale gesprochene Antwort aus.\n"
    "Beginne sofort mit der Vorhersage, ohne Analyse oder Vorrede.\n"
    "STRIKTE REGEL: Du bist ein reiner Text-Transformator. Erfinde keine eigenen Geschichten, Linien oder Metaphern.\n"
    "Übertrage die bereitgestellten Basistexte vollständig und ohne Sinnveränderung in den Tonfall einer weisen, düsteren Wahrsagerin.\n"
    "Das in den Basistexten genannte dominante Element, zum Beispiel Holz, Feuer, Erde, Wasser oder Metall, muss namentlich und unmissverständlich ausgesprochen werden. Es darf niemals weggelassen oder durch ein anderes Wort ersetzt werden.\n"
    "Es geht immer um eine menschliche Hand, niemals um ein Handtuch.\n"
    "Verwende nie die Wörter Handtuch, Tuch oder Stoff.\n"
    "Kein Markdown, keine Klammern, keine Emojis.\n"
    "Ton: weise, leicht dunkel, konkret.\n"
    "Formuliere ausschließlich vollständige, natürlich klingende und grammatikalisch korrekte deutsche Sätze.\n"
    "Prüfe vor der Ausgabe Grammatik, Satzbau, Wortstellung, Bezüge und Zeichensetzung. Gib niemals holprige, mehrdeutige oder unvollständige Sätze aus.\n"
    "Variiere jede Antwort in Wortwahl, Rhythmus und Satzstruktur, aber niemals auf Kosten von korrektem, natürlichem Deutsch oder klarer Bedeutung.\n"
    "Vermeide Wiederholungen derselben Phrasen und nutze passende Synonyme.\n"
    "/no_think\n"
)

PALM_CENTER = 0.6524
FINGER_CENTER = 0.8286
PALM_SCALE = 15.0
FINGER_SCALE = 10.0

_ELEMENT_ANGLES: dict[str, float] = {
    "holz": math.radians(0),
    "wasser": math.radians(72),
    "feuer": math.radians(144),
    "metall": math.radians(216),
    "erde": math.radians(288),
}

RAW_SCORE_STATS: dict[str, dict[str, float]] = {
    "holz": {"mean": 9.97978, "std": 0.475036},
    "feuer": {"mean": 10.007811, "std": 0.437006},
    "erde": {"mean": 10.007582, "std": 0.391533},
    "metall": {"mean": 10.024907, "std": 0.457624},
    "wasser": {"mean": 9.979921, "std": 0.40583},
}

CONFIDENCE_THRESHOLD = 0.2
STRONG_THRESHOLD = 1.0
WEAK_THRESHOLD = -1.0


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def extract_features(lengths: dict[str, float]) -> dict[str, float]:
    palm_width = lengths.get("palm_width", 0.0)
    palm_height = lengths.get("palm_height", 0.0)
    finger_lengths = [
        lengths.get("index_length", 0.0),
        lengths.get("middle_length", 0.0),
        lengths.get("ring_length", 0.0),
        lengths.get("pinky_length", 0.0),
    ]
    avg_finger = sum(finger_lengths) / len(finger_lengths)
    return {
        "palm_aspect_ratio": _safe_div(palm_width, palm_height),
        "finger_length_ratio": _safe_div(avg_finger, palm_height),
    }


def compute_raw_scores(features: dict[str, float]) -> dict[str, float]:
    palm_c = (features["palm_aspect_ratio"] - PALM_CENTER) * PALM_SCALE
    finger_c = (features["finger_length_ratio"] - FINGER_CENTER) * FINGER_SCALE
    hx, hy = palm_c, finger_c
    return {
        elem: 10.0 + hx * math.cos(elem_angle) + hy * math.sin(elem_angle)
        for elem, elem_angle in _ELEMENT_ANGLES.items()
    }


def normalize_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in raw_scores.items():
        stats = RAW_SCORE_STATS[key]
        std = max(stats["std"], 0.01)
        out[key] = max(-3.0, min(3.0, (value - stats["mean"]) / std))
    return out


def determine_states(normalized_scores: dict[str, float]) -> dict[str, str]:
    states: dict[str, str] = {}
    for element, z in normalized_scores.items():
        if z > STRONG_THRESHOLD:
            states[element] = "zu_stark"
        elif z < WEAK_THRESHOLD:
            states[element] = "zu_schwach"
        else:
            states[element] = "in_balance"
    return states


def compute_confidence(scores: dict[str, float]) -> float:
    sorted_vals = sorted(scores.values(), reverse=True)
    return round(sorted_vals[0] - sorted_vals[1], 3)


def build_result(lengths: dict[str, float]) -> dict[str, Any]:
    features = extract_features(lengths)
    raw_scores = compute_raw_scores(features)
    normalized_scores = normalize_scores(raw_scores)
    states = determine_states(normalized_scores)
    sorted_elems = sorted(normalized_scores, key=normalized_scores.__getitem__, reverse=True)
    dominant_element = sorted_elems[0]
    second_element = sorted_elems[1]
    weakest_element = sorted_elems[-1]
    confidence = compute_confidence(normalized_scores)
    is_border_hand = confidence < CONFIDENCE_THRESHOLD
    return {
        "element_scores_raw": raw_scores,
        "element_scores_normalized": normalized_scores,
        "element_states": states,
        "dominant_element": dominant_element,
        "second_element": second_element,
        "weakest_element": weakest_element,
        "confidence": confidence,
        "is_border_hand": is_border_hand,
    }


@lru_cache(maxsize=1)
def _load_content() -> dict[str, Any]:
    try:
        package = json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("The content package is not valid: %s", PACKAGE_PATH)
        return {}
    return package.get("hand_analysis", package)


def _get_lines(result: dict[str, Any]) -> list[str]:
    lines_list: list[str] = []
    try:
        content = _load_content()
    except Exception:
        return lines_list

    dominant = result.get("dominant_element")
    second = result.get("second_element")
    weakest = result.get("weakest_element")
    is_border = result.get("is_border_hand", False)

    if is_border and dominant and second:
        pair_a = f"{dominant}-{second}"
        pair_b = f"{second}-{dominant}"
        lookup_key = pair_a if pair_a in (content.get("shot_1") or {}) else pair_b
    else:
        lookup_key = dominant

    if lookup_key:
        for shot_key in ("shot_1", "shot_3", "shot_4"):
            line = content.get(shot_key, {}).get(lookup_key)
            if line:
                lines_list.append(line)

    if weakest:
        line = content.get("shot_5", {}).get(weakest)
        if line:
            lines_list.append(line)

    return lines_list


def _get_base_texts(result: dict[str, Any]) -> str:
    lines = _get_lines(result)
    if not lines:
        return ""
    joined = " ".join(line for line in lines if line)
    return (
        "Hier sind deine verbindlichen Basistexte. Übersetze sie fließend in deinen "
        "mystischen Stil. WICHTIG: Das einzelne Element-Wort (wie 'Holz', 'Feuer' etc.) "
        "aus den Texten MUSS von dir als echtes, gesprochenes Wort in die Sätze eingebaut werden. "
        "Es darf absolut kein Fakt, kein Inhalt und vor allem kein Element-Name weggelassen, "
        "verändert oder verkürzt werden. "
        "Variiere Satzstruktur und Wortwahl deutlich, aber achte immer auf natürliches, grammatikalisch korrektes Deutsch. Jede Antwort muss eigenständig klingen: "
        f"{joined}"
    )


class HandAnalyzer:
    def build_prompt(self, hand_event: HandEvent | None) -> str | None:
        if not hand_event or not hand_event.lengths:
            return None

        try:
            result = build_result(hand_event.lengths)
        except Exception:
            logger.exception("Hand element scoring failed")
            return None

        base_texts = _get_base_texts(result)
        if not base_texts:
            return None

        trigger = hand_event.trigger.value if hand_event.trigger else "unknown"
        hand = self._hand_label(hand_event)

        return (
            f"{_HAND_ANALYSIS_SYSTEM_PROMPT}\n"
            f"Beobachtung: {trigger}. Gesehene Hand: {hand}. {base_texts}"
        )

    @staticmethod
    def _hand_label(hand_event: HandEvent) -> str:
        if not hand_event.hand:
            return "unbekannte Hand"
        if hand_event.hand.value == "left":
            return "linke Hand"
        if hand_event.hand.value == "right":
            return "rechte Hand"
        return "unbekannte Hand"
