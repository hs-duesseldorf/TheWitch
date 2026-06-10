import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from shared.events import HandEvent, Scene

ELEMENTS = ["holz", "feuer", "erde", "wasser"]
logger = logging.getLogger(__name__)
PACKAGE_PATH = Path(__file__).resolve().parent / "package.json"

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
_SCENE_VARIATION_SYSTEM_PROMPT = (
    "/no_think\n"
    "Antworte ausschließlich auf Deutsch.\n"
    "Gib nur die finale gesprochene Antwort aus: kein Markdown, keine Formatierung und keine Erklärungen.\n"
    "Du bist ein reiner Text-Transformator.\n"
    "Formuliere den Basetext als kurze, natürlich gesprochene Zeile einer weisen, düsteren Wahrsagerin um.\n"
    "Bewahre alle konkreten Anweisungen, Handlungen und Fakten.\n"
    "Erfinde keine neuen Informationen.\n"
    "Kein Markdown, keine Klammern, keine Emojis, keine Sternchen, keine Backticks.\n"
    "Ton: ruhig, präzise, leicht dunkel und klar sprechbar.\n"
    "Formuliere ausschließlich vollständige, natürlich klingende und grammatikalisch korrekte deutsche Sätze.\n"
    "Prüfe vor der Ausgabe Grammatik, Satzbau, Wortstellung, Bezüge und Zeichensetzung. Gib niemals holprige, mehrdeutige oder unvollständige Sätze aus.\n"
    "Variiere jede Antwort in Wortwahl, Rhythmus und Satzstruktur, aber niemals auf Kosten von korrektem, natürlichem Deutsch oder klarer Bedeutung.\n"
    "Vermeide Wiederholungen derselben Formulierung und nutze passende Synonyme.\n"
    "/no_think\n"
)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def round2(value: float) -> float:
    return round(value, 2)


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def extract_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "lengths" not in payload:
        return {
            "palm_aspect_ratio": payload["palm_aspect_ratio"],
            "finger_length_ratio": payload["finger_length_ratio"],
            "index_to_ring_ratio": payload["index_to_ring_ratio"],
            "finger_profile": payload["finger_profile"],
        }

    lengths = payload.get("lengths", {})
    palm_width = lengths.get("palm_width", lengths.get("palm_width", 0.0))
    palm_height = lengths.get("palm_height", lengths.get("palm_height", 0.0))

    index_length = lengths.get("index_length", 0.0)
    middle_length = lengths.get("middle_length", 0.0)
    ring_length = lengths.get("ring_length", 0.0)
    pinky_length = lengths.get("pinky_length", 0.0)

    finger_lengths = [index_length, middle_length, ring_length, pinky_length]
    max_finger = max(finger_lengths) if finger_lengths else 0.0
    avg_finger = sum(finger_lengths) / len(finger_lengths) if finger_lengths else 0.0

    return {
        "palm_aspect_ratio": safe_div(palm_width, palm_height),
        "finger_length_ratio": safe_div(avg_finger, palm_height),
        "index_to_ring_ratio": safe_div(index_length, ring_length),
        "finger_profile": {
            "index": safe_div(index_length, max_finger),
            "middle": safe_div(middle_length, max_finger),
            "ring": safe_div(ring_length, max_finger),
            "little": safe_div(pinky_length, max_finger),
        },
    }


def normalize_input(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    if "lengths" not in input_payload:
        return {
            "features": extract_features(input_payload),
            "meta": {
                "request_id": input_payload.get("request_id"),
                "session_id": input_payload.get("session_id"),
                "handedness": input_payload.get("handedness"),
                "tracking_quality": input_payload.get("tracking_quality"),
                "trigger": input_payload.get("trigger"),
                "type": input_payload.get("type"),
            },
        }

    meta = {
        "request_id": input_payload.get("request_id"),
        "session_id": input_payload.get("session_id"),
        "handedness": input_payload.get("hand", input_payload.get("handedness")),
        "tracking_quality": input_payload.get("tracking_quality"),
        "trigger": input_payload.get("trigger"),
        "type": input_payload.get("type"),
    }

    return {
        "features": extract_features(input_payload),
        "meta": meta,
    }


def compute_raw_scores(features: Dict[str, Any]) -> Dict[str, float]:
    palm_ratio = features["palm_aspect_ratio"]
    finger_ratio = features["finger_length_ratio"]

    palm_centered = (palm_ratio - 0.68) * 10.0
    finger_centered = (finger_ratio - 0.85) * 40.0

    base = 10.0

    # Erde: Quadratisch (+) und kurze Finger (-)
    erde = base + palm_centered - finger_centered

    # Feuer: Rechteckig (-) und kurze Finger (-)
    feuer = base - palm_centered - finger_centered

    # Holz: Quadratisch (+) und lange Finger (+)
    holz = base + palm_centered + finger_centered

    # Wasser: Rechteckig (-) und lange Finger (+)
    wasser = base - palm_centered + finger_centered

    return {
        "erde": erde,
        "feuer": feuer,
        "holz": holz,
        "wasser": wasser,
    }


def normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    values = list(raw_scores.values())
    mean = sum(values) / len(values)

    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5

    # Falls alle Werte fast gleich sind
    if std < 1e-6:
        return {k: 0.0 for k in raw_scores}

    normalized = {k: clamp((v - mean) / std, -2.0, 2.0) for k, v in raw_scores.items()}
    return normalized


def determine_base_states(scores: Dict[str, float]) -> Dict[str, str]:
    states = {}
    for element, score in scores.items():
        if score > 26:
            states[element] = "zu_stark"
        elif score < 14:
            states[element] = "zu_schwach"
        else:
            states[element] = "in_balance"
    return states


def determine_dominant_element(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0]


def determine_weakest_element(scores: Dict[str, float]) -> str:
    return min(scores.items(), key=lambda item: item[1])[0]


def element_ratio(raw_scores: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: round((v / sum(raw_scores.values())) * 100, 2) for k, v in raw_scores.items()
    }


def build_result(
    input_payload: Dict[str, Any], average_payload: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    normalized = normalize_input(input_payload)
    features = normalized["features"]
    meta = normalized["meta"]

    raw_scores = compute_raw_scores(features)
    normalized_scores = normalize_scores(raw_scores)
    final_states = determine_base_states(element_ratio(raw_scores))

    dominant_element = determine_dominant_element(normalized_scores)
    weakest_element = determine_weakest_element(normalized_scores)

    return {
        "request_id": meta.get("request_id"),
        "handedness": meta.get("handedness"),
        "element_scores_raw": {k: round2(v) for k, v in raw_scores.items()},
        "element_scores_normalized": {
            k: round2(v) for k, v in normalized_scores.items()
        },
        "element_ratio": element_ratio(raw_scores),
        "element_states": final_states,
        "dominant_element": dominant_element,
        "weakest_element": weakest_element,
    }


@lru_cache(maxsize=1)
def load_content_package() -> Dict[str, Any]:
    try:
        return json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("The content package is not valid: %s", PACKAGE_PATH)
        return {}


def GetLines(result: Dict[str, Any]) -> list[str]:
    lines_list = []
    package = load_content_package()
    lines = package.get("hand_analysis", package)

    dominant_element = result.get("dominant_element")
    weakest_element = result.get("weakest_element")

    if dominant_element:
        shot_1_line = lines.get("shot_1", {}).get(dominant_element)
        if shot_1_line:
            lines_list.append(shot_1_line)
        else:
            logger.debug("shot_1_line nicht gefunden fuer %r", dominant_element)

        shot_3_line = lines.get("shot_3", {}).get(dominant_element)
        if shot_3_line:
            lines_list.append(shot_3_line)
        else:
            logger.debug("shot_3_line nicht gefunden fuer %r", dominant_element)

        shot_4_line = lines.get("shot_4", {}).get(dominant_element)
        if shot_4_line:
            lines_list.append(shot_4_line)
        else:
            logger.debug("shot_4_line nicht gefunden fuer %r", dominant_element)

    if weakest_element:
        shot_5_line = lines.get("shot_5", {}).get(weakest_element)
        if shot_5_line:
            lines_list.append(shot_5_line)
        else:
            logger.debug("shot_5_line nicht gefunden fuer %r", weakest_element)
    return lines_list


def build_hand_analysis_prompt(hand_event: HandEvent | None) -> str:
    if not hand_event:
        return _HAND_ANALYSIS_SYSTEM_PROMPT + "\nKeine Handdaten verfuegbar."

    trigger = hand_event.trigger.value if hand_event.trigger else "unknown"
    hand = _hand_label(hand_event)
    lengths = hand_event.lengths

    data_desc = []
    if lengths:
        base_texts = _get_base_texts(trigger, hand_event, lengths)
        if base_texts:
            data_desc.append(base_texts)

    data_str = " ".join(data_desc) if data_desc else "Keine messbaren Daten"

    return (
        f"{_HAND_ANALYSIS_SYSTEM_PROMPT}\n"
        f"Beobachtung: {trigger}. Gesehene Hand: {hand}. {data_str}"
    )


def build_scene_prompt(
    scene: Scene | str,
    *,
    base_text: str,
    extra_context: str | None = None,
) -> str:
    scene_name = scene.value if isinstance(scene, Scene) else scene
    examples = get_scene_examples(scene)
    parts = [
        _SCENE_VARIATION_SYSTEM_PROMPT,
        f"Szene: {scene_name}.",
        "Hier ist dein verbindlicher Basetext. Variiere Tonfall, Rhythmus, Satzbau und Wortwahl deutlich.",
        "Die Bedeutung und jede konkrete Handlungsanweisung müssen erhalten bleiben.",
        "Die Neuformulierung muss natürlich klingen und aus grammatikalisch korrekten, vollständigen deutschen Sätzen bestehen.",
        "Schreibe lieber einfacher und korrekt als poetisch und fehlerhaft.",
        "Vermeide verschachtelte Sätze.",
        "Antworte mit genau einer einzigen fertigen Ausgabe.",
        "Gib keine Alternativen, keine Liste, keine Nummerierung und keine Erklärungen aus.",
        "Nutze die Beispiele nur als Stilvorbild. Die Beispiele selbst dürfen nicht wiedergegeben werden.",
        "Jede Neuformulierung muss sich deutlich von vorherigen unterscheiden. Vermeide identische Phrasen.",
        f"Basetext: {base_text.strip()}",
    ]
    
    if examples:
        parts.append(
            "Die folgenden Beispiele dienen ausschließlich als stilistische Orientierung."
            "Übernimm höchstens Tonfall, Rhythmus, Satzlänge und Atmosphäre."
        )
        for index, example in enumerate(examples, start=1):
            parts.append(f"Beispiel {index}: {example}")
    
    if extra_context:
        parts.append(f"Zusatzkontext: {extra_context.strip()}")
    return "\n".join(parts)


def get_scene_base_text(scene: Scene | str) -> str | None:
    scene_name = scene.value if isinstance(scene, Scene) else scene
    scene_entry = load_content_package().get("scenes", {}).get(scene_name)
    return _entry_base_text(scene_entry)


def _entry_base_text(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry.strip() or None
    
    if isinstance(entry, dict):
        value = entry.get("base_text")
        if isinstance(value, str):
            return value.strip() or None
    return None


def get_scene_examples(scene: Scene | str) -> list[str]:
    scene_name = scene.value if isinstance(scene, Scene) else scene
    scene_entry = load_content_package().get("scenes", {}).get(scene_name)

    if not isinstance(scene_entry, dict):
        return []

    examples = scene_entry.get("examples", [])
    if not isinstance(examples, list):
        return []

    return [
        item.strip()
        for item in examples
        if isinstance(item, str) and item.strip()
    ]


def analyze_hand_event(hand_event: HandEvent | None) -> dict[str, Any] | None:
    if not hand_event or not hand_event.lengths:
        return None

    try:
        return build_result(
            {
                "type": "hand",
                "trigger": hand_event.trigger.value
                if hand_event.trigger
                else "unknown",
                "hand": hand_event.hand.value if hand_event.hand else None,
                "lengths": hand_event.lengths,
            }
        )
    except Exception:
        return None


def _get_base_texts(
    trigger: str,
    hand_event: HandEvent,
    lengths: dict[str, float],
) -> str:
    payload = {
        "type": "hand",
        "trigger": trigger,
        "hand": hand_event.hand.value if hand_event.hand else None,
        "lengths": lengths,
    }

    try:
        result = build_result(payload)
        lines = GetLines(result)
    except Exception:
        return ""

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


def _build_analysis_transform_prompt(scene_name: str, base_text: str) -> str:
    return "\n".join(
        [
            _HAND_ANALYSIS_SYSTEM_PROMPT,
            f"Szene: {scene_name}.",
            "Hier ist dein verbindlicher Basetext. Übertrage ihn in einen neuen Satzbau mit anderer Wortwahl.",
            "Jeder Fakt und jeder genannte Elementname müssen erhalten bleiben.",
            "Formuliere natürliches, grammatikalisch korrektes Deutsch mit vollständigen und klar verständlichen Sätzen.",
            "Keine zwei Antworten dürfen gleich klingen. Variiere Phrasen, passende Synonyme und Satzstruktur, ohne die sprachliche Qualität zu verschlechtern.",
            f"Basetext: {base_text.strip()}",
        ]
    )


def _package_line(section: str, element: str | None) -> str:
    if not element:
        return ""
    package = load_content_package()
    return (
        package.get("hand_analysis", package).get(section, {}).get(element, "").strip()
    )


def _join_sentences(*parts: str) -> str:
    return " ".join(part for part in parts if part)


def _element_name(result: dict[str, Any], key: str) -> str | None:
    value = result.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def _hand_label(hand_event: HandEvent) -> str:
    if not hand_event.hand:
        return "unbekannte Hand"
    if hand_event.hand.value == "left":
        return "linke Hand"
    if hand_event.hand.value == "right":
        return "rechte Hand"
    return "unbekannte Hand"


def build_combined_analysis_prompt(
    hand_event: HandEvent | None,
) -> str | None:
    result = analyze_hand_event(hand_event)
    if result is None:
        return None

    dominant = _element_name(result, "dominant_element")
    weakest = _element_name(result, "weakest_element")
    if not dominant or not weakest:
        return None

    parts = []
    parts.append(
        f"Das staerkste Element deiner Hand ist {dominant}. "
        f"Das schwaechste Element ist {weakest}. "
        "Jetzt offenbart sich, was zwischen beiden in dir wirkt."
    )

    core = _join_sentences(
        _package_line("shot_1", dominant),
        _package_line("shot_3", dominant),
        _package_line("shot_4", dominant),
    )
    if core:
        parts.append(core)

    parts.append(
        f"Das schwaechste Element deiner Hand ist {weakest}. "
        "Gerade dort spuerst du, was dir fehlt und wonach dein Gleichgewicht verlangt."
    )

    advice = _package_line("shot_5", weakest)
    if advice:
        parts.append(advice)

    base_text = "\n\n".join(parts)
    return _build_analysis_transform_prompt(
        Scene.SCENE6.value,
        base_text,
    )


build_prompt = build_hand_analysis_prompt


if __name__ == "__main__":
    sample_input = {
        "request_id": "example-001",
        "session_id": "session-42",
        "handedness": "right",
        "tracking_quality": 0.93,
        "palm_aspect_ratio": 0.48,
        "finger_length_ratio": 0.77,
        "index_to_ring_ratio": 0.49,
        "finger_profile": {"index": 0.67, "middle": 0.77, "ring": 0.68, "little": 0.53},
    }

    sample2 = {
        "request_id": "person-1-left",
        "session_id": "session-42",
        "handedness": "left",
        "tracking_quality": 0.93,
        "lengths": {
            "palm_width": 0.064823,
            "palm_height": 0.09616,
            "thumb_length": 0.091336,
            "index_length": 0.084551,
            "middle_length": 0.097795,
            "ring_length": 0.086348,
            "pinky_length": 0.06959,
        },
    }

    result = build_result(sample_input)
    Lines = GetLines(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    for line in Lines:
        print(line)
