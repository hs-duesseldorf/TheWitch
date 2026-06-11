import json
import math
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from shared.events import HandEvent, Scene

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]
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

# Population statistics – calibrated against 895-sample dataset
# (dataset_with_vectors.json).
# mean = arithmetic mean of raw pentagon scores across all samples
# std  = population standard deviation of raw pentagon scores
# Run run_calibration() in test_consistency.py to recalibrate
RAW_SCORE_STATS: Dict[str, Dict[str, float]] = {
    "holz":   {"mean": 9.97978, "std": 0.475036},
    "feuer":   {"mean": 10.007811, "std": 0.437006},
    "erde":   {"mean": 10.007582, "std": 0.391533},
    "metall":   {"mean": 10.024907, "std": 0.457624},
    "wasser":   {"mean": 9.979921, "std": 0.40583},
}



# (palm_c, finger_c) form a 2D vector. Each element owns a 72° sector.
# Score = base + |v| * cos(angle(v) - element_angle)
PALM_CENTER   = 0.6524
FINGER_CENTER = 0.8286
PALM_SCALE    = 15.0
FINGER_SCALE  = 10.0

_ELEMENT_ANGLES: Dict[str, float] = {
    "holz":   math.radians(0),
    "wasser": math.radians(72),
    "feuer":  math.radians(144),
    "metall": math.radians(216),
    "erde":   math.radians(288),
}

# Gap between top-2 normalised scores below this → border hand
CONFIDENCE_THRESHOLD = 0.2

# Z-score thresholds for element state classification
STRONG_THRESHOLD =  1.0   # top ~16 %
WEAK_THRESHOLD   = -1.0   # bottom ~16 %



def round2(value: float) -> float:
    return round(value, 2)

def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def extract_features(payload: Dict[str, Any]) -> Dict[str, Any]:
    lengths       = payload["lengths"]
    palm_width    = lengths.get("palm_width", 0.0)
    palm_height   = lengths.get("palm_height", 0.0)
    index_length  = lengths.get("index_length", 0.0)
    middle_length = lengths.get("middle_length", 0.0)
    ring_length   = lengths.get("ring_length", 0.0)
    pinky_length  = lengths.get("pinky_length", 0.0)

    finger_lengths = [index_length, middle_length, ring_length, pinky_length]
    max_finger = max(finger_lengths) if any(finger_lengths) else 0.0
    avg_finger = sum(finger_lengths) / len(finger_lengths)

    return {
        "palm_aspect_ratio":   _safe_div(palm_width, palm_height),
        "finger_length_ratio": _safe_div(avg_finger, palm_height),
    }



def normalize_input(payload: Dict[str, Any]) -> Dict[str, Any]:
    req_id = str(payload.get("request_id", ""))
    session_id = req_id.split("-")[0].strip().upper() if "-" in req_id else req_id
    handedness = payload.get("handedness") or payload.get("hand")
    if not handedness and "-" in req_id:
        handedness = req_id.split("-")[-1].strip().lower()

    return {
        "features": extract_features(payload),
        "meta": {
            "request_id": req_id,
            "session_id": session_id,
            "handedness": handedness,
            "trigger":    payload.get("trigger"),
            "type":       payload.get("type"),
        },
    }




def compute_raw_scores(features: Dict[str, Any]) -> Dict[str, float]:
    """
    score each element as the dot-product of the measurement vector with the element's unit axis. 
    The element whose axis is closest in angle to the measurement wins its 72° sector.
    """
    palm_c   = (features["palm_aspect_ratio"]   - PALM_CENTER)   * PALM_SCALE
    finger_c = (features["finger_length_ratio"] - FINGER_CENTER) * FINGER_SCALE

    # hand vector
    hx, hy = palm_c, finger_c

    raw = {
        elem: 10.0 + hx * math.cos(elem_angle) + hy * math.sin(elem_angle)
        for elem, elem_angle in _ELEMENT_ANGLES.items()
    }
    return raw



def normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Z-score normalise against population statistics."""
    out: Dict[str, float] = {}
    for key, value in raw_scores.items():
        stats = RAW_SCORE_STATS[key]
        std   = max(stats["std"], 0.01)
        out[key] = max(-3.0, min(3.0, (value - stats["mean"]) / std))
    return out

def element_ratio(raw_scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw_scores.values())
    return {k: round(_safe_div(v, total) * 100, 2) for k, v in raw_scores.items()}


def determine_states(normalized_scores: Dict[str, float]) -> Dict[str, str]:
    states: Dict[str, str] = {}
    for element, z in normalized_scores.items():
        if z > STRONG_THRESHOLD:
            states[element] = "zu_stark"
        elif z < WEAK_THRESHOLD:
            states[element] = "zu_schwach"
        else:
            states[element] = "in_balance"
    return states


def compute_confidence(scores: Dict[str, float]) -> float:
    sorted_vals = sorted(scores.values(), reverse=True)
    return round(sorted_vals[0] - sorted_vals[1], 3)



def build_result(
    input_payload: Dict[str, Any],
) -> Dict[str, Any]:
    normalized        = normalize_input(input_payload)
    features          = normalized["features"]
    meta              = normalized["meta"]

    raw_scores        = compute_raw_scores(features)
    normalized_scores = normalize_scores(raw_scores)
    states            = determine_states(normalized_scores)

    sorted_elems      = sorted(normalized_scores, key=normalized_scores.__getitem__, reverse=True)
    dominant_element  = sorted_elems[0]
    second_element    = sorted_elems[1]
    weakest_element   = sorted_elems[-1]
    confidence        = compute_confidence(normalized_scores)
    is_border_hand    = confidence < CONFIDENCE_THRESHOLD

    return {
        "request_id":                meta.get("request_id"),
        "handedness":                meta.get("handedness"),
        "element_scores_raw":        {k: round2(v) for k, v in raw_scores.items()},
        "element_scores_normalized": {k: round2(v) for k, v in normalized_scores.items()},
        "element_ratio":             element_ratio(raw_scores),
        "element_states":            states,
        "dominant_element":          dominant_element,
        "second_element":            second_element,
        "weakest_element":           weakest_element,
        "confidence":                confidence,
        "is_border_hand":            is_border_hand,
    }



@lru_cache(maxsize=1)
def load_content_package() -> Dict[str, Any]:
    try:
        return json.loads(PACKAGE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("The content package is not valid: %s", PACKAGE_PATH)
        return {}

def GetLines(result: Dict[str, Any]) -> list[str]:
    lines_list: list[str] = []

    try:
        package_path = Path(__file__).resolve().parent / "package.json"
        with open(package_path, "r") as f:
            lines = json.loads(f.read())
    except Exception:
        print("the json file is not valid")
        return lines_list

    dominant    = result.get("dominant_element")
    second      = result.get("second_element")
    weakest     = result.get("weakest_element")
    is_border   = result.get("is_border_hand", False)

    if is_border and dominant and second:
        # build pair key — try both orderings, use whichever exists in package.json
        pair_a = f"{dominant}-{second}"
        pair_b = f"{second}-{dominant}"
        lookup_key = pair_a if pair_a in (lines.get("shot_1") or {}) else pair_b
    else:
        lookup_key = dominant

    if lookup_key:
        for shot_key in ("shot_1", "shot_3", "shot_4"):
            line = lines.get(shot_key, {}).get(lookup_key)
            if line:
                lines_list.append(line)
            else:
                print(f"DEBUG: {shot_key} not found for '{lookup_key}'")

    if weakest:
        line = lines.get("shot_5", {}).get(weakest)
        if line:
            lines_list.append(line)
        else:
            print(f"DEBUG: shot_5 not found for '{weakest}'")

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

def calibrate_stats(dataset: list, verbose: bool = True) -> Dict[str, Any]:
    import statistics as _stats

    palm_vals:   list[float] = []
    finger_vals: list[float] = []
    valid: list[Dict[str, Any]] = []

    for item in dataset:
        lengths = item.get("lengths", {})
        if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0:
            continue
        try:
            features = extract_features(item)
            if features["palm_aspect_ratio"] == 0 or features["finger_length_ratio"] == 0:
                continue
            palm_vals.append(features["palm_aspect_ratio"])
            finger_vals.append(features["finger_length_ratio"])
            valid.append(item)
        except Exception:
            continue

    if not palm_vals:
        print("No valid samples found.")
        return {}

    new_palm_center   = _stats.median(palm_vals)
    new_finger_center = _stats.median(finger_vals)

    if verbose:
        print(f"\n=== Feature centres (n={len(palm_vals)}) ===")
        print(f"  PALM_CENTER   = {new_palm_center:.4f}  (was {PALM_CENTER})")
        print(f"  FINGER_CENTER = {new_finger_center:.4f}  (was {FINGER_CENTER})")

    all_scores: Dict[str, list[float]] = {e: [] for e in ELEMENTS}
    for item in valid:
        features = extract_features(item)
        palm_c   = (features["palm_aspect_ratio"]   - new_palm_center)   * PALM_SCALE
        finger_c = (features["finger_length_ratio"] - new_finger_center) * FINGER_SCALE
        angle     = math.atan2(finger_c, palm_c)
        magnitude = math.hypot(palm_c, finger_c)
        for elem, elem_angle in _ELEMENT_ANGLES.items():
            all_scores[elem].append(10.0 + magnitude * math.cos(angle - elem_angle))

    new_stats: Dict[str, Dict[str, float]] = {}
    for e in ELEMENTS:
        vals = all_scores[e]
        mean = sum(vals) / len(vals)
        std  = _stats.pstdev(vals)
        new_stats[e] = {"mean": round(mean, 6), "std": round(std, 6)}

    if verbose:
        print(f"\n=== New RAW_SCORE_STATS ===")
        for e, s in new_stats.items():
            print(f'    "{e}":   {{"mean": {s["mean"]}, "std": {s["std"]}}},')

    from collections import Counter as _Counter
    dominance: Dict[str, int] = {e: 0 for e in ELEMENTS}
    confidences: list[float] = []
    for item in valid:
        features = extract_features(item)
        palm_c   = (features["palm_aspect_ratio"]   - new_palm_center)   * PALM_SCALE
        finger_c = (features["finger_length_ratio"] - new_finger_center) * FINGER_SCALE
        angle     = math.atan2(finger_c, palm_c)
        magnitude = math.hypot(palm_c, finger_c)
        raw  = {e: 10.0 + magnitude * math.cos(angle - a) for e, a in _ELEMENT_ANGLES.items()}
        norm = {k: (v - new_stats[k]["mean"]) / max(new_stats[k]["std"], 0.01) for k, v in raw.items()}
        dom  = max(norm, key=norm.__getitem__)
        dominance[dom] += 1
        sv = sorted(norm.values(), reverse=True)
        confidences.append(sv[0] - sv[1])

    total = sum(dominance.values()) or 1
    if verbose:
        print(f"\n=== Distribution (n={total}) ===")
        for e, cnt in sorted(dominance.items(), key=lambda x: -x[1]):
            print(f"  {e:<8} {cnt:>4} ({cnt/total*100:5.1f}%)")
        avg_conf = sum(confidences) / len(confidences)
        border   = sum(1 for c in confidences if c < CONFIDENCE_THRESHOLD)
        print(f"\n  Avg confidence:  {avg_conf:.3f}")
        print(f"  Border hands:    {border} ({border/total*100:.1f}%)")

    return {
        "palm_center":     new_palm_center,
        "finger_center":   new_finger_center,
        "raw_score_stats": new_stats,
    }


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

    result = build_result(sample2)
    Lines = GetLines(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    for line in Lines:
        print(line)

