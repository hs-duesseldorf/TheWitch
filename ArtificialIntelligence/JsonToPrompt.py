import json
from typing import Dict, Any

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def round2(value: float) -> float:
    return round(value, 2)


def compute_raw_scores(features: Dict[str, Any]) -> Dict[str, float]:
    palm_aspect_ratio = features["palm_aspect_ratio"]
    finger_length_ratio = features["finger_length_ratio"]
    index_to_ring_ratio = features["index_to_ring_ratio"]
    finger_profile = features["finger_profile"]

    index_fp = finger_profile["index"]
    middle_fp = finger_profile["middle"]
    ring_fp = finger_profile["ring"]
    little_fp = finger_profile["little"]

    # Roh-Scores nach eurem konzeptionellen Mapping
    holz = (
            1.5 * finger_length_ratio
            + 1.0 * middle_fp
            + 0.5 * index_fp
            - 0.5 * palm_aspect_ratio
    )

    feuer = (
            1.2 * abs(index_to_ring_ratio - 0.5) * 2.0
            + 0.8 * finger_length_ratio
            - 0.5 * palm_aspect_ratio
    )

    erde = (
            1.5 * (1.0 - palm_aspect_ratio)
            + 0.5 * (1.0 - finger_length_ratio)
    )

    metall = (
            1.2 * (1.0 - abs(index_to_ring_ratio - 0.5) * 2.0)
            + 0.8 * palm_aspect_ratio
    )

    wasser = (
            1.0 * little_fp
            + 1.0 * palm_aspect_ratio
            - 0.5 * finger_length_ratio
    )

    return {
        "holz": holz,
        "feuer": feuer,
        "erde": erde,
        "metall": metall,
        "wasser": wasser,
    }


def normalize_scores(raw_scores: Dict[str, float]) -> Dict[str, float]:
    values = list(raw_scores.values())
    mean = sum(values) / len(values)

    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5

    # Falls alle Werte fast gleich sind
    if std < 1e-6:
        return {k: 0.0 for k in raw_scores}

    normalized = {
        k: clamp((v - mean) / std, -2.0, 2.0)
        for k, v in raw_scores.items()
    }
    return normalized


def determine_base_states(scores: Dict[str, float]) -> Dict[str, str]:
    states = {}
    for element, score in scores.items():
        if score > 1.0:
            states[element] = "zu_stark"
        elif score < -1.0:
            states[element] = "zu_schwach"
        elif -0.5 <= score <= 0.5:
            states[element] = "in_balance"
        else:
            states[element] = "neutral"
    return states


def apply_blockages(
        scores: Dict[str, float],
        states: Dict[str, str]
) -> Dict[str, str]:
    result = states.copy()

    # Gegensätzliche Elemente gleichzeitig stark -> Blockade
    if scores["holz"] > 1.0 and scores["metall"] > 1.0:
        if abs(scores["holz"] - scores["metall"]) < 0.75:
            result["holz"] = "blockiert"
            result["metall"] = "blockiert"

    if scores["wasser"] > 1.0 and scores["feuer"] > 1.0:
        if abs(scores["wasser"] - scores["feuer"]) < 0.75:
            result["wasser"] = "blockiert"
            result["feuer"] = "blockiert"

    # Optional: Erde kann Holz "festhalten", wenn beide stark sind
    if scores["erde"] > 1.0 and scores["holz"] > 1.0:
        if abs(scores["erde"] - scores["holz"]) < 0.5 and result["holz"] != "blockiert":
            result["holz"] = "blockiert"

    return result


def determine_dominant_element(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0]


def build_result(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    features = {
        "palm_aspect_ratio": input_payload["palm_aspect_ratio"],
        "finger_length_ratio": input_payload["finger_length_ratio"],
        "index_to_ring_ratio": input_payload["index_to_ring_ratio"],
        "finger_profile": input_payload["finger_profile"],
    }

    raw_scores = compute_raw_scores(features)
    normalized_scores = normalize_scores(raw_scores)
    base_states = determine_base_states(normalized_scores)
    final_states = apply_blockages(normalized_scores, base_states)
    dominant_element = determine_dominant_element(normalized_scores)

    blocked_elements = [
        element for element, state in final_states.items()
        if state == "blockiert"
    ]

    return {
        "request_id": input_payload.get("request_id"),
        "session_id": input_payload.get("session_id"),
        "handedness": input_payload.get("handedness"),
        "tracking_quality": input_payload.get("tracking_quality"),
        "element_scores_raw": {
            k: round2(v) for k, v in raw_scores.items()
        },
        "element_scores_normalized": {
            k: round2(v) for k, v in normalized_scores.items()
        },
        "element_states": final_states,
        "dominant_element": dominant_element,
        "blocked_elements": blocked_elements
    }


if __name__ == "__main__":
    sample_input = {
        "request_id": "example-001",
        "session_id": "session-42",
        "handedness": "right",
        "tracking_quality": 0.93,
        "palm_aspect_ratio": 0.30,
        "finger_length_ratio": 0.80,
        "index_to_ring_ratio": 0.40,
        "finger_profile": {
            "index": 0.70,
            "middle": 0.90,
            "ring": 0.75,
            "little": 0.50
        }
    }

    result = build_result(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
