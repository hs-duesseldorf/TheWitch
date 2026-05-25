import json
from pathlib import Path
from typing import Dict, Any
import random

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]

input_average = {
    "palm_aspect_ratio": 0.48,
    "finger_length_ratio": 0.77,
    "index_to_ring_ratio": 0.49,
    "finger_profile": {
        "index": 0.67,
        "middle": 0.77,
        "ring": 0.68,
        "little": 0.53,
    },
}


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
    palm_aspect_ratio = features["palm_aspect_ratio"]
    finger_length_ratio = features["finger_length_ratio"]
    index_to_ring_ratio = features["index_to_ring_ratio"]
    finger_profile = features["finger_profile"]

    index_fp = finger_profile["index"]
    middle_fp = finger_profile["middle"]
    ring_fp = finger_profile["ring"]
    little_fp = finger_profile["little"]

    base_finger_weight = 1.0  # Vorher 1.5
    base_palm_weight = 0.5

    holz = (
                base_finger_weight * finger_length_ratio 
            + base_palm_weight * palm_aspect_ratio 
            + 1.0 * (index_to_ring_ratio ** 2) # Higher score if the index finger is longer
        )

    # 2. Feuer (Fire): Long palm, long fingers, ring finger dominance (Expressive and dynamic)
    feuer = (
            base_finger_weight * finger_length_ratio 
            + base_palm_weight * palm_aspect_ratio 
            + 1.0 * (1.0 / max(0.1, index_to_ring_ratio)) # Higher score if the ring finger is longer (lower index ratio)
    )

    # 3. Erde (Earth): Wide/short palm, short fingers, index finger dominance (Thick and solid shape)
    erde = (
          1.5 * (1.0 - palm_aspect_ratio) 
            + 1.5 * (1.0 - finger_length_ratio)  # Erhöht, um kurzen Fingern mehr Gewicht zu geben
            + 0.5 * (index_to_ring_ratio ** 2) # Higher score if the index finger is longer
    )

    # 4. Metall (Metal): Square palm (1:1), balanced finger lengths (Symmetry and order)
    # * Designed to score higher as values approach the balanced 1:1 ratio.
    palm_symmetry = 1.0 - abs(palm_aspect_ratio - 0.68) * 2.0
    finger_symmetry = 1.0 - abs(index_to_ring_ratio - 0.95) * 2.0
    metall = 1.2 * palm_symmetry + 1.2 * (finger_symmetry ** 2)

    # 5. Wasser (Water): long palm, long fingers, ring finger dominance (Fluid and adaptable)
    middle_dominance = middle_fp - (index_fp + ring_fp) / 2
    wasser = (
            base_finger_weight * finger_length_ratio 
            + base_palm_weight * palm_aspect_ratio 
            + 2.0 * middle_dominance  # Reagiert jetzt auf echte, variable Anatomie!
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

def find_core_element(current_features: Dict[str, Any]) -> str:
    # 1. Features des Durchschnitts extrahieren
    avg_features = extract_features(input_average)

    # 2. Raw Scores direkt berechnen (ohne build_result aufzurufen!)
    current_raw = compute_raw_scores(current_features)
    avg_raw = compute_raw_scores(avg_features)

    # 3. Scores z-standardisieren
    current_normalized = normalize_scores(current_raw)
    avg_normalized = normalize_scores(avg_raw)

    # 4. Hoechste positive Abweichung vom Durchschnitt finden
    element_differences = {
        element: current_normalized[element] - avg_normalized[element]
        for element in ELEMENTS
    }

    sorted_diffs = sorted(element_differences.items(), key=lambda item: item[1], reverse=True)
    top_element, top_diff = sorted_diffs[0]
    runner_diff = sorted_diffs[1][1] if len(sorted_diffs) > 1 else top_diff

    # Stabilitaet: nur wechseln, wenn der Vorsprung klar ist
    min_delta = 0.15
    if top_diff - runner_diff < min_delta:
        return max(current_normalized.items(), key=lambda item: item[1])[0]

    return top_element

def element_ratio(raw_scores: Dict[str, Any]) -> Dict[str, Any]:
    return {k: round((v / sum(raw_scores.values())) * 100, 2) for k, v in raw_scores.items()}



def build_result(input_payload: Dict[str, Any], average_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    normalized = normalize_input(input_payload)
    features = normalized["features"]
    meta = normalized["meta"]

    raw_scores = compute_raw_scores(features)
    normalized_scores = normalize_scores(raw_scores)
    final_states= determine_base_states(element_ratio(raw_scores))

    dominant_element = determine_dominant_element(normalized_scores)
    weakest_element = determine_weakest_element(normalized_scores)
    core_element = find_core_element(features)


    return {
        "request_id": meta.get("request_id"),
        "session_id": meta.get("session_id"),
        "handedness": meta.get("handedness"),
        "tracking_quality": meta.get("tracking_quality"),
        "trigger": meta.get("trigger"),
        "type": meta.get("type"),
        "element_scores_raw": {
            k: round2(v) for k, v in raw_scores.items()
        },
        "element_scores_normalized": {
            k: round2(v) for k, v in normalized_scores.items()
        },
        "element_ratio" : element_ratio(raw_scores),
        "element_states": final_states,
        "dominant_element": dominant_element,
        "weakest_element": weakest_element,
        "core_element": core_element,
    }

def GetLines(result: Dict[str, Any]) -> list[str]:
    lines_list = []

    try :
        package_path = Path(__file__).resolve().parent / "package.json"
        with open(package_path, "r") as json_file:
            lines = json.loads(json_file.read())
    except:
        print("the json file is not valid")
        return lines_list
    
    core_element = result.get("core_element")
    dominant_element = result.get("dominant_element")
    weakest_element = result.get("weakest_element")
    element_states = result.get("element_states", {})

    if core_element:
        core_line = lines.get("core_element", {}).get(core_element)
        if core_line:
            lines_list.append(core_line)
        else:
            print(f"DEBUG: core_line nicht gefunden fuer '{core_element}'")

    if core_element and dominant_element:
        dominant_line = lines.get("dominant_element", {}).get(core_element, {}).get(dominant_element)
        if dominant_line:
            lines_list.append(dominant_line)

    if core_element and weakest_element:
        state = element_states.get(weakest_element, "in_balance")
        if state == "in_balance":
            balanced_line = lines.get("balanced_element", {}).get(dominant_element, {}).get(weakest_element)
            if balanced_line:
                lines_list.append(balanced_line)
        else:
            weak_line = lines.get("weak_element", {}).get(dominant_element, {}).get(weakest_element)
            if weak_line:
                lines_list.append(weak_line)


    #advise
    dom_state = element_states.get(dominant_element)
    weak_state = element_states.get(weakest_element)
    
    if dom_state == "zu_stark":
        adv_line = lines.get("advise_strong", {}).get(dominant_element)
        if adv_line:
            lines_list.append(adv_line)
        else:
            print("the line is not available : adv_line : strong")

    elif weak_state == "zu_schwach":
        adv_line = lines.get("advise_weak", {}).get(weakest_element)
        if adv_line:
            lines_list.append(adv_line)
        else:
            print("the line is not available : adv_line : weak")
    
    else:
        adv_line = lines.get("advise_no_st", {}).get("advise_no_st")
        if adv_line:
            lines_list.append(adv_line)
        else:
            print("the line is not available : adv_line : no_st")

    return lines_list



if __name__ == "__main__":

    sample_input = {
        "request_id": "example-001",
        "session_id": "session-42",
        "handedness": "right",
        "tracking_quality": 0.93,
        "palm_aspect_ratio": 0.48,
        "finger_length_ratio": 0.77,
        "index_to_ring_ratio": 0.49,
        "finger_profile": {
            "index": 0.67,
            "middle": 0.77,
            "ring": 0.68,
            "little": 0.53
      }
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
        "pinky_length": 0.06959
    }
}




    result = build_result(sample2)
    Lines = GetLines(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    for line in Lines:
        print(line)