import json
from typing import Dict, Any
import random

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

    #std_dev_fin = np.std([index_fp, middle_fp, ring_fp, little_fp])

    # Roh-Scores nach eurem konzeptionellen Mapping
    holz = (
            1.5 * finger_length_ratio
            + 1.0 * middle_fp
            + 0.5 * index_fp
            + 0.5 * palm_aspect_ratio
    )

    feuer = (
            1.2 * abs(index_to_ring_ratio - 0.5) * 2.0
            + 0.8 * finger_length_ratio
            + 0.5 * palm_aspect_ratio
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
    # holz = (
    #         (palm_aspect_ratio) * 25  # Longer palm
    #         #+ (1.0 - clamp(std_dev_fin * 10, 0, 1)) * 30  # Higher score as std approaches 0
    #         + (finger_length_ratio) * 25  # Longer fingers
    #         + (index_to_ring_ratio) * 20  # Index finger dominance (higher i_to_r)
    # )
    # feuer = (
    #         (palm_aspect_ratio) * 25  # Longer palm (vertically longer)
    #         #+ (std_dev_fin * 5) * 25  # Higher score for larger std
    #         + (finger_length_ratio) * 25  # Longer fingers
    #         + (0.5 - index_to_ring_ratio) * 25  # Ring finger dominance
    # )
    # erde = (
    #         (0.5 - palm_aspect_ratio) * 30  # Shorter palm (horizontally wider)
    #         + (1.0 - clamp(std_dev_fin * 5, 0, 1)) * 25  # Smaller std
    #         + (0.85 - finger_length_ratio) * 25  # Shorter fingers
    #         + (index_to_ring_ratio) * 20  # Index finger dominance
    # )
    # metall = (
    #         (0.5 - abs(0.5 - palm_aspect_ratio)) * 30  # Palm ratio 1:1 weight
    #         #+ (1.0 - clamp(std_dev_fin * 5, 0, 1)) * 30  # Higher score for smaller std
    #         + (1.0 - abs(0.75 - finger_length_ratio)) * 20  # Average finger length (0.75)
    #         + (0.5 - abs(0.5 - index_to_ring_ratio)) * 20  # Similar index/ring length (1.0)
    # )
    # wasser = (
    #         (0.5 - abs(0.45 - palm_aspect_ratio)) * 25  # Palm ratio near 1:1
    #         #+ (1.0 - abs(0.10 - std_dev_fin)) * 25  # Average std (0.10)
    #         + (finger_length_ratio) * 25  # Higher score for longer fingers
    #         + (0.5 - index_to_ring_ratio) * 25  # Ring finger dominance (lower i_to_r)
    # )

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
        if score > 35:
            states[element] = "zu_stark"
        elif score < 8:
            states[element] = "zu_schwach"
        else:
            states[element] = "in_balance"
    return states

def determine_dominant_element(scores: Dict[str, float]) -> str:
    return max(scores.items(), key=lambda item: item[1])[0]

def element_ratio(raw_scores: Dict[str, Any]) -> Dict[str, Any]:
    return {k: round((v / sum(raw_scores.values())) * 100, 2) for k, v in raw_scores.items()}

def build_result(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    features = {
        "palm_aspect_ratio": input_payload["palm_aspect_ratio"],
        "finger_length_ratio": input_payload["finger_length_ratio"],
        "index_to_ring_ratio": input_payload["index_to_ring_ratio"],
        "finger_profile": input_payload["finger_profile"],
    }

    raw_scores = compute_raw_scores(features)
    normalized_scores = normalize_scores(raw_scores)
    final_states= determine_base_states(element_ratio(raw_scores))

    dominant_element = determine_dominant_element(normalized_scores)

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
        "element_ratio" : element_ratio(raw_scores),
        "element_states": final_states,
        "dominant_element": dominant_element,
        "core_element" : find_core_element(input_average, sample_input)
    }

def GetLines(result: Dict[str, Any]) -> list[str]:
    lines_list = []

    try :
        with open("package.json", "r") as json_file:
            lines = json.loads(json_file.read())
    except:
        print("the json file is not valid")
        return lines_list
    lines_list.append(lines.get("core_element").get(result["core_element"]))

    for key, values in result["element_states"].items():
        one_line = lines.get("elemente").get(key).get(values)
        if not one_line==None :
            lines_list.append(one_line[random.randint(0,2)])
        else :
            print("the line is not available")

    return lines_list

def find_core_element(average : Dict[str, Any], input: Dict[str, Any]) :
    features = {
        "palm_aspect_ratio": abs(average["palm_aspect_ratio"] - input["palm_aspect_ratio"]),
        "finger_length_ratio": abs(average["finger_length_ratio"] - input["finger_length_ratio"]),
        "index_to_ring_ratio": abs(average["index_to_ring_ratio"] - input["index_to_ring_ratio"]),
        "index": abs(average["finger_profile"]["index"] - input["finger_profile"]["index"]),
        "middle": abs(average["finger_profile"]["middle"] - input["finger_profile"]["middle"]),
        "ring": abs(average["finger_profile"]["ring"] - input["finger_profile"]["ring"]),
        "little": abs(average["finger_profile"]["little"] - input["finger_profile"]["little"])
    }

    best_feature = max(features, key=features.get)

    # print(features)
    # print(best_feature)

    match best_feature:
        case "palm_aspect_ratio":
            return "earth"

        case "finger_length_ratio":
            return "water"

        case "index_to_ring_ratio":
            return "fire"

        case "index" | "middle":
            return "wood"

        case "ring" | "little":
            return "metal"

        case _:
            print("best feature is not valid")
            return "NaN"


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

    # sample_input = {
    #     "request_id": "example-001",
    #     "session_id": "session-42",
    #     "handedness": "right",
    #     "tracking_quality": 0.93,
    #     "palm_aspect_ratio": 0.51,
    #     "finger_length_ratio": 0.77,
    #     "index_to_ring_ratio": 0.51,
    #     "finger_profile": {
    #         "index": 0.67,
    #         "middle": 0.77,
    #         "ring": 0.68,
    #         "little": 0.53
    #     }
    # }

    input_average = {
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

    result = build_result(sample_input)
    Lines = GetLines(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    for line in Lines:
        print(line)
