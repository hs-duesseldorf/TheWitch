import json
from pathlib import Path
from typing import Dict, Any
import random

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]


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
    palm_width = lengths.get("palm width", lengths.get("palm_width", 0.0))
    palm_height = lengths.get("palm_ height", lengths.get("palm height", lengths.get("palm_height", 0.0)))

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

    #std_dev_fin = np.std([index_fp, middle_fp, ring_fp, little_fp])

    #we have to make sure that every element is not symmetric easily. and every hand should be have various ratio....

    # Roh-Scores nach eurem konzeptionellen Mapping
    # 1. Holz (Wood): Long palm, long fingers, index finger dominance (Elongated and elegant shape)
    holz = (
            1.5 * finger_length_ratio  # Weight for long fingers
            + 1.0 * palm_aspect_ratio  # Weight for long palm
            + 0.5 * (index_to_ring_ratio ** 2)  # Higher score if the index finger is longer
    )

    # 2. Feuer (Fire): Long palm, long fingers, ring finger dominance (Expressive and dynamic)
    feuer = (
            1.5 * finger_length_ratio  # Weight for long fingers
            + 1.0 * palm_aspect_ratio  # Weight for long palm
            - 0.5 * (index_to_ring_ratio ** 2)  # Higher score if the ring finger is longer (lower index ratio)
    )

    # 3. Erde (Earth): Wide/short palm, short fingers, index finger dominance (Thick and solid shape)
    erde = (
            1.5 * (1.0 - palm_aspect_ratio)  # Higher score for a wider/shorter palm
            + 1.0 * (1.0 - finger_length_ratio)  # Higher score for shorter fingers
            + 0.5 * (index_to_ring_ratio ** 2)  # Higher score if the index finger is longer
    )

    # 4. Metall (Metal): Square palm (1:1), balanced finger lengths (Symmetry and order)
    # * Designed to score higher as values approach the balanced 1:1 ratio.
    metall = (
            1.5 * (1.0 - abs(palm_aspect_ratio - 0.5) * 2.0)  # Highest score when palm ratio is near 0.5 (1:1)
            + 1.0 * ((1.0 - abs(index_to_ring_ratio - 0.5) * 2.0) ** 2) # Highest score when index and ring fingers are similar in length
    )

    # 5. Wasser (Water): long palm, long fingers, ring finger dominance (Fluid and adaptable)
    wasser = (
            1.5 * finger_length_ratio  # Weight for long fingers
            + 1.0 * palm_aspect_ratio # weight for long palm
            + 1.0 * middle_fp  # Factor in middle finger characteristics/length
            - 0.5 * (index_to_ring_ratio ** 2)  # Higher score if the ring finger is longer
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

def build_result(input_payload: Dict[str, Any], average_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    normalized = normalize_input(input_payload)
    features = normalized["features"]
    meta = normalized["meta"]

    raw_scores = compute_raw_scores(features)
    normalized_scores = normalize_scores(raw_scores)
    final_states= determine_base_states(element_ratio(raw_scores))

    dominant_element = determine_dominant_element(normalized_scores)

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
        "core_element": find_core_element(input_payload)
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
    if core_element:
        core_line = lines.get("core_element", {}).get(core_element)
        if core_line:
            lines_list.append(core_line)

    for key, values in result.get("element_states", {}).items():
        one_line = lines.get("elemente", {}).get(key, {}).get(values)
        if one_line:
            lines_list.append(one_line[random.randint(0,2)])
        else:
            print("the line is not available")

    return lines_list


import numpy as np
import torch
import torch.nn as nn


class LegacyCompatibleMLP(nn.Module):
    def __init__(self, input_dim=7, num_classes=5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


MODEL_PATH = "hand_element_mode_87l.pth"
device = torch.device("cpu")
mlp_model = LegacyCompatibleMLP()
import os

if os.path.exists(MODEL_PATH):
    mlp_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    mlp_model.eval()
    print(f"[SUCCESS] hand_element_mode_87l.pth loaded inside find_core_element.")
else:
    print(f"[WARNING] {MODEL_PATH} not found. find_core_element will output 'NaN'.")

def find_core_element(input_payload: Dict[str, Any]) -> str:
    if not os.path.exists(MODEL_PATH):
        return "NaN"

    input_features = extract_features(input_payload)

    palm_aspect_ratio = input_features["palm_aspect_ratio"]
    finger_length_ratio = input_features["finger_length_ratio"]
    index_to_ring_ratio = input_features["index_to_ring_ratio"]

    index_fp = input_features["finger_profile"]["index"]
    middle_fp = input_features["finger_profile"]["middle"]
    ring_fp = input_features["finger_profile"]["ring"]
    little_fp = input_features["finger_profile"]["little"]

    feature_vector = [
        palm_aspect_ratio,
        finger_length_ratio,
        index_to_ring_ratio,
        index_fp,
        middle_fp,
        ring_fp,
        little_fp
    ]

    input_tensor = torch.tensor([feature_vector], dtype=torch.float32)

    with torch.no_grad():
        outputs = mlp_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]

    ELEMENT_MAP = ["holz", "feuer", "erde", "metall", "wasser"]

    best_idx = int(np.argmax(probabilities))
    return ELEMENT_MAP[best_idx]


if __name__ == "__main__":
    # sample_input = {
    #     "request_id": "example-001",
    #     "session_id": "session-42",
    #     "handedness": "right",
    #     "tracking_quality": 0.93,
    #     "palm_aspect_ratio": 0.30,
    #     "finger_length_ratio": 0.80,
    #     "index_to_ring_ratio": 0.40,
    #     "finger_profile": {
    #         "index": 0.70,
    #         "middle": 0.90,
    #         "ring": 0.75,
    #         "little": 0.50
    #     }
    # }

    sample_input = {
        "request_id": "example-001",
        "session_id": "session-42",
        "handedness": "right",
        "tracking_quality": 0.93,
        "lengths": {
            "palm_width": 0.064508,
            "palm_height": 0.107848,
            "thumb_length": 0.085996,
            "index_length": 0.074546,
            "middle_length": 0.095658,
            "ring_length": 0.089342,
            "pinky_length": 0.074921
        }
    }

    result = build_result(sample_input)
    Lines = GetLines(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    for line in Lines:
        print(line)