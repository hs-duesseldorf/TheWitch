import json
import math
from pathlib import Path
from typing import Dict, Any

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]


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


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _round2(value: float) -> float:
    return round(value, 2)

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
        "element_scores_raw":        {k: _round2(v) for k, v in raw_scores.items()},
        "element_scores_normalized": {k: _round2(v) for k, v in normalized_scores.items()},
        "element_ratio":             element_ratio(raw_scores),
        "element_states":            states,
        "dominant_element":          dominant_element,
        "second_element":            second_element,
        "weakest_element":           weakest_element,
        "confidence":                confidence,
        "is_border_hand":            is_border_hand,
    }


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
    sample = {
        "request_id": "P1-left",
        "lengths": {
            "palm_width": 0.064823, "palm_height": 0.09616,
            "thumb_length": 0.091336, "index_length": 0.084551,
            "middle_length": 0.097795, "ring_length": 0.086348,
            "pinky_length": 0.06959,
        },
    }
    result = build_result(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
