from __future__ import annotations

from typing import Any, Sequence


def build_feature_vector_message(
    *,
    status: str,
    hand_label: str,
    hand_score: float,
    embedding_vector: Sequence[float],
    hand_proportions: dict[str, float],
) -> dict[str, Any]:
    return {
        "status": status,
        "hand": hand_label,
        "score": round(float(hand_score), 6),
        "proportions": {key: round(float(value), 6) for key, value in hand_proportions.items()},
        "vector": [round(float(value), 6) for value in embedding_vector],
    }


def build_status_message(
    *,
    status: str,
    message: str,
    hand_detected: bool,
    hand_label: str,
    hand_score: float = 0.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "message": message}
    if hand_detected and hand_label:
        payload["hand"] = hand_label
        payload["score"] = round(float(hand_score), 6)
    return payload
