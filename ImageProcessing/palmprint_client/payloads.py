from __future__ import annotations

from typing import Any, Sequence

from .palmprint_data import build_palmprint_data

def build_feature_vector_message(
    *,
    status: str,
    hand_label: str,
    embedding_vector: Sequence[float],
    hand_proportions: dict[str, float],
) -> dict[str, Any]:
    return build_palmprint_data(
        status=status,
        hand=hand_label,
        proportions=hand_proportions,
        vector=embedding_vector,
    ).to_dict()


def build_status_message(
    *,
    status: str,
    message: str,
    hand_detected: bool,
    hand_label: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "message": message}
    if hand_detected and hand_label:
        payload["hand"] = hand_label
    return payload
