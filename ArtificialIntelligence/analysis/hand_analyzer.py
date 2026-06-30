from __future__ import annotations
import json
import logging
import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

from shared.events import HandEvent, Scene

logger = logging.getLogger("ai")

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]
ROOM_MAPPING = {j: [i for i in range(5) if j != i] for j in range(5)}
MODEL_DIR = "hand_analysis_models"

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

class SubWeakMLP(nn.Module):
    def __init__(self, input_dim=263, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return self.fc3(x)


rf_model_s1, xgb_model_s1 = None, None
rf_path = os.path.join(MODEL_DIR, "rf_stage1.pkl")
xgb_path = os.path.join(MODEL_DIR, "xgb_stage1.pkl")

if os.path.exists(rf_path) and os.path.exists(xgb_path):
    with open(rf_path, "rb") as f: rf_model_s1 = pickle.load(f)
    with open(xgb_path, "rb") as f: xgb_model_s1 = pickle.load(f)

weak_mlp_models = {}
for d_idx in range(5):
    path = os.path.join(MODEL_DIR, f"sub_mlp_{d_idx}.pth")
    if os.path.exists(path):
        model = SubWeakMLP()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        weak_mlp_models[d_idx] = model

def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den

def extract_features_tensor(lengths: Dict[str, float]) -> list:
    pw, ph = lengths.get("palm_width", 0.0), lengths.get("palm_height", 0.0)
    idx, mid, rng, pky = lengths.get("index_length", 0.0), lengths.get("middle_length", 0.0), lengths.get("ring_length", 0.0), lengths.get("pinky_length", 0.0)
    fingers = [idx, mid, rng, pky]
    max_f = max(fingers) if fingers else 0.0
    avg_f = sum(fingers) / len(fingers) if fingers else 0.0
    return [
        _safe_div(pw, ph), _safe_div(avg_f, ph), _safe_div(idx, rng),
        _safe_div(idx, max_f), _safe_div(mid, max_f), _safe_div(rng, max_f), _safe_div(pky, max_f)
    ]

def _prepare_input_np(hand_event: HandEvent) -> np.ndarray | None:
    lengths = hand_event.lengths or {}
    vector = hand_event.vector or []
    if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0 or len(vector) != 256:
        return None
    return np.array([extract_features_tensor(lengths) + vector], dtype=np.float32)

def _predict_dominant(input_np: np.ndarray) -> tuple[str, str, bool]:
    if rf_model_s1 is None or xgb_model_s1 is None:
        return "holz", "wasser", False

    rf_probs = rf_model_s1.predict_proba(input_np)[0]
    xgb_probs = xgb_model_s1.predict_proba(input_np)[0]
    avg_probs = (rf_probs + xgb_probs) / 2.0

    sorted_indices = np.argsort(avg_probs)[::-1]
    dominant_idx = sorted_indices[0]
    second_idx = sorted_indices[1]

    prob_diff = avg_probs[dominant_idx] - avg_probs[second_idx]
    is_border = prob_diff <= 0.05

    return ELEMENTS[dominant_idx], ELEMENTS[second_idx], is_border

def _predict_weak(input_np: np.ndarray, dominant_elem: str) -> str:
    if rf_model_s1 is None or xgb_model_s1 is None: return "feuer"
    pred_d_idx = ELEMENTS.index(dominant_elem)
    if pred_d_idx not in weak_mlp_models: return "feuer"

    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    with torch.no_grad():
        outputs = weak_mlp_models[pred_d_idx](input_tensor)
        pred_w_idx = int(torch.argmax(outputs, dim=1).item())
    return ELEMENTS[ROOM_MAPPING[pred_d_idx][pred_w_idx]]

def build_result(hand_event: HandEvent) -> Dict[str, Any]:
    input_np = _prepare_input_np(hand_event)

    dominant, second, is_border = _predict_dominant(input_np)
    weakest = _predict_weak(input_np, dominant)

    if second == weakest:
        remaining = [e for e in ELEMENTS if e != dominant and e != weakest]
        second = remaining[0]

    return {
        "dominant_element": dominant,
        "second_element": second,
        "weakest_element": weakest,
        "is_border_hand": is_border
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

    lookup_key = dominant
    shot_1 = content.get("shot_1") or {}

    if is_border and dominant and second:
        pair_a = f"{dominant}-{second}"
        pair_b = f"{second}-{dominant}"

        if pair_a in shot_1:
            lookup_key = pair_a
        elif pair_b in shot_1:
            lookup_key = pair_b

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
            result = build_result(hand_event)
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