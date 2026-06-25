import json
import os
import pickle
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn

ELEMENTS = ["holz", "feuer", "erde", "metall", "wasser"]
ROOM_MAPPING = {d: [w for w in range(5) if d != w] for d in range(5)}
MODEL_DIR = "hand_analysis_models"


class SubWeakMLP(nn.Module):
    def __init__(self, input_dim=263, num_classes=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x): return self.network(x)

with open(os.path.join(MODEL_DIR, "rf_stage1.pkl"), "rb") as f: rf_model_s1 = pickle.load(f)
with open(os.path.join(MODEL_DIR, "xgb_stage1.pkl"), "rb") as f: xgb_model_s1 = pickle.load(f)

weak_mlp_models = {}
for d_idx in range(5):
    path = os.path.join(MODEL_DIR, f"sub_mlp_{d_idx}.pth")
    if os.path.exists(path):
        model = SubWeakMLP()
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        weak_mlp_models[d_idx] = model


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0 else 0.0

def extract_features_tensor(lengths: Dict[str, float]) -> list:
    pw, ph = lengths.get("palm_width", 0.0), lengths.get("palm_height", 0.0)
    idx, mid, rng, pky = lengths.get("index_length", 0.0), lengths.get("middle_length", 0.0), lengths.get("ring_length",
                                                                                                          0.0), lengths.get(
        "pinky_length", 0.0)

    fingers = [idx, mid, rng, pky]
    max_f = max(fingers) if fingers else 0.0
    avg_f = sum(fingers) / len(fingers) if fingers else 0.0

    return [
        safe_div(pw, ph), safe_div(avg_f, ph), safe_div(idx, rng),
        safe_div(idx, max_f), safe_div(mid, max_f), safe_div(rng, max_f), safe_div(pky, max_f)
    ]

def predict_dominant_element(input_payload: Dict[str, Any]) -> str:
    lengths = input_payload.get("lengths", {})
    vector = input_payload.get("vector", [])

    if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0 or len(vector) != 256:
        return "NaN"
    input_np = np.array([extract_features_tensor(lengths) + vector], dtype=np.float32)

    rf_probs = rf_model_s1.predict_proba(input_np)[0]
    xgb_probs = xgb_model_s1.predict_proba(input_np)[0]
    best_idx = int(np.argmax((rf_probs + xgb_probs) / 2.0))
    return ELEMENTS[best_idx]


def predict_weak_element(input_payload: Dict[str, Any]) -> str:
    lengths = input_payload.get("lengths", {})
    vector = input_payload.get("vector", [])

    if not lengths or lengths.get("middle_length", 0) == 0 or lengths.get("palm_height", 0) == 0 or len(vector) != 256:
        return "NaN"
    input_np = np.array([extract_features_tensor(lengths) + vector], dtype=np.float32)

    rf_probs = rf_model_s1.predict_proba(input_np)[0]
    xgb_probs = xgb_model_s1.predict_proba(input_np)[0]
    pred_d_idx = int(np.argmax((rf_probs + xgb_probs) / 2.0))

    if pred_d_idx not in weak_mlp_models: return "NaN"

    input_tensor = torch.tensor(input_np, dtype=torch.float32)
    with torch.no_grad():
        outputs = weak_mlp_models[pred_d_idx](input_tensor)
        pred_w_idx = int(torch.argmax(outputs, dim=1).item())

    return ELEMENTS[ROOM_MAPPING[pred_d_idx][pred_w_idx]]

def build_result(input_payload: Dict[str, Any], average_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "request_id": input_payload.get("request_id"),
        "hand": input_payload.get("hand"),
        "trigger": input_payload.get("trigger"),
        "type": input_payload.get("type"),
        "dominant_element": predict_dominant_element( input_payload),
        "weak_element": predict_weak_element(input_payload)
    }

if __name__ == "__main__":
    sample_input = {
        "request_id": "P155-right",
        "type": "hand",
        "trigger": "hand_detected",
        "hand": "right",
        "lengths": {
            "palm_width": 0.063148,
            "palm_height": 0.096668,
            "thumb_length": 0.087722,
            "index_length": 0.078355,
            "middle_length": 0.089251,
            "ring_length": 0.087197,
            "pinky_length": 0.067574
        },
        "vector": [
            -0.041464,
            0.02024,
            -0.023007,
            -0.009356,
            0.025619,
            -0.032627,
            -0.047302,
            -0.071416,
            0.062909,
            -0.002386,
            -0.074268,
            0.058652,
            -0.147777,
            0.033358,
            0.032633,
            0.031059,
            0.09131,
            0.035888,
            -0.00321,
            0.065617,
            0.034424,
            0.053123,
            0.126026,
            0.025639,
            0.024122,
            -0.066781,
            -0.000872,
            -0.099697,
            0.109671,
            -0.01755,
            0.094638,
            -0.011851,
            -0.059797,
            -0.043913,
            -0.048586,
            0.00593,
            0.015137,
            0.14131,
            -0.024491,
            -0.017433,
            0.029154,
            0.077841,
            0.125143,
            0.014694,
            -0.049515,
            -0.111571,
            -0.086716,
            0.018549,
            -0.031411,
            -0.077356,
            0.089583,
            -0.003958,
            0.003561,
            -0.026276,
            0.012196,
            0.125861,
            -0.051702,
            -0.198081,
            0.063342,
            -0.035066,
            0.055077,
            0.000299,
            -0.002105,
            -0.003933,
            0.04282,
            0.063577,
            0.089213,
            0.041492,
            0.056119,
            0.004819,
            0.081111,
            0.082207,
            0.046212,
            -0.05511,
            0.065093,
            0.01495,
            0.043967,
            0.117488,
            0.021559,
            -0.075625,
            -0.077437,
            0.020886,
            -0.058472,
            -0.251447,
            -0.049351,
            -0.066851,
            0.052053,
            0.048074,
            0.086428,
            -0.091448,
            0.009017,
            0.021672,
            -0.002377,
            0.087679,
            0.10915,
            0.022105,
            -0.079569,
            0.019107,
            -0.135485,
            0.022367,
            -0.035843,
            0.076713,
            -0.023308,
            0.094404,
            0.022674,
            -0.001187,
            -0.003049,
            -0.00176,
            0.055653,
            -0.008948,
            0.006506,
            -0.075602,
            -0.01515,
            -0.017497,
            0.05303,
            -0.069652,
            -0.018975,
            -0.02917,
            0.014675,
            -0.088061,
            -0.06489,
            -0.098792,
            0.008851,
            0.001903,
            4e-05,
            -0.028193,
            -0.008332,
            -0.164123,
            0.001445,
            -0.01817,
            0.11758,
            -0.004051,
            -0.076563,
            0.072107,
            -0.052713,
            0.020908,
            0.154041,
            0.06192,
            0.029421,
            0.0134,
            0.028679,
            0.046733,
            -0.006299,
            -0.078798,
            0.070199,
            0.043968,
            0.060888,
            -0.088791,
            0.067006,
            -0.016887,
            0.052745,
            0.102848,
            -0.049934,
            -0.027437,
            0.041815,
            -0.054216,
            0.022727,
            0.08692,
            -0.036155,
            -0.027726,
            0.084524,
            7.4e-05,
            0.025057,
            0.03619,
            -0.033803,
            -0.020411,
            0.027349,
            -0.001094,
            -0.063221,
            0.011315,
            -0.019962,
            -0.001081,
            0.016183,
            0.048137,
            0.012821,
            -0.000799,
            -0.058041,
            0.044288,
            -0.013248,
            -9e-06,
            0.013777,
            0.085252,
            -0.054535,
            0.001557,
            -0.069287,
            0.052868,
            -0.078544,
            0.043124,
            -0.129217,
            0.044919,
            0.004199,
            0.050488,
            0.057279,
            -0.0344,
            -0.033843,
            -0.005742,
            -0.025504,
            0.062331,
            0.010912,
            -0.057477,
            -0.146641,
            -0.053293,
            -0.107782,
            -0.014687,
            -0.074924,
            -0.034872,
            -0.033999,
            -0.032543,
            -0.014508,
            -0.084503,
            -0.011836,
            -0.039952,
            0.000275,
            -0.004519,
            -0.010402,
            0.010626,
            0.027676,
            0.138218,
            0.042092,
            0.015115,
            -0.051574,
            -0.021177,
            0.035486,
            -0.116531,
            -0.020355,
            -0.049736,
            -0.024418,
            -0.026153,
            0.033475,
            0.018833,
            -0.044517,
            -0.065455,
            -0.050804,
            0.016146,
            0.030783,
            -0.024069,
            0.114947,
            -0.070577,
            -0.08751,
            0.100987,
            0.040393,
            0.09168,
            0.032677,
            -0.051861,
            -0.024581,
            0.020698,
            -0.075739,
            -0.115482,
            0.060337,
            -0.056945,
            -0.075696,
            -0.122535,
            0.086591,
            -0.005213,
            0.048198,
            0.052905
        ]
    }

    result = build_result(sample_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))
