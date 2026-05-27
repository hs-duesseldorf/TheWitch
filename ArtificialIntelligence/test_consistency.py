import json
from hand_analysis import build_result

# =====================================================================
# 데이터셋 (오타 및 수치 오류 완벽 교정 완료)
# =====================================================================

# Person 1 (Linke Hand)
person_1_input = {
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

person_1_1_input = {
    "request_id": "person-1-1-left",
    "session_id": "session-42",
    "handedness": "left",
    "lengths": {
        "palm_width": 0.066162,
        "palm_height": 0.102347,
        "thumb_length": 0.090121,
        "index_length": 0.083411,
        "middle_length": 0.096438,
        "ring_length": 0.084703,
        "pinky_length": 0.067061
    }
}

# Person 2 (Rechte Hand)
person_2_input = {
    "request_id": "person-2-right",
    "session_id": "session-42",
    "handedness": "right",
    "tracking_quality": 0.93,
    "lengths": {
        "palm_width": 0.065816,
        "palm_height": 0.096213,
        "thumb_length": 0.09111,
        "index_length": 0.086843,
        "middle_length": 0.101181,
        "ring_length": 0.090138,
        "pinky_length": 0.07071
    }
}

person_2_1_input = {
    "request_id": "person-2-1-right",
    "session_id": "session-42",
    "handedness": "left",
    "lengths": {
        "palm_width": 0.067653,
        "palm_height": 0.094835,
        "thumb_length": 0.097902,
        "index_length": 0.083484,
        "middle_length": 0.095678,
        "ring_length": 0.08343,
        "pinky_length": 0.06309
    }
}

person_2_2_input = {
    "request_id": "person-2-2-right",
    "session_id": "session-42",
    "handedness": "left",
    "lengths": {
        "palm_width": 0.0623,
        "palm_height": 0.099003,
        "thumb_length": 0.093183,
        "index_length": 0.079928,
        "middle_length": 0.093395,
        "ring_length": 0.081941,
        "pinky_length": 0.068093
    }
}

person_3_input = {
    "request_id": "person-3-right",
    "session_id": "session-42",
    "handedness": "right",
    "lengths": {
        "palm_width": 0.07003,
        "palm_height": 0.096085,
        "thumb_length": 0.09391,
        "index_length": 0.08446,
        "middle_length": 0.100144,
        "ring_length": 0.088793,
        "pinky_length": 0.065955
    }
}

person_3_1_input = {
    "request_id": "person-3-1-right",
    "session_id": "session-42",
    "lengths": {
        "palm_width": 0.070793,
        "palm_height": 0.100699,
        "thumb_length": 0.102591,
        "index_length": 0.081032,
        "middle_length": 0.09652,
        "ring_length": 0.083566,
        "pinky_length": 0.067484
    }
}

person_4_input = {
    "request_id": "person-4-left",
    "session_id": "session-42",
    "lengths": {
        "palm_width": 0.06682,
        "palm_height": 0.100714,
        "thumb_length": 0.097273,
        "index_length": 0.081936,
        "middle_length": 0.097627,
        "ring_length": 0.090127,
        "pinky_length": 0.072404
    }
}

person_5_input = {
    "request_id": "person-5-left",
    "session_id": "session-42",
    "lengths": {
        "palm_width": 0.059105,
        "palm_height": 0.089929,
        "thumb_length": 0.100096,
        "index_length": 0.079477,
        "middle_length": 0.093064,
        "ring_length": 0.081487,
        "pinky_length": 0.065981
    }
}


def test_two_different_hands():
    result_p1 = build_result(person_1_input)
    result_p1_1 = build_result(person_1_1_input)
    result_p2 = build_result(person_2_input)
    result_p2_1 = build_result(person_2_1_input)
    result_p2_2 = build_result(person_2_2_input)
    result_p3 = build_result(person_3_input)
    result_p3_1 = build_result(person_3_1_input)
    result_p4 = build_result(person_4_input)
    result_p5 = build_result(person_5_input)

    print("--- ERGEBNIS PERSON 1 (Links) ---")
    print(f"core element: {str(result_p1['core_element']).upper()}")
    print(f"dominant: {str(result_p1['dominant_element']).upper()}")
    print("Zustände:           ", result_p1["element_states"])
    print("Prozentuale Ratios: ", result_p1["element_ratio"])

    print("\n--- ERGEBNIS PERSON 1.1 (Links) ---")
    print(f"core element: {str(result_p1_1['core_element']).upper()}")
    print(f"dominant: {str(result_p1_1['dominant_element']).upper()}")
    print("Zustände:           ", result_p1_1["element_states"])
    print("Prozentuale Ratios: ", result_p1_1["element_ratio"])

    print("\n--- ERGEBNIS PERSON 2 (Rechts) ---")
    print(f"core element: {str(result_p2['core_element']).upper()}")
    print(f"dominant: {str(result_p2['dominant_element']).upper()}")
    print("Zustände:           ", result_p2["element_states"])
    print("Prozentuale Ratios: ", result_p2["element_ratio"])

    print("\n--- ERGEBNIS PERSON 2.1 (Links) ---")
    print(f"core element: {str(result_p2_1['core_element']).upper()}")
    print(f"dominant: {str(result_p2_1['dominant_element']).upper()}")
    print("Zustände:           ", result_p2_1["element_states"])
    print("Prozentuale Ratios: ", result_p2_1["element_ratio"])

    print("\n--- ERGEBNIS PERSON 2.2 (Links) ---")
    print(f"core element: {str(result_p2_2['core_element']).upper()}")
    print(f"dominant: {str(result_p2_2['dominant_element'].upper())}")
    print("Zustände:           ", result_p2_2["element_states"])
    print("Prozentuale Ratios: ", result_p2_2["element_ratio"])

    print("\n--- ERGEBNIS PERSON 3 (Rechts) ---")
    print(f"core element: {str(result_p3['core_element']).upper()}")
    print(f"dominant: {str(result_p3['dominant_element']).upper()}")
    print("Zustände:           ", result_p3["element_states"])
    print("Prozentuale Ratios: ", result_p3["element_ratio"])

    print("\n--- ERGEBNIS PERSON 3.1 (Rechts) ---")
    print(f"core element: {str(result_p3_1['core_element']).upper()}")
    print(f"dominant: {str(result_p3_1['dominant_element']).upper()}")
    print("Zustände:           ", result_p3_1["element_states"])
    print("Prozentuale Ratios: ", result_p3_1["element_ratio"])

    print("\n--- ERGEBNIS PERSON 4 (Links) ---")
    print(f"core element: {str(result_p4['core_element']).upper()}")
    print(f"dominant: {str(result_p4['dominant_element']).upper()}")
    print("Zustände:           ", result_p4["element_states"])
    print("Prozentuale Ratios: ", result_p4["element_ratio"])

    print("\n--- ERGEBNIS PERSON 5 (Links) ---")
    print(f"core element: {str(result_p5['core_element']).upper()}")
    print(f"dominant: {str(result_p5['dominant_element']).upper()}")
    print("Zustände:           ", result_p5["element_states"])
    print("Prozentuale Ratios: ", result_p5["element_ratio"])


if __name__ == "__main__":
    test_two_different_hands()