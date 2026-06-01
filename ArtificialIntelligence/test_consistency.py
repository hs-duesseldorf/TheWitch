import json
from hand_analysis import build_result

# Person 1 (Linke Hand)
person_1_input = {
    "handedness": "left",
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
    "handedness": "left",
    "lengths": {
        "palm_width": 0.066162,
        "palm_height": 0.102347,
        "thumb length": 0.090121,
        "index_length": 0.083411,
        "middle length": 0.096438,
        "ring_length": 0.084703,
        "pinky_length": 0.067061
    }
}

# Person 2 (Rechte Hand)
person_2_input = {
    "handedness": "right",
    "lengths": {
        "palm_width": 0.065816, # vereinheitlicht zu palm_width
        "palm_height": 0.096213,
        "thumb_length": 0.09111,
        "index_length": 0.086843,
        "middle_length": 0.101181,
        "ring_length": 0.090138,
        "pinky_length": 0.07071
    }
}


person_2_1_input = {
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
    "handedness": "left",
    "lengths": {
        "palm_width": 0.0623,
        "palm_height" : 0.099003,
        "thumb_length": 6.093183,
        "index_length": 0.079928,
        "middle_length": 0.093395,
        "ring_length": 0.081941,
        "pinky_length": 0.068093
    }
}


person_3_input = {
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
    "handedness": "right",
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
    "handedness": "left",
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
    "handedness": "left",
    "lengths": {
        "palm_width": 0.059105,
        "palm_height": 0.089929,
        "thumb_length": 0.100096,
        "index_length": 0.079477,
        "middle_length": 0.093064,
        "ring_length":0.081487,
        "pinky_length": 0.065981
    }
}

person_6_input = { 
    "handedness": "left",
   "lengths":{
        "palm_width": 0.067627,
        "palm_height": 0.094915,
        "thumb_length": 0.089369,
        "index length": 0.086019,
        "middle_length": 0.102306,
        "ring length": 0.089295,
        "pinky_length": 0.071646
    }
}

person_7_input = {
    "handedness": "right",
    "lengths": {
        "palm_width": 0.071476,
        "palm_height": 0.109064,
        "thumb_length": 0.100975,
        "index length": 0.086389,
        "middle_length": 0.09965,
        "ring_length": 0.085125,
        "pinky_length": 0.072942
    }
}
person_8_input = {
    "handedness": "left",
    "lengths": {
        "palm_width": 0.063659,
        "palm_height": 0.099951,
        "thumb_length": 0.09101,
        "index_length": 0.084768,
        "middle_length": 0.099127,
        "ring_length":0.088671,
        "pinky_length": 0.06991
    }
}

person_9_input = {
    "hand": "left",
    "lengths": {
        "palm_width": 0.067187,
        "palm_height": 0.097198,
        "thumb_length": 0.090471,
        "index_length": 0.086512,
        "middle_length": 0.097389,
        "ring_length": 0.085578,
        "pinky_length": 0.067006
    }
}

person_10_input = {
    "hand": "left",
    "lengths": {
        "palm_width": 0.065258,
        "palm_height":0.095572,
        "thumb_length": 0.095355,
        "index_length": 0.087284,
        "middle_length": 0.100675,
        "ring_length":0.089798,
        "pinky_length": 0.070167
    }
}
person_11_input = {
    "hand": "right",
    "lengths": {
        "palm_width": 0.065938,
        "palm_height": 0.095267,
        "thumb_length": 0.094228,
        "index_length": 0.08903,
        "middle_length": 0.102737,
        "ring_length":0.091342,
        "pinky_length": 0.070738
    }
}

person_12_input = {
    "hand": "right",
    "lengths": {
        "palm_width": 0.069997,
        "palm_height": 0.107937,
        "thumb_length": 0.099824,
        "index_length": 0.086438,
        "middle_length": 0.099315,
        "ring_length": 0.085584,
        "pinky_length": 0.072242
    }
}

person_13_input = {
    "hand": "right",
    "lengths": {
        "palm_width": 0.071476,
        "palm_height": 0.109064,
        "thumb_length": 0.100975,
        "index_length": 0.086389,
        "middle_length": 0.09965,
        "ring_length":0.085125,
        "pinky_length": 0.072942
    }
}





def test():
    persons = [
        ("Person 1 (Links)", person_1_input),
        ("Person 1.1 (Links)", person_1_1_input),
        ("Person 2 (Rechts)", person_2_input),
        ("Person 2.1 (Links)", person_2_1_input),
        ("Person 2.2 (Links)", person_2_2_input),
        ("Person 3 (Rechts)", person_3_input),
        ("Person 3.1 (Rechts)", person_3_1_input),
        ("Person 4 (Links)", person_4_input),
        ("Person 5 (Links)", person_5_input),
        ("Person 6 (Links)", person_6_input),
        ("Person 7 (Rechts)", person_7_input),
        ("Person 8 (Links)", person_8_input),
        ("Person 9 (Links)", person_9_input),
        ("Person 10 (Links)", person_10_input),
        ("Person 11 (Rechts)", person_11_input),
        ("Person 12 (Rechts)", person_12_input),
        ("Person 13 (Rechts)", person_13_input),
    ]

    results = []
    for label, payload in persons:
        result = build_result(payload)
        results.append((label, result))
        print(f"\n--- ERGEBNIS {label} ---")
        print(json.dumps(result, indent=2))

    print("\n--- ZUSAMMENFASSUNG von dominantelementen---")
    for label, result in results:
        print(f"{label}: {result['dominant_element']}")




    
if __name__ == "__main__":
    test()