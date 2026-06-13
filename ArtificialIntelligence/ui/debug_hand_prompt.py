import argparse
import json

from shared.events import Hand, HandEvent, HandTrigger

from ArtificialIntelligence.analysis.hand_analyzer import HandAnalyzer


def _load_payload(path: str | None) -> dict:
    if not path:
        return {
            "type": "hand",
            "trigger": "hand_detected",
            "hand": "left",
            "lengths": {
                "palm width": 0.067788,
                "palm_ height": 0.106587,
                "thumb_length": 0.093488,
                "index_length": 0.08549,
                "middle_length": 0.098961,
                "ring_length": 0.087926,
                "pinky_length": 0.071598,
            },
        }

    with open(path, "r") as handle:
        return json.load(handle)


def _to_hand_event(payload: dict) -> HandEvent:
    trigger_value = payload.get("trigger", "hand_detected")
    hand_value = payload.get("hand")

    trigger = HandTrigger(trigger_value)
    hand = Hand(hand_value) if hand_value else None

    return HandEvent(
        trigger=trigger,
        hand=hand,
        lengths=payload.get("lengths", {}),
        vector=payload.get("vector", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug hand prompt generation")
    parser.add_argument("--json", help="Path to a JSON payload file", default=None)
    args = parser.parse_args()

    payload = _load_payload(args.json)
    event = _to_hand_event(payload)
    prompt = HandAnalyzer().build_prompt(event)
    print(prompt)


if __name__ == "__main__":
    main()
