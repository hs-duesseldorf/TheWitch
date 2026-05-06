from __future__ import annotations
from pathlib import Path
from transitions.extensions import GraphMachine


# States

STATES = ["greeting", "hand_prompt", "hand_wrong", "analyzing", "analysis_paused", "result", "end"]
INITIAL = "greeting"


# Callbacks

def on_visitor_seated(_):        print("Ahh du bist da… Lege dein Hand hinein")
def on_dare(_):                  print("Traust du dich?")
def on_visitor_left(_):          print("Nicht jeder ist bereit, die Wahrheit zu hören… böses Lachen")
def on_hand_wrong(_):            print("Ich kann deine Hand noch nicht sehen… leuchtet rot")
def on_analysis_start(_):        print("Beginn: Analyse der Hand. Interessant / Aha")
def on_analysis_paused(_):       print("Warte… ich war noch nicht fertig. leuchtet rot")
def on_analysis_continued(_):    print("Dann lass uns mal weiter machen")
def on_hand_lost_in_analysis(_): print("Ich kann deine Hand nicht mehr sehen… leuchtet rot")


# Transitions  (trigger, from, to, callback)

TRANSITIONS = [
    ("visitor_sits",        "greeting",                         "hand_prompt",       on_visitor_seated),

    ("hand_placed_wrong",   ["hand_prompt", "analysis_paused"], "hand_wrong",        on_hand_wrong),
    ("hand_placed_correct", ["hand_prompt", "hand_wrong",
                             "analysis_paused"],                "analyzing",         on_analysis_start),
    ("no_hand_detected",    "hand_prompt",                      "hand_prompt",       on_dare),
    ("visitor_left",        "hand_prompt",                      "end",               on_visitor_left),

    ("hand_still_wrong",    "hand_wrong",                       "hand_wrong",        on_hand_wrong),

    ("hand_removed",        "analyzing",                        "analysis_paused",   on_analysis_paused),
    ("hand_lost_fully",     "analysis_paused",                  "analysis_paused",   on_hand_lost_in_analysis),
    ("hand_returned",       "analysis_paused",                  "analyzing",         on_analysis_continued),
    ("analysis_done",       "analyzing",                        "result",            None),

    ("reset",               "*",                                "greeting",          None),
]


class WitchStateMachine:
    def __init__(self) -> None:
        self.machine = GraphMachine(
            model=self,
            states=STATES,
            transitions=[
                {"trigger": t, "source": src, "dest": dst, **({"after": cb} if cb else {})}
                for t, src, dst, cb in TRANSITIONS
            ],
            initial=INITIAL,
            auto_transitions=False,
            ignore_invalid_triggers=True,
            send_event=True,
            graph_engine="mermaid",
            title="The Witch State Machine",
        )

    def trigger(self, event: str) -> str:
        """Call this with the websocket event name. Returns current state."""
        trigger_fn = getattr(self, event, None)
        if trigger_fn is None:
            print(f"Unknown event: {event!r}")
        else:
            trigger_fn()
        return self.state

    def save_markdown(self, path: Path | None = None) -> Path:
        path = path or Path(__file__).with_name("StateMachine.md")
        mermaid = self.get_graph().source.replace("direction LR", "direction TB")
        path.write_text(f"```mermaid\n{mermaid.strip()}\n```\n", encoding="utf-8")
        return path


if __name__ == "__main__":
    machine = WitchStateMachine()
    path = machine.save_markdown()
    print(f"Saved to {path}")