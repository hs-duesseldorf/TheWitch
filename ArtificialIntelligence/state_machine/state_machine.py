from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from transitions.extensions import GraphMachine

from shared.events import HandEvent, HandTrigger, PersonEvent, PersonTrigger, Scene


IDLE = Scene.SCENE_0_IDLE.value
ATTENTION = Scene.SCENE_1_ATTENTION.value
INTRO = Scene.SCENE_2_INTRO.value
REFOCUS = Scene.SCENE_2_REFOCUS.value
SCAN_READY = Scene.SCENE_3_SCAN_READY.value
HAND_CORRECTION = Scene.SCENE_3_HAND_CORRECTION.value
SCANNING = Scene.SCENE_3_SCANNING.value
SCAN_COMPLETE = Scene.SCENE_3_SCAN_COMPLETE.value
TRANSFORMATION = Scene.SCENE_4_TRANSFORMATION.value
INTRODUCTION = Scene.SCENE_5_INTRODUCTION.value
SHOT_1_VISUAL = Scene.SCENE_6_SHOT_1_VISUAL.value
SHOT_2_TASK = Scene.SCENE_6_SHOT_2_TASK.value
SHOT_3_ELEMENT = Scene.SCENE_6_SHOT_3_ELEMENT.value
SHOT_4_POSITIVE_NEGATIVE = Scene.SCENE_6_SHOT_4_POSITIVE_NEGATIVE.value
SHOT_5_BALANCE = Scene.SCENE_6_SHOT_5_BALANCE.value
RETURN = Scene.SCENE_7_RETURN.value
SMOKE_END = Scene.SCENE_7_SMOKE_END.value
VANISH_END = Scene.SCENE_7_VANISH_END.value
END = Scene.END.value

STATES = [scene.value for scene in Scene]
INITIAL = IDLE
ANY_SOURCE = "*"


@dataclass(frozen=True)
class StateChange:
    trigger: str
    source: str
    dest: str


def _transition(trigger: str, source: str | list[str], dest: str) -> dict[str, object]:
    return {"trigger": trigger, "source": source, "dest": dest}


TRANSITIONS = [
    _transition("ip_person_seated", IDLE, ATTENTION),
    _transition("ip_hand_absent", IDLE, ATTENTION),
    _transition("ip_hand_present", IDLE, INTRO),
    _transition("attention_done", ATTENTION, IDLE),
    _transition("intro_done", INTRO, SCAN_READY),
    _transition("ip_hand_removed", INTRO, REFOCUS),
    _transition("refocus_done", REFOCUS, INTRO),
    _transition("ip_hand_removed", SCAN_READY, REFOCUS),
    _transition("ip_hand_wrong", SCAN_READY, HAND_CORRECTION),
    _transition("ip_hand_right", SCAN_READY, SCANNING),
    _transition("correction_done", HAND_CORRECTION, SCAN_READY),
    _transition("ip_scan_incomplete", SCANNING, SCAN_READY),
    _transition("ip_hand_wrong", SCANNING, HAND_CORRECTION),
    _transition("ip_scan_complete", SCANNING, SCAN_COMPLETE),
    _transition("scan_complete_output_done", SCAN_COMPLETE, TRANSFORMATION),
    _transition("transformation_done", TRANSFORMATION, INTRODUCTION),
    _transition("introduction_done", INTRODUCTION, SHOT_1_VISUAL),
    _transition("shot_1_done", SHOT_1_VISUAL, SHOT_2_TASK),
    _transition("shot_2_done", SHOT_2_TASK, SHOT_3_ELEMENT),
    _transition("shot_3_done", SHOT_3_ELEMENT, SHOT_4_POSITIVE_NEGATIVE),
    _transition("shot_4_done", SHOT_4_POSITIVE_NEGATIVE, SHOT_5_BALANCE),
    _transition("shot_5_done", SHOT_5_BALANCE, RETURN),
    _transition("return_done", RETURN, SMOKE_END),
    _transition("end_done", [SMOKE_END, VANISH_END], END),
    _transition("reset", ANY_SOURCE, IDLE),
]

TRANSITION_IDS = list(dict.fromkeys(item["trigger"] for item in TRANSITIONS))


ANIMATION_TRIGGER_BY_STATE: dict[str, str] = {
    ATTENTION: "attention_done",
    INTRO: "intro_done",
    REFOCUS: "refocus_done",
    SCAN_COMPLETE: "scan_complete_output_done",
    TRANSFORMATION: "transformation_done",
    INTRODUCTION: "introduction_done",
    SHOT_1_VISUAL: "shot_1_done",
    SHOT_2_TASK: "shot_2_done",
    SHOT_3_ELEMENT: "shot_3_done",
    SHOT_4_POSITIVE_NEGATIVE: "shot_4_done",
    SHOT_5_BALANCE: "shot_5_done",
    RETURN: "return_done",
    SMOKE_END: "end_done",
    VANISH_END: "end_done",
}


HAND_TRIGGER_BY_STATE: dict[str, dict[str, str]] = {
    IDLE: {
        "absent": "ip_hand_absent",
        "present": "ip_hand_present",
        "wrong": "ip_hand_present",
        "ready": "ip_hand_present",
    },
    ATTENTION: {
        "absent": "attention_done",
        "present": "attention_done",
        "wrong": "attention_done",
        "ready": "attention_done",
    },
    INTRO: {
        "absent": "ip_hand_removed",
        "present": "intro_done",
        "wrong": "intro_done",
        "ready": "intro_done",
    },
    REFOCUS: {
        "present": "refocus_done",
        "wrong": "refocus_done",
        "ready": "refocus_done",
    },
    SCAN_READY: {
        "absent": "ip_hand_removed",
        "present": "ip_hand_right",
        "wrong": "ip_hand_wrong",
        "ready": "ip_hand_right",
    },
    HAND_CORRECTION: {
        "ready": "correction_done",
    },
    SCANNING: {
        "absent": "ip_scan_incomplete",
        "present": "ip_scan_incomplete",
        "wrong": "ip_hand_wrong",
        "ready": "ip_scan_complete",
    },
}


SCENES_THAT_START_ANALYSIS = frozenset({SCAN_COMPLETE, SHOT_1_VISUAL})
SCENES_THAT_DELIVER_FORTUNE = frozenset({SHOT_1_VISUAL})


STATE_DESCRIPTIONS: dict[str, str] = {
    IDLE: "Szene 0 - Idle: Wahrsagerin beschaeftigt sich selbst.",
    ATTENTION: "Szene 1 - Begruessung: Wahrsagerin macht Besucher auf den Stein aufmerksam.",
    INTRO: "Szene 2 - Einleitung: Wahrsagerin reagiert.",
    REFOCUS: "Szene 2 - Einleitung: Wahrsagerin richtet Besucher erneut aus.",
    SCAN_READY: "Szene 3 - Handscan: wartet auf korrekt eingelegte Hand.",
    HAND_CORRECTION: "Szene 3 - Handscan: fordert auf, die Hand richtig reinzulegen.",
    SCANNING: "Szene 3 - Handscan: Scan laeuft.",
    SCAN_COMPLETE: "Szene 3 - Handscan: Hand kann herausgenommen werden.",
    TRANSFORMATION: "Szene 4 - Transformation: Wahrsagerin verwandelt sich.",
    INTRODUCTION: "Szene 5 - Vorstellung: Wahrsagerin stellt sich kurz vor.",
    SHOT_1_VISUAL: "Szene 6 Shot 1: visuelle Darstellung der Hand.",
    SHOT_2_TASK: "Szene 6 Shot 2: kleine Interaktionsaufgabe.",
    SHOT_3_ELEMENT: "Szene 6 Shot 3: Element wird eingeordnet.",
    SHOT_4_POSITIVE_NEGATIVE: "Szene 6 Shot 4: positive und negative Punkte.",
    SHOT_5_BALANCE: "Szene 6 Shot 5: Element wird in Gleichgewicht gefuehrt.",
    RETURN: "Szene 7 - Ende: Wahrsagerin verwandelt sich zurueck.",
    SMOKE_END: "Ende A: Raum wird schwarz, Lichter gehen aus.",
    VANISH_END: "Ende B: Wahrsagerin verschwindet, Licht geht aus.",
    END: "Ende.",
}


class WitchStateMachine:
    def __init__(self) -> None:
        self._transition_handlers: dict[str, list[Callable[[StateChange], Awaitable[None]]]] = {}
        self._state: str = INITIAL
        self._machine = GraphMachine(
            model=self,
            states=STATES,
            transitions=TRANSITIONS,
            initial=INITIAL,
            auto_transitions=False,
            ignore_invalid_triggers=True,
            graph_engine="mermaid",
            title="The Witch State Machine",
        )

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value

    def register_transition_handler(self, trigger: str, handler: Callable[[StateChange], Awaitable[None]]) -> None:
        if trigger not in self._transition_handlers:
            self._transition_handlers[trigger] = []
        self._transition_handlers[trigger].append(handler)

    def get_transition_handlers(self, trigger: str) -> list[Callable[[StateChange], Awaitable[None]]]:
        return self._transition_handlers.get(trigger, [])

    @property
    def machine(self):
        return self._machine

    def hand_event(self, event: HandEvent) -> list[StateChange]:
        condition = hand_condition(event)
        max_changes = 1 if condition == "absent" else 8
        changes: list[StateChange] = []
        seen_states = {self.state}

        for _ in range(max_changes):
            trigger = HAND_TRIGGER_BY_STATE.get(self.state, {}).get(condition)
            change = self.advance(trigger)
            if change is None:
                break
            changes.append(change)
            if self.state in seen_states:
                break
            seen_states.add(self.state)

        return changes

    def person_event(self, event: PersonEvent) -> list[StateChange]:
        if event.trigger is not PersonTrigger.DETECTED:
            return []
        change = self.advance("ip_person_seated")
        return [change] if change else []

    def event_done(self, scene: str | None = None) -> list[StateChange]:
        if scene is not None and scene != self.state:
            return []
        change = self.advance(ANIMATION_TRIGGER_BY_STATE.get(self.state))
        return [change] if change else []

    def advance(self, trigger: str | None) -> StateChange | None:
        if trigger not in TRANSITION_IDS:
            return None

        run_trigger = getattr(self, trigger, None)
        if run_trigger is None:
            return None

        source = self.state
        if not run_trigger():
            return None
        return StateChange(trigger=trigger, source=source, dest=self.state)

    def force_state(self, state: str) -> str:
        if state in STATES:
            self._machine.set_state(state, model=self)
        return self.state

    def description(self) -> str:
        return STATE_DESCRIPTIONS.get(self.state, self.state)

    def save_markdown(self, path: Path | None = None) -> Path:
        path = path or Path(__file__).with_name("StateMachine.md")
        mermaid = self._machine.get_graph().source.replace("direction LR", "direction TB")
        path.write_text(f"```mermaid\n{mermaid.strip()}\n```\n", encoding="utf-8")
        return path


def hand_condition(event: HandEvent) -> str:
    if event.trigger is HandTrigger.ABSENT:
        return "absent"
    if event.trigger in {HandTrigger.WRONG_SIDE, HandTrigger.NOT_FULLY_IN_VIEW, HandTrigger.TILTED}:
        return "wrong"
    return "ready" if event.vector else "present"


if __name__ == "__main__":
    machine = WitchStateMachine()
    path = machine.save_markdown()
    print(f"Saved to {path}")