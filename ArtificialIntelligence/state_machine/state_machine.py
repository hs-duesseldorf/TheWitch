from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from transitions.extensions import GraphMachine

from shared.events import HandEvent, HandTrigger, PersonEvent, PersonTrigger, Scene

DEBUG_HAND_ABSENT = Scene.SCENE_DEBUG_SHOT_1_HAND_ABSENT.value
DEBUG_HAND_TILTED = Scene.SCENE_DEBUG_SHOT_2_HAND_TILTED.value
DEBUG_HAND_WRONG_SIDE = Scene.SCENE_DEBUG_SHOT_3_HAND_WRONG_SIDE.value

IDLE = Scene.SCENE_0_IDLE.value
OUTRO = Scene.SCENE_6_OUTRO.value
RESTART = Scene.SCENE_RESTART.value

SC1_START = Scene.SCENE_1_START.value
SC2_AWAITING_HAND = Scene.SCENE_2_AWAITING_HAND.value
SC2_HAND_FOUND = Scene.SCENE_2_HAND_FOUND.value
SC3_HANDSCAN_IN_PROCESS = Scene.SCENE_3_HANDSCAN_IN_PROCESS.value
SC3_HANDSCAN_DONE = Scene.SCENE_3_HANDSCAN_DONE.value
SC4_TRANSFORMATION = Scene.SCENE_4_TRANSFORMATION.value
SC5_HANDREAD_VISUALISATION = Scene.SCENE_5_HANDREAD_VISUALISATION.value
SC5_1_CORE_ELEMENT = Scene.SCENE_5_SHOT_1_CORE_ELEMENT.value
SC5_2_WEAK_ELEMENT = Scene.SCENE_5_SHOT_2_WEAK_ELEMENT.value
SC5_3_ADVICE = Scene.SCENE_5_SHOT_3_ADVICE.value

STATES = [scene.value for scene in Scene]

INITIAL = SC2_AWAITING_HAND
ANY_SOURCE = "*"

DEBUG_STATES = [DEBUG_HAND_ABSENT, 
                DEBUG_HAND_TILTED, 
                DEBUG_HAND_WRONG_SIDE]

@dataclass(frozen=True)
class StateChange:
    trigger: str
    source: str
    dest: str


def _transition(trigger: str, source: str | list[str], dest: str, **kwargs) -> dict[str, object]:
    data = {
        "trigger": trigger,
        "source": source,
        "dest": dest,
    }
    data.update(kwargs)
    return data


TRANSITIONS = [
    # Camera / hand input.

    # Debug
    _transition("ip_hand_absent",[ DEBUG_HAND_TILTED,DEBUG_HAND_WRONG_SIDE, SC2_AWAITING_HAND, SC2_HAND_FOUND], 
                                DEBUG_HAND_ABSENT, before="store_previous_transition"),
    _transition("ip_hand_tilted",[ DEBUG_HAND_ABSENT,DEBUG_HAND_WRONG_SIDE, SC2_AWAITING_HAND, SC2_HAND_FOUND], 
                                DEBUG_HAND_TILTED, before="store_previous_transition"),
    _transition("ip_hand_wrong_side",[ DEBUG_HAND_ABSENT,DEBUG_HAND_TILTED, SC2_AWAITING_HAND, SC2_HAND_FOUND], 
                                DEBUG_HAND_WRONG_SIDE, before="store_previous_transition"),
    _transition("exit_debug", [DEBUG_HAND_ABSENT, DEBUG_HAND_TILTED, DEBUG_HAND_WRONG_SIDE], 
                                None, before="return_to_previous_transition"),
    # Scene 0
    _transition("ip_person_seated", IDLE, SC1_START), 
    # Scene 1
    _transition("instructions_stone_done", SC1_START, SC2_AWAITING_HAND),
    # Scene 2
    _transition("ip_hand_present", SC2_AWAITING_HAND, SC2_HAND_FOUND),
    # Scene 3
    _transition("ip_hand_correct", SC2_HAND_FOUND, SC3_HANDSCAN_IN_PROCESS),
    _transition("hand_scanning", SC3_HANDSCAN_IN_PROCESS, SC3_HANDSCAN_DONE),
    _transition("scan_complete", SC3_HANDSCAN_DONE, SC4_TRANSFORMATION),
    # Scene 4
    _transition("transformation_done", SC4_TRANSFORMATION, SC5_HANDREAD_VISUALISATION),
    # Scene 5
    _transition("hand_visual_done", SC5_HANDREAD_VISUALISATION, SC5_1_CORE_ELEMENT),
    _transition("core_element_done", SC5_1_CORE_ELEMENT, SC5_2_WEAK_ELEMENT),
    _transition("weak_element_done", SC5_2_WEAK_ELEMENT, SC5_3_ADVICE),
    _transition("advice_done", SC5_3_ADVICE, OUTRO),
    # End
    _transition("ip_person_left", OUTRO, RESTART),
    _transition("reset", ANY_SOURCE, SC2_AWAITING_HAND),
]

TRANSITION_IDS = list(dict.fromkeys(item["trigger"] for item in TRANSITIONS))

ANIMATION_TRIGGER_BY_STATE: dict[str, str] = {
    SC1_START: "instructions_stone_done",
    SC4_TRANSFORMATION: "transformation_done",
    SC5_HANDREAD_VISUALISATION: "hand_visual_done",
    SC5_1_CORE_ELEMENT: "core_element_done",
    SC5_2_WEAK_ELEMENT: "weak_element_done",
    SC5_3_ADVICE: "advice_done",
    RESTART: "reset",
}

PERSON_TRIGGER_BY_STATE: dict[str, str] = {
    IDLE: "ip_person_seated",
    OUTRO: "ip_person_left",
}

HAND_TRIGGER_BY_STATE: dict[str, dict[str, str]] = {
    DEBUG_HAND_ABSENT: {
        "handback": "ip_hand_wrong_side",
        "tilted": "ip_hand_tilted",
        "ready": "exit_debug",
        "present": "exit_debug",
    }, 
    DEBUG_HAND_TILTED: {
        "absent": "ip_hand_absent",
        "handback": "ip_hand_wrong_side",
        "ready": "exit_debug",
        "present": "exit_debug",
    }, 
    DEBUG_HAND_WRONG_SIDE: {
        "absent": "ip_hand_absent",
        "tilted": "ip_hand_tilted",
        "ready": "exit_debug",
        "present": "exit_debug",
    },
    SC2_AWAITING_HAND: {
        "absent": "ip_hand_absent",
        "handback": "ip_hand_wrong_side",
        "tilted": "ip_hand_tilted",
        "ready": "ip_hand_present",
        "present": "ip_hand_present",
    },
    SC2_HAND_FOUND: {
        "absent": "ip_hand_absent",
        "handback": "ip_hand_wrong_side",
        "tilted": "ip_hand_tilted",
        "ready": "ip_hand_correct",
        "present": "ip_hand_correct",
    },
}

SCENES_THAT_START_ANALYSIS = frozenset({SC3_HANDSCAN_DONE, SC5_HANDREAD_VISUALISATION})
SCENES_THAT_DELIVER_FORTUNE = frozenset({SC5_HANDREAD_VISUALISATION})

STATE_DESCRIPTIONS: dict[str, str] = {
    DEBUG_HAND_ABSENT: "Debug, wenn Hand überhaupt nicht anwesend ist",
    DEBUG_HAND_TILTED: "Debug, wenn Hand nicht gerade, mit der Innenfläche nach oben, gehalten wird",
    DEBUG_HAND_WRONG_SIDE: "Debug, wenn die Hand Rückenseite gezeigt wird",

    IDLE: "Szene 0: Wahrsagerin reagiert nicht, Besucher betritt den Raum",
    OUTRO: "Szene 6: Wahrsagerin transformiert sich zurück und hat \"nichts mehr zu sagen\"",
    RESTART: "Szene 0: Besucher hat den Raum verlassen, Neustart des Ablaufs, warten auf neuen Besucher",

    SC1_START: "Szene 1: Besucher hat sich hingesetzt, Ausstellung startet, Anweisungen werden geliefert",
    SC2_AWAITING_HAND: "Szene 2: Hand liegt im Stein und ist (grob) für die Kamera erkennbar",
    SC2_HAND_FOUND: "Szene 2: Hand bleibt ruhig unter der Kamera",
    SC3_HANDSCAN_IN_PROCESS: "Szene 3: Hand ist erkannt und stabil, analyse startet",
    SC3_HANDSCAN_DONE: "Szene 3: Hand wurde analysiert, Witch sagt, Hand kann raus",
    SC4_TRANSFORMATION: "Szene 4: Witch transformiert sich für die Analyse der Hand",

    SC5_HANDREAD_VISUALISATION: "Szene 5: Visuelle Darstellung der Hand auf dem Bildschirm",
    SC5_1_CORE_ELEMENT: "Szene 5 Shot 1: Stärkstes Element wird erklärt",
    SC5_2_WEAK_ELEMENT: "Szene 5 Shot 2: Schwächstes Element wird erklärt",
    SC5_3_ADVICE: "Szene 5 Shot 3: Tipps und Tricks fürs Leben",

}


class WitchStateMachine:
    def __init__(self) -> None:
        self.previous_state = None
        self.previous_trigger = None
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
            send_event = True,
        )

    def store_previous_transition(self, event_data):
        source = event_data.transition.source
        if source not in DEBUG_STATES:
            self.previous_state = source
            self.previous_trigger = event_data.event.name

    def return_to_previous_transition(self, event_data):
        if self.previous_state:
            event_data.transition.dest = self.previous_state
            event_data.result = True

    
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
            if trigger == "exit_debug":
                break
            if self.state in seen_states:
                break
            seen_states.add(self.state)

        return changes

    def person_event(self, event: PersonEvent) -> list[StateChange]:
        if event.trigger is PersonTrigger.DETECTED:
            change = self.advance("ip_person_seated")
        if event.trigger is PersonTrigger.ABSENT:
            change = self.advance("ip_person_left")
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
    if event.trigger in {HandTrigger.ABSENT,  HandTrigger.NOT_FULLY_IN_VIEW}:
        return "absent"
    if event.trigger is HandTrigger.WRONG_SIDE:
        return "handback"
    if event.trigger is HandTrigger.TILTED:
        return "tilted"
    return "ready" if event.vector else "present"


if __name__ == "__main__":
    machine = WitchStateMachine()
    path = machine.save_markdown()
    print(f"Saved to {path}")