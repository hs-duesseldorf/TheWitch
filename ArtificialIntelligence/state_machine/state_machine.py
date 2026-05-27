from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transitions.extensions import GraphMachine

from shared.events import HandEvent, HandTrigger, PersonEvent, PersonTrigger, Scene


DEBUG = Scene.SCENE_DEBUG_HAND_DETECTION.value
DEBUG_HAND_ABSENT = Scene.SCENE_DEBUG_SHOT_1_HAND_ABSENT.value
DEBUG_HAND_MOVING = Scene.SCENE_DEBUG_SHOT_2_HAND_MOVING.value
DEBUG_HAND_TILTED = Scene.SCENE_DEBUG_SHOT_3_HAND_TILTED.value
DEBUG_HAND_OUTSIDE_FRAME = Scene.SCENE_DEBUG_SHOT_4_HAND_OUTSIDE_FRAME.value
DEBUG_HAND_WRONG_SIDE = Scene.SCENE_DEBUG_SHOT_5_HAND_WRONG_SIDE.value

IDLE = Scene.SCENE_0_IDLE.value

SC1_AWAITING_HAND = Scene.SCENE_1_AWAITING_HAND.value
SC1_2_YES_HAND_FOUND = Scene.SCENE_1_SHOT_2_YES_HAND_FOUND.value

SC2_1_HAND_STAYS_FOCUSED = Scene.SCENE_2_SHOT_1_HAND_STAYS_FOCUSED.value

SC3_SCANNING_HAND = Scene.SCENE_3_SCANNING_HAND.value
SC3_1_CORRECT_HAND = Scene.SCENE_3_SHOT_1_CORRECT_HAND.value
SC3_4_SCAN_DONE = Scene.SCENE_3_SHOT_4_SCAN_DONE.value

SC4_TRANSFORM = Scene.SCENE_4_TRANSFORM.value

SC5_WITCH_ORIGIN_STORY = Scene.SCENE_5_WITCH_ORIGIN_STORY.value

SC6_VISUAL_IMAGE_HAND = Scene.SCENE_6_VISUAL_IMAGE_HAND.value
SC6_1_POINT_OUT_DETAILS = Scene.SCENE_6_SHOT_1_POINT_OUT_DETAILS.value
SC6_2_INTERACTIVE_TASK = Scene.SCENE_6_SHOT_2_INTERACTIVE_TASK.value
SC6_2_1_TASK_DONE = Scene.SCENE_6_SHOT_2_1_TASK_DONE.value
SC6_2_2_TASK_IGNORED = Scene.SCENE_6_SHOT_2_2_TASK_IGNORED.value
SC6_3_ASSIGN_ELEMENTS = Scene.SCENE_6_SHOT_3_ASSIGN_ELEMENTS.value
SC6_4_ELEMENT_ANALYSIS = Scene.SCENE_6_SHOT_4_ELEMENT_ANALYSIS.value
SC6_5_INNER_BALANCE = Scene.SCENE_6_SHOT_5_INNER_BALANCE.value

SC7_LAST_WORDS = Scene.SCENE_7_LAST_WORDS.value
SC7_1_RETURN_TO_IDLE = Scene.SCENE_7_SHOT_1_RETURN_TO_IDLE.value
SC7_2_DISAPPEAR = Scene.SCENE_7_SHOT_2_DISAPPEAR.value

RESTART = Scene.SCENE_RESTART.value


STATES = [scene.value for scene in Scene]
INITIAL = IDLE
ANY_SOURCE = "*"

DEBUG_STATES = {
    DEBUG,
    DEBUG_HAND_ABSENT,
    DEBUG_HAND_MOVING,
    DEBUG_HAND_TILTED,
    DEBUG_HAND_OUTSIDE_FRAME,
    DEBUG_HAND_WRONG_SIDE,
}

@dataclass(frozen=True)
class StateChange:
    trigger: str
    source: str
    dest: str


def transition(trigger: str, source: str | list[str], dest: str, **kwargs) -> dict[str, object]:
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
    transition("enter_debug", [SC1_AWAITING_HAND, 
                               SC1_2_YES_HAND_FOUND, 
                               SC2_1_HAND_STAYS_FOCUSED,
                               SC3_SCANNING_HAND, 
                               SC3_1_CORRECT_HAND], 
                               DEBUG, before="store_previous_state"),
    transition("ip_hand_absent", DEBUG, DEBUG_HAND_ABSENT),
    transition("ip_hand_moving", DEBUG, DEBUG_HAND_MOVING),
    transition("ip_hand_tilted", DEBUG, DEBUG_HAND_TILTED),
    transition("ip_hand_outside_frame", DEBUG, DEBUG_HAND_OUTSIDE_FRAME),
    transition("ip_hand_wrong_side", DEBUG, DEBUG_HAND_WRONG_SIDE),
    transition("exit_debug", [DEBUG, 
                              DEBUG_HAND_ABSENT, 
                              DEBUG_HAND_MOVING, 
                              DEBUG_HAND_TILTED, 
                              DEBUG_HAND_OUTSIDE_FRAME, 
                              DEBUG_HAND_WRONG_SIDE], 
                              DEBUG, after="return_to_previous_state"),
    
    # Scene 0
    transition("ip_person_seated", IDLE, SC1_AWAITING_HAND), 
    # Scene 1
    transition("ip_hand_present", SC1_AWAITING_HAND, SC1_2_YES_HAND_FOUND),
    transition("hand_found", SC1_2_YES_HAND_FOUND, SC2_1_HAND_STAYS_FOCUSED),
    # Scene 2
    transition("hand_stays_still", SC2_1_HAND_STAYS_FOCUSED, SC3_SCANNING_HAND),
    # Scene 3
    transition("ip_hand_correct", SC3_SCANNING_HAND, SC3_1_CORRECT_HAND),
    transition("hand_scanning", SC3_1_CORRECT_HAND, SC3_4_SCAN_DONE),
    transition("ip_scan_complete", SC3_4_SCAN_DONE, SC4_TRANSFORM),
    # Scene 4+5
    transition("transformation_done", SC4_TRANSFORM, SC5_WITCH_ORIGIN_STORY),
    transition("originstory_done", SC5_WITCH_ORIGIN_STORY, SC6_VISUAL_IMAGE_HAND),
    # Scene 6
    transition("hand_visual_done", SC6_VISUAL_IMAGE_HAND, SC6_1_POINT_OUT_DETAILS),
    transition("reading_hand_details_done", SC6_1_POINT_OUT_DETAILS, SC6_2_INTERACTIVE_TASK),
    transition("ip_hand_present", SC6_2_INTERACTIVE_TASK, SC6_2_1_TASK_DONE),
    transition("ip_hand_absent", SC6_2_INTERACTIVE_TASK, SC6_2_2_TASK_IGNORED),
    transition("task_done_or_skipped", [SC6_1_POINT_OUT_DETAILS, SC6_2_1_TASK_DONE, SC6_2_2_TASK_IGNORED], SC6_3_ASSIGN_ELEMENTS),
    transition("reading_assign_element_done", SC6_3_ASSIGN_ELEMENTS, SC6_4_ELEMENT_ANALYSIS),
    transition("reading_analysis_done", SC6_4_ELEMENT_ANALYSIS, SC6_5_INNER_BALANCE),
    transition("reading_balance_done", SC6_5_INNER_BALANCE, SC7_LAST_WORDS),
    # Scene 7
    transition("detransfrom_end", SC7_LAST_WORDS, SC7_1_RETURN_TO_IDLE),
    transition("disappear_end", SC7_LAST_WORDS, SC7_2_DISAPPEAR),
    transition("end_done", [SC7_1_RETURN_TO_IDLE, SC7_2_DISAPPEAR], RESTART),
    # End
    transition("reset", ANY_SOURCE, IDLE),
]

TRANSITION_IDS = list(dict.fromkeys(item["trigger"] for item in TRANSITIONS))

ANIMATION_TRIGGER_BY_STATE = {
    SC3_4_SCAN_DONE: "ip_scan_complete",
    SC4_TRANSFORM: "transformation_done",
    SC5_WITCH_ORIGIN_STORY: "originstory_done",
    SC6_VISUAL_IMAGE_HAND: "hand_visual_done",
    SC6_1_POINT_OUT_DETAILS: "reading_hand_details_done",
    SC6_2_1_TASK_DONE: "task_done_or_skipped",
    SC6_2_2_TASK_IGNORED: "task_done_or_skipped",
    SC6_3_ASSIGN_ELEMENTS: "reading_assign_element_done",
    SC6_4_ELEMENT_ANALYSIS: "reading_analysis_done",
    SC6_5_INNER_BALANCE: "reading_balance_done",
    SC7_1_RETURN_TO_IDLE: "end_done",
    SC7_2_DISAPPEAR: "end_done",
    RESTART: "reset",
}


HAND_TRIGGER_BY_STATE = {
    IDLE: {
        "present": "ip_person_seated",
        "wrong": "ip_person_seated",
        "ready": "ip_person_seated",
    },
    SC1_AWAITING_HAND: {
        "absent": "enter_debug",
        "present": "ip_hand_present",
        "wrong": "ip_hand_present",
        "ready": "ip_hand_present",
    },
    SC1_2_YES_HAND_FOUND: {
        "absent": "enter_debug",
        "present": "hand_found",
        "wrong": "hand_found",
        "ready": "hand_found",
    },
    SC2_1_HAND_STAYS_FOCUSED: {
        "absent": "enter_debug",
        "present": "hand_stays_still",
        "wrong": "hand_stays_still",
        "ready": "hand_stays_still",
    },
    SC3_SCANNING_HAND: {
        "absent": "enter_debug",
        "present": "enter_debug",
        "wrong": "enter_debug",
        "ready": "ip_hand_correct",
    },
    SC3_1_CORRECT_HAND: {
        "absent": "enter_debug",
        "present": "enter_debug",
        "wrong": "enter_debug",
        "ready": "hand_scanning",
    },
    SC3_4_SCAN_DONE: {
        "ready": "ip_scan_complete",
    },
    SC6_2_INTERACTIVE_TASK: {
        "absent": "enter_debug",
        "ready": "ip_hand_present",
    },
    DEBUG: {
        "absent": "ip_hand_absent",
        "tilted": "ip_hand_tilted",
        "wrong_side": "ip_hand_wrong_side",
        "outside_frame": "ip_hand_outside_frame",
        "present": "exit_debug",
        "ready": "exit_debug",
    },
    DEBUG_HAND_ABSENT: {
        "present": "exit_debug",
        "ready": "exit_debug",
    },
    DEBUG_HAND_MOVING: {
        "present": "exit_debug",
        "ready": "exit_debug",
    },
    DEBUG_HAND_TILTED: {
        "present": "exit_debug",
        "ready": "exit_debug",
    },
    DEBUG_HAND_OUTSIDE_FRAME: {
        "present": "exit_debug",
        "ready": "exit_debug",
    },
    DEBUG_HAND_WRONG_SIDE: {
        "present": "exit_debug",
        "ready": "exit_debug",
    }
}

SCENES_THAT_START_ANALYSIS = frozenset({SC3_4_SCAN_DONE, SC6_VISUAL_IMAGE_HAND})
SCENES_THAT_DELIVER_FORTUNE = frozenset({SC6_VISUAL_IMAGE_HAND})

STATE_DESCRIPTIONS = {
    DEBUG: "Szene wird immer aufgerufen, wenn Hand in relevanten Szenen nicht richtig erkannt wird",
    DEBUG_HAND_ABSENT: "Debug, wenn Hand überhaupt nicht anwesend ist",
    DEBUG_HAND_MOVING: "Debug, wenn Hand sich bewegt", # GERADE NICHT IMPLEMENTIERT!!
    DEBUG_HAND_TILTED: "Debug, wenn Hand nicht gerade, mit der Innenfläche nach oben, gehalten wird",
    DEBUG_HAND_OUTSIDE_FRAME: "Debug, wenn Hand nicht vollständig von der Kamera zu erfassen ist",
    DEBUG_HAND_WRONG_SIDE: "Debug, wenn die falsche Hand (Rechts/Links) verwendet wurde",

    IDLE: "Szene 0: Wahrsagerin reagiert nicht, Besucher betritt den Raum",

    SC1_AWAITING_HAND: "Szene 1: Besucher hat sich hingesetzt, Ausstellung startet",
    #SC1_1_NO_HAND_FOUND: "Szene 1 Shot 1: Aufforderung der Witch, Hand in den Stein zu legen",
    SC1_2_YES_HAND_FOUND: "Szene 1 Shot 2: Hand liegt im Stein und ist (grob) für die Kamera erkennbar",

    SC2_1_HAND_STAYS_FOCUSED: "Szene 2 Shot 1: Hand bleibt ruhig unter der Kamera",
    #SC2_2_HAND_WITHDRAWN: "Szene 2 Shot 2: Hand wurde zurückgezogen, Aufforderung sie zurück zu legen",
    #SC2_3_STILL_NO_HAND: "Szene 2 Shot 3: Hand bleibt fern, erneute, strengere Aufforderung sie zurück zu legen",

    SC3_SCANNING_HAND: "Szene 3: Hand wird erkannt, warten auf stabiles Bild",
    SC3_1_CORRECT_HAND: "Szene 3 Shot 1: Hand liegt still, stabil und gut erkenntlich",
    #SC3_2_HAND_NOT_READABLE: "Szene 3 Shot 2: Hand ist nicht vollständig auf der Kamera erkennbar",
    #SC3_3_HAND_FAST_MOVEMENTS: "Szene 3 Shot 3: Hand bewegt sich, Scan dauert lange",
    SC3_4_SCAN_DONE: "Szene 3 Shot 4: Hand wurde gescannt, Witch sagt, Hand kann raus",

    SC4_TRANSFORM: "Szene 4: Witch transformiert sich für die Analyse der Hand",

    SC5_WITCH_ORIGIN_STORY: "Szene 5: Witch erklärt ihre Geschichte, leutet Analyse ein",

    SC6_VISUAL_IMAGE_HAND: "Szene 6: Visuelle Darstellung der Hand auf dem Bildschirm",
    SC6_1_POINT_OUT_DETAILS: "Szene 6 Shot 1: Details der Hand werden erklärt",
    SC6_2_INTERACTIVE_TASK: "Szene 6 Shot 2: Kleine, interaktive mögliche Zwischenaufgabe der Witch an den Besucher, Hand wieder in Stein",
    SC6_2_1_TASK_DONE: "Szene 6 Shot 2_1: Aufgabe erledigt, Hand kann zurückgenommen werden",
    SC6_2_2_TASK_IGNORED: "Szene 6 Shot 2_2: Aufgabe ignoriert, Witch macht weiter",
    SC6_3_ASSIGN_ELEMENTS: "Szene 6 Shot 3: Zuordnung zu den passenden Elementen, Erklärung dieser",
    SC6_4_ELEMENT_ANALYSIS: "Szene 6 Shot 4: Erläuterung der Stärken und Schattenseiten",
    SC6_5_INNER_BALANCE: "Szene 6 Shot 5: Erläuterung des inneren Gleichgewichts durch die restlichen Elemente",

    SC7_LAST_WORDS: "Szene 7: Schlussworte der Witch",
    SC7_1_RETURN_TO_IDLE: "Szene 7 Shot 1: Witch verwandelt sich zurück, sieht aus wie beim Start",
    SC7_2_DISAPPEAR: "Szene 7 Shot 2: Witch verschwindet in einer Wolke, Bildschirm bleibt leer",

    RESTART: "Szene 0: Besucher hat den Raum verlassen, Neustart des Ablaufs, warten auf neuen Besucher",

}


class WitchStateMachine:
    def __init__(self) -> None:
        self.previous_state = None
        self.machine = GraphMachine(
            model=self,
            states=STATES,
            transitions=TRANSITIONS,
            initial=INITIAL,
            auto_transitions=False,
            ignore_invalid_triggers=True,
            graph_engine="mermaid",
            title="The Witch State Machine",
        )

    def store_previous_state(self, event):
        self.previous_state = event.transition.source
    def return_to_previous_state(self):
        if self.previous_state:
            self.to_state(self.previous_state)

    def hand_event(self, event: HandEvent) -> list[StateChange]:
        # Gives out detailed or general events depending on if we are in a DEBUG State or not
        condition = self.hand_condition_by_state(event)
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
            self.machine.set_state(state, model=self)
        return self.state

    def description(self) -> str:
        return STATE_DESCRIPTIONS.get(self.state, self.state)

    def save_markdown(self, path: Path | None = None) -> Path:
        path = path or Path(__file__).with_name("StateMachine.md")
        mermaid = self.machine.get_graph().source.replace("direction LR", "direction TB")
        path.write_text(f"```mermaid\n{mermaid.strip()}\n```\n", encoding="utf-8")
        return path
        
    def hand_condition_by_state(self, event: HandEvent) -> str:
        if self.state in DEBUG_STATES:
            return hand_condition_debug(event)
        return hand_condition(event)



def hand_condition(event: HandEvent) -> str:
    if event.trigger is HandTrigger.ABSENT:
        return "absent"
    if event.trigger in {HandTrigger.WRONG_SIDE, HandTrigger.NOT_FULLY_IN_VIEW, HandTrigger.TILTED}:
        return "wrong"
    return "ready" if event.vector else "present"

def hand_condition_debug(event: HandEvent) -> str:
    match event.trigger:
        case HandTrigger.ABSENT: 
            return "absent"
        case HandTrigger.WRONG_SIDE:
            return "wrong_side"
        case HandTrigger.NOT_FULLY_IN_VIEW:
            return "outside_frame"
        case HandTrigger.TILTED:
            return "tilted"
        #case HandTrigger.MOVING:
        #    return "moving"
    return "ready" if event.vector else "present"

if __name__ == "__main__":
    machine = WitchStateMachine()
    path = machine.save_markdown()
    print(f"Saved to {path}")
