from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transitions.extensions import GraphMachine

from shared.events import HandTrigger, PersonTrigger, Scene


@dataclass(frozen=True)
class State:
    state: Scene
    description: str
    auto_trigger: str | None = None
    is_analysis: bool = False
    hand_locked: bool = False

    @property
    def id(self) -> str:
        return self.state.value


@dataclass(frozen=True)
class Transition:
    trigger: str
    source: str
    dest: str


STATES = [
    State(Scene.SCENE_0_IDLE, "Idle"),
    State(Scene.SCENE_1_WELCOME, "Welcome"),
    State(
        Scene.SCENE_2_SEATED, "Seated greeting", auto_trigger="scene_2_seated_done"
    ),
    State(Scene.SCENE_3_INTRO, "Witch intro", auto_trigger="scene_3_intro_done"),
    State(Scene.SCENE_4_AWAITING_HAND, "Awaiting hand"),
    State(
        Scene.SCENE_5_HANDSCAN,
        "Handscan",
        auto_trigger="scene_5_handscan_done",
        hand_locked=True,
    ),
    State(
        Scene.SCENE_6_HAND_DETECTED,
        "Hand detected; confirming presence",
    ),
    State(
        Scene.SCENE_7_HANDSCAN_DONE,
        "Handscan complete; visitor may remove hand",
        auto_trigger="scene_7_handscan_done_done",
    ),
    State(
        Scene.SCENE_8_ANALYSIS,
        "Analysis",
        auto_trigger="scene_8_analysis_done",
        is_analysis=True,
    ),
    State(Scene.SCENE_9_OUTRO, "Outro"),
    State(Scene.DEBUG_HAND_PULLED_AWAY, "Debug: hand pulled away"),
    State(Scene.DEBUG_HAND_TILTED, "Debug: hand tilted"),
    State(Scene.DEBUG_WRONG_SIDE, "Debug: back of hand"),
    State(Scene.DEBUG_NOT_FULLY_IN_VIEW, "Debug: hand not fully in view"),
    State(Scene.DEBUG_GASLIGHT, "Debug: gaslight"),
]


TRANSITIONS = [
    Transition(
        PersonTrigger.DETECTED.value,
        Scene.SCENE_0_IDLE.value,
        Scene.SCENE_1_WELCOME.value,
    ),
    Transition(
        PersonTrigger.SEATED.value,
        Scene.SCENE_0_IDLE.value,
        Scene.SCENE_1_WELCOME.value,
    ),
    Transition(
        PersonTrigger.SEATED.value,
        Scene.SCENE_1_WELCOME.value,
        Scene.SCENE_2_SEATED.value,
    ),
    Transition(
        PersonTrigger.ABSENT.value, Scene.SCENE_9_OUTRO.value, Scene.SCENE_0_IDLE.value
    ),
    Transition(
        HandTrigger.DETECTED.value,
        Scene.SCENE_4_AWAITING_HAND.value,
        Scene.SCENE_5_HANDSCAN.value,
    ),
    Transition(
        HandTrigger.TILTED.value,
        Scene.SCENE_4_AWAITING_HAND.value,
        Scene.DEBUG_HAND_TILTED.value,
    ),
    Transition(
        HandTrigger.WRONG_SIDE.value,
        Scene.SCENE_4_AWAITING_HAND.value,
        Scene.DEBUG_WRONG_SIDE.value,
    ),
    Transition(
        HandTrigger.NOT_FULLY_IN_VIEW.value,
        Scene.SCENE_4_AWAITING_HAND.value,
        Scene.DEBUG_NOT_FULLY_IN_VIEW.value,
    ),
    Transition(
        "gaslight_reject", Scene.SCENE_4_AWAITING_HAND.value, Scene.DEBUG_GASLIGHT.value
    ),
    Transition(
        HandTrigger.ABSENT.value,
        Scene.SCENE_5_HANDSCAN.value,
        Scene.DEBUG_HAND_PULLED_AWAY.value,
    ),
    Transition(
        HandTrigger.TILTED.value,
        Scene.SCENE_5_HANDSCAN.value,
        Scene.DEBUG_HAND_TILTED.value,
    ),
    Transition(
        HandTrigger.WRONG_SIDE.value,
        Scene.SCENE_5_HANDSCAN.value,
        Scene.DEBUG_WRONG_SIDE.value,
    ),
    Transition(
        HandTrigger.NOT_FULLY_IN_VIEW.value,
        Scene.SCENE_5_HANDSCAN.value,
        Scene.DEBUG_NOT_FULLY_IN_VIEW.value,
    ),
    Transition(
        "scene_2_seated_done", Scene.SCENE_2_SEATED.value, Scene.SCENE_3_INTRO.value
    ),
    Transition(
        "scene_3_intro_done",
        Scene.SCENE_3_INTRO.value,
        Scene.SCENE_4_AWAITING_HAND.value,
    ),
    Transition(
        "restart_hand_prompt",
        Scene.SCENE_5_HANDSCAN.value,
        Scene.SCENE_4_AWAITING_HAND.value,
    ),
    Transition(
        "scene_5_handscan_done",
        Scene.SCENE_5_HANDSCAN.value,
        Scene.SCENE_6_HAND_DETECTED.value,
    ),
    Transition(
        HandTrigger.DETECTED.value,
        Scene.SCENE_6_HAND_DETECTED.value,
        Scene.SCENE_7_HANDSCAN_DONE.value,
    ),
    Transition(
        HandTrigger.ABSENT.value,
        Scene.SCENE_6_HAND_DETECTED.value,
        Scene.DEBUG_HAND_PULLED_AWAY.value,
    ),
    Transition(
        HandTrigger.TILTED.value,
        Scene.SCENE_6_HAND_DETECTED.value,
        Scene.DEBUG_HAND_TILTED.value,
    ),
    Transition(
        HandTrigger.WRONG_SIDE.value,
        Scene.SCENE_6_HAND_DETECTED.value,
        Scene.DEBUG_WRONG_SIDE.value,
    ),
    Transition(
        HandTrigger.NOT_FULLY_IN_VIEW.value,
        Scene.SCENE_6_HAND_DETECTED.value,
        Scene.DEBUG_NOT_FULLY_IN_VIEW.value,
    ),
    Transition(
        "scene_7_handscan_done_done",
        Scene.SCENE_7_HANDSCAN_DONE.value,
        Scene.SCENE_8_ANALYSIS.value,
    ),
    Transition(
        "scene_8_analysis_done", Scene.SCENE_8_ANALYSIS.value, Scene.SCENE_9_OUTRO.value
    ),
]


class _GraphModel:
    pass


class StateMachine:
    def __init__(self) -> None:
        self._state = Scene.SCENE_0_IDLE.value

        self._defs = {state.id: state for state in STATES}

        self._transitions = {(t.source, t.trigger): t.dest for t in TRANSITIONS}

        self._graph_model = _GraphModel()

        self._graph_machine = GraphMachine(
            model=self._graph_model,
            states=[state.id for state in STATES],
            transitions=[
                {
                    "trigger": t.trigger,
                    "source": t.source,
                    "dest": t.dest,
                }
                for t in TRANSITIONS
            ],
            initial=self._state,
            auto_transitions=False,
        )

        self._graph_model.state = self._state

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_scene(self) -> Scene:
        return Scene(self._state)

    @property
    def machine(self):
        return self

    def trigger(self, trigger: str | None) -> Transition | None:
        changes = self.advance(trigger)
        return changes[-1] if changes else None

    def advance(self, trigger: str | None) -> list[Transition]:
        if trigger is None:
            return []

        if trigger == "reset":
            return [self.reset()]

        source = self._state

        destination = self._transitions.get((source, trigger))
        if destination is None:
            return []

        return [self._set_state(trigger, destination)]

    def event_done(self, scene: str | None = None) -> list[Transition]:
        if scene is not None and scene != self._state:
            return []

        auto_trigger = self._get_state_def(self._state).auto_trigger
        return self.advance(auto_trigger)

    def force_state(self, state: str) -> str:
        if state in self._defs:
            self._state = state
            self._graph_model.state = state
        return self._state

    def reset(self) -> Transition:
        return self._set_state("reset", Scene.SCENE_0_IDLE.value)

    def get_graph(self):
        return self._graph_machine.get_graph(title=None, force_new=True)

    def save_markdown(self, path: Path | None = None) -> Path:
        path = path or Path(__file__).with_name("StateMachine.md")

        graph = self.get_graph()

        path.write_text(
            f"```mermaid\n{graph.source}\n```\n",
            encoding="utf-8",
        )

        return path

    def _set_state(self, trigger: str, destination: str) -> Transition:
        source = self._state

        self._state = destination
        self._graph_model.state = destination

        return Transition(
            trigger=trigger,
            source=source,
            dest=destination,
        )

    def _get_state_def(self, state: str | None) -> State:
        return self._defs.get(
            state or self._state,
            State(Scene.SCENE_0_IDLE, state or self._state),
        )
