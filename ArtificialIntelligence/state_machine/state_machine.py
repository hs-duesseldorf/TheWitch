from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transitions.extensions import GraphMachine

from shared.events import HandEvent, HandTrigger, PersonEvent, PersonTrigger, Scene


@dataclass(frozen=True)
class StateDef:
    state: Scene
    description: str
    auto_trigger: str | None = None
    is_analysis: bool = False
    hand_locked: bool = False

    @property
    def id(self) -> str:
        return self.state.value


@dataclass(frozen=True)
class StateChange:
    trigger: str
    source: str
    dest: str


def _transition(
    trigger: str,
    source: str | list[str],
    dest: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    return {"trigger": trigger, "source": source, "dest": dest, **kwargs}


_DBG = [
    Scene.DBG_ABSENT.value,
    Scene.DBG_TILTED.value,
    Scene.DBG_WRONG.value,
    Scene.DBG_NOT_FULLY.value,
]

STATE_DEFS = [
    StateDef(state=Scene.SCENE0, description="Idle"),
    StateDef(state=Scene.SCENE1, description="Welcome"),
    StateDef(state=Scene.SCENE2, description="Seated greeting", auto_trigger="seated_done"),
    StateDef(state=Scene.SCENE3, description="Witch intro", auto_trigger="intro_done"),
    StateDef(state=Scene.SCENE4, description="Awaiting hand"),
    StateDef(state=Scene.SCENE5, description="Handscan", auto_trigger="scan_complete", hand_locked=True),
    StateDef(
        state=Scene.HAND_REMOVAL,
        description="Handscan complete; visitor may remove hand",
        auto_trigger="hand_removal_done",
        hand_locked=True,
    ),
    StateDef(state=Scene.SCENE6, description="Analysis", auto_trigger="analysis_done", is_analysis=True, hand_locked=True),
    StateDef(state=Scene.SCENE7, description="Outro", hand_locked=True),
    StateDef(state=Scene.DBG_ABSENT, description="Debug: hand absent"),
    StateDef(state=Scene.DBG_TILTED, description="Debug: hand tilted"),
    StateDef(state=Scene.DBG_WRONG, description="Debug: back of hand"),
    StateDef(state=Scene.DBG_NOT_FULLY, description="Debug: hand not fully in view"),
]

TRANSITIONS = [
    # Person events
    _transition(PersonTrigger.DETECTED.value, Scene.SCENE0.value, Scene.SCENE1.value),
    _transition(PersonTrigger.DETECTED.value, Scene.SCENE1.value, Scene.SCENE1.value),
    _transition(PersonTrigger.SEATED.value, Scene.SCENE0.value, Scene.SCENE1.value),
    _transition(PersonTrigger.SEATED.value, Scene.SCENE1.value, Scene.SCENE2.value),
    # _transition(PersonTrigger.ABSENT.value, "*", Scene.SCENE7.value, conditions="_is_not_outro_or_idle"),
    # _transition(PersonTrigger.ABSENT.value, Scene.SCENE7.value, Scene.SCENE0.value),

    # Correct hand — advance
    _transition(HandTrigger.DETECTED.value, Scene.SCENE4.value, Scene.SCENE5.value),
    _transition(HandTrigger.DETECTED.value, _DBG, None, before="return_to_previous_transition"),

    # Wrong hand from awaiting → debug
    _transition(HandTrigger.ABSENT.value, Scene.SCENE4.value, Scene.DBG_ABSENT.value, before="store_previous_transition"),
    _transition(HandTrigger.TILTED.value, Scene.SCENE4.value, Scene.DBG_TILTED.value, before="store_previous_transition"),
    _transition(HandTrigger.WRONG_SIDE.value, Scene.SCENE4.value, Scene.DBG_WRONG.value, before="store_previous_transition"),
    _transition(HandTrigger.NOT_FULLY_IN_VIEW.value, Scene.SCENE4.value, Scene.DBG_NOT_FULLY.value, before="store_previous_transition"),

    # Inter-debug transitions
    _transition(HandTrigger.ABSENT.value, [Scene.DBG_TILTED.value, Scene.DBG_WRONG.value, Scene.DBG_NOT_FULLY.value], Scene.DBG_ABSENT.value, before="store_previous_transition"),
    _transition(HandTrigger.TILTED.value, [Scene.DBG_ABSENT.value, Scene.DBG_WRONG.value, Scene.DBG_NOT_FULLY.value], Scene.DBG_TILTED.value, before="store_previous_transition"),
    _transition(HandTrigger.WRONG_SIDE.value, [Scene.DBG_ABSENT.value, Scene.DBG_TILTED.value, Scene.DBG_NOT_FULLY.value], Scene.DBG_WRONG.value, before="store_previous_transition"),
    _transition(HandTrigger.NOT_FULLY_IN_VIEW.value, [Scene.DBG_ABSENT.value, Scene.DBG_TILTED.value, Scene.DBG_WRONG.value], Scene.DBG_NOT_FULLY.value, before="store_previous_transition"),

    # Internal sequence
    _transition("seated_done", Scene.SCENE2.value, Scene.SCENE3.value),
    _transition("intro_done", Scene.SCENE3.value, Scene.SCENE4.value),
    _transition("restart_hand_prompt", Scene.SCENE5.value, Scene.SCENE4.value),
    _transition("scan_complete", Scene.SCENE5.value, Scene.HAND_REMOVAL.value),
    _transition("hand_removal_done", Scene.HAND_REMOVAL.value, Scene.SCENE6.value),
    _transition("analysis_done", Scene.SCENE6.value, Scene.SCENE7.value),
    _transition("reset", "*", Scene.SCENE0.value),
]


class WitchStateMachine:
    def __init__(self) -> None:
        self.previous_state: str | None = None
        self._states = [s.id for s in STATE_DEFS]
        self._defs = {s.id: s for s in STATE_DEFS}
        self._state = self._states[0]
        self.manual_mode = False
        self._machine = GraphMachine(
            model=self,
            states=self._states,
            transitions=TRANSITIONS,
            initial=self._states[0],
            auto_transitions=False,
            ignore_invalid_triggers=True,
            graph_engine="mermaid",
            title="The Witch State Machine",
            send_event=True,
        )

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value

    @property
    def machine(self):
        return self._machine

    def store_previous_transition(self, event_data) -> None:
        source = event_data.transition.source
        if not source.startswith("scene_debug_"):
            self.previous_state = source

    def return_to_previous_transition(self, event_data) -> None:
        if self.previous_state is None:
            return
        event_data.transition.dest = self.previous_state
        event_data.result = True

    def _is_not_outro_or_idle(self, event_data) -> bool:
        return self.state not in (Scene.SCENE7.value, Scene.SCENE0.value)

    def advance(self, trigger: str | None) -> list[StateChange]:
        if trigger is None:
            return []
        source = self.state
        run_trigger = getattr(self, trigger, None)
        if run_trigger is None:
            return []
        if not run_trigger():
            return []
        return [StateChange(trigger=trigger, source=source, dest=self.state)]

    def event_done(self, scene: str | None = None) -> list[StateChange]:
        if scene is not None and scene != self.state:
            return []
        return self.advance(self._get_state_def(self.state).auto_trigger)

    def force_state(self, state: str) -> str:
        if state in self._states:
            self._machine.set_state(state, model=self)
        return self.state

    def save_markdown(self, path: Path | None = None) -> Path:
        path = path or Path(__file__).with_name("StateMachine.md")
        mermaid = self._machine.get_graph().source.replace(
            "direction LR", "direction TB"
        )
        mermaid = "\n".join(line.rstrip() for line in mermaid.splitlines())
        path.write_text(f"```mermaid\n{mermaid.strip()}\n```\n", encoding="utf-8")
        return path

    def _get_state_def(self, state: str | None) -> StateDef:
        return self._defs.get(
            state or self.state, StateDef(Scene.SCENE0, state or self.state)
        )


if __name__ == "__main__":
    machine = WitchStateMachine()
    path = machine.save_markdown()
    print(f"Saved to {path}")
