from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias

import msgspec


class HandTrigger(str, Enum):
    DETECTED = "hand_detected"
    ABSENT = "hand_absent"
    WRONG_SIDE = "hand_wrong_side"
    NOT_FULLY_IN_VIEW = "hand_not_fully_in_view"
    TILTED = "hand_tilted"


class PersonTrigger(str, Enum):
    DETECTED = "person_detected"
    SEATED = "person_seated"
    ABSENT = "person_absent"


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Scene(str, Enum):
    DBG_ABSENT = "scene_debug_shot_1_hand_absent"
    DBG_TILTED = "scene_debug_shot_2_hand_tilted"
    DBG_WRONG = "scene_debug_shot_3_hand_wrong_side"
    DBG_NOT_FULLY = "scene_debug_shot_4_hand_not_fully_in_view"

    SCENE0 = "scene_idle"
    SCENE1 = "scene_0_welcome"
    SCENE2 = "scene_1_seated"
    SCENE3 = "scene_2_intro"
    SCENE4 = "scene_3_awaiting_hand"
    SCENE5 = "scene_4_handscan"
    HAND_REMOVAL = "scene_4_handscan_done"
    SCENE6 = "scene_5_analysis"
    SCENE7 = "scene_6_outro"


class Event(msgspec.Struct, frozen=True, kw_only=True, tag_field="type", omit_defaults=True):
    origin: str | None = None


class HandEvent(Event, tag="hand"):
    trigger: HandTrigger
    hand: Hand | None = None
    lengths: dict[str, float] = msgspec.field(default_factory=dict)
    vector: list[float] = msgspec.field(default_factory=list)


class PersonEvent(Event, tag="person"):
    trigger: PersonTrigger


class SceneCommandEvent(Event, tag="scene_command"):
    scene: Scene
    animation: str | None = None
    effects: dict[str, Any] = msgspec.field(default_factory=dict)
    trigger: str | None = None
    text: str | None = None


class EventDoneEvent(Event, tag="event_done"):
    pass


class AnalysisStartedEvent(Event, tag="analysis_started"):
    scene: Scene


class AnalysisResultEvent(Event, tag="analysis_result"):
    text: str
    scene: Scene | None = None


class ErrorEvent(Event, tag="error"):
    message: str


IPEvent: TypeAlias = HandEvent | PersonEvent
AI3DEvent: TypeAlias = (
    SceneCommandEvent
    | EventDoneEvent
    | AnalysisStartedEvent
    | AnalysisResultEvent
    | ErrorEvent
)
WitchEvent: TypeAlias = (
    HandEvent
    | PersonEvent
    | SceneCommandEvent
    | EventDoneEvent
    | AnalysisStartedEvent
    | AnalysisResultEvent
    | ErrorEvent
)
