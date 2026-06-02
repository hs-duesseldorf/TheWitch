from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias

import msgspec

# Triggers & identifiers


class HandTrigger(str, Enum):
    DETECTED = "hand_detected"
    ABSENT = "hand_absent"
    WRONG_SIDE = "hand_wrong_side"
    NOT_FULLY_IN_VIEW = "hand_not_fully_in_view"
    TILTED = "hand_tilted"


class PersonTrigger(str, Enum):
    DETECTED = "person_detected"
    ABSENT = "person_absent"


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Scene(str, Enum):
    #Shortened Statemachine

    # DEBUG / HAND DETECTION
    SCENE_DEBUG_SHOT_1_HAND_ABSENT = "scene_debug_shot_1_hand_absent"
    SCENE_DEBUG_SHOT_2_HAND_TILTED = "scene_debug_shot_2_hand_tilted"
    SCENE_DEBUG_SHOT_3_HAND_WRONG_SIDE = "scene_debug_shot_3_hand_wrong_side"
    # OTHER SCENES
    SCENE_0_IDLE = "scene_0_idle"
    SCENE_6_OUTRO = "scene_6_outro"
    SCENE_RESTART = "scene_restart"
    # SCENE 1 - Welcome / instructions
    SCENE_1_START = "scene_1_start"
    # SCENE 2 - Hand inside stone
    SCENE_2_AWAITING_HAND = "scene_2_awaiting_hand"
    SCENE_2_HAND_FOUND = "scene_2_hand_found"
    # SCENE 3 - Handscan
    SCENE_3_HANDSCAN_IN_PROCESS = "scene_3_handscan_in_process"
    SCENE_3_HANDSCAN_DONE = "scene_3_handscan_done"
    # SCENE 4 - Transformation
    SCENE_4_TRANSFORMATION = "scene_4_transformation"
    # SCENE 5 - Hand reading
    SCENE_5_HANDREAD_VISUALISATION = "scene_5_handread_visualisation"
    SCENE_5_SHOT_1_CORE_ELEMENT = "scene_5_shot_1_core_element"
    SCENE_5_SHOT_2_WEAK_ELEMENT = "scene_5_shot_2_weak_element"
    SCENE_5_SHOT_3_ADVICE = "scene_5_shot_3_advice"


# Events
# `tag_field` names the JSON discriminator key ("type").
# `tag` sets its value for each subclass.
# Use isinstance() / match to branch on event type after decoding.


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
