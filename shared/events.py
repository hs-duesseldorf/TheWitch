from __future__ import annotations

from enum import Enum
from typing import Any, TypeAlias

import msgspec


# Collection of Classes concerning Triggers and Events
# defining all possible Properties and their respective strings 
# f.e. a Hand can either have the Property of being the right or left Hand

# Mostly used by the StateMachine and Sensor-Inputs sent via Websockets

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
    DEBUG_HAND_TILTED = "scene_debug_shot_2_hand_tilted"
    DEBUG_WRONG_SIDE = "scene_debug_shot_3_hand_wrong_side"
    DEBUG_NOT_FULLY_IN_VIEW = "scene_debug_shot_4_hand_not_fully_in_view"
    DEBUG_HAND_PULLED_AWAY = "scene_debug_shot_5_hand_pulled_away"
    DEBUG_GASLIGHT = "scene_debug_gaslight"

    SCENE_0_IDLE = "scene_0_idle"
    SCENE_1_WELCOME = "scene_1_welcome"
    SCENE_2_SEATED = "scene_2_seated"
    SCENE_3_INTRO = "scene_3_intro"
    SCENE_4_AWAITING_HAND = "scene_4_awaiting_hand"
    SCENE_5_HANDSCAN = "scene_5_handscan"
    SCENE_6_HAND_DETECTED = "scene_6_hand_detected"
    SCENE_7_HANDSCAN_DONE = "scene_7_handscan_done"
    SCENE_8_ANALYSIS = "scene_8_analysis"
    SCENE_9_OUTRO = "scene_9_outro"


# The following Events contain the necessary Information needed for further processes triggered by them
# f.e. every Event can hold Information about it's origin - where it was triggered from

class Event(
    msgspec.Struct, frozen=True, kw_only=True, tag_field="type", omit_defaults=True
):
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


# These Alias are used to determine which kind of Event needs to listen to or work with what kind of Event
# f.e. IPEvents can only be those concerning the Hand or Person

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
