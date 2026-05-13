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


class AnimationTrigger(str, Enum):
    FINISHED = "animation_finished"


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Scene(str, Enum):
    SCENE_0_IDLE = "scene_0_idle"
    SCENE_1_ATTENTION = "scene_1_attention"
    SCENE_2_INTRO = "scene_2_intro"
    SCENE_2_REFOCUS = "scene_2_refocus"
    SCENE_3_SCAN_READY = "scene_3_scan_ready"
    SCENE_3_HAND_CORRECTION = "scene_3_hand_correction"
    SCENE_3_SCANNING = "scene_3_scanning"
    SCENE_3_SCAN_COMPLETE = "scene_3_scan_complete"
    SCENE_4_TRANSFORMATION = "scene_4_transformation"
    SCENE_5_INTRODUCTION = "scene_5_introduction"
    SCENE_6_SHOT_1_VISUAL = "scene_6_shot_1_visual"
    SCENE_6_SHOT_2_TASK = "scene_6_shot_2_task"
    SCENE_6_SHOT_3_ELEMENT = "scene_6_shot_3_element"
    SCENE_6_SHOT_4_POSITIVE_NEGATIVE = "scene_6_shot_4_positive_negative"
    SCENE_6_SHOT_5_BALANCE = "scene_6_shot_5_balance"
    SCENE_7_RETURN = "scene_7_return"
    SCENE_7_SMOKE_END = "scene_7_smoke_end"
    SCENE_7_VANISH_END = "scene_7_vanish_end"
    END = "end"


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


class AnimationEvent(Event, tag="animation"):
    trigger: AnimationTrigger
    scene: Scene
    animation: str | None = None


class FortuneRequestEvent(Event, tag="fortune_request"):
    pass


class AnalysisStartedEvent(Event, tag="analysis_started"):
    scene: Scene


class AnalysisResultEvent(Event, tag="analysis_result"):
    text: str


class FortuneEvent(Event, tag="fortune"):
    text: str
    audio: str | None = None
    sample_rate: int | None = None


class ErrorEvent(Event, tag="error"):
    message: str


WitchEvent: TypeAlias = (
    HandEvent
    | PersonEvent
    | SceneCommandEvent
    | AnimationEvent
    | FortuneRequestEvent
    | AnalysisStartedEvent
    | AnalysisResultEvent
    | FortuneEvent
    | ErrorEvent
)
