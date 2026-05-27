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
    # DEBUG / HAND DETECTION
    SCENE_DEBUG_HAND_DETECTION = "scene_debug_hand_detection"
    SCENE_DEBUG_SHOT_1_HAND_ABSENT = "scene_debug_shot_1_hand_absent"
    SCENE_DEBUG_SHOT_2_HAND_MOVING = "scene_debug_shot_2_hand_moving"
    SCENE_DEBUG_SHOT_3_HAND_TILTED = "scene_debug_shot_3_hand_tilted"
    SCENE_DEBUG_SHOT_4_HAND_OUTSIDE_FRAME = "scene_debug_shot_4_hand_not_fully_in_view"
    SCENE_DEBUG_SHOT_5_HAND_WRONG_SIDE = "scene_debug_shot_5_hand_wrong_side"
    # START / IDLE
    SCENE_0_IDLE = "scene_0_idle"
    # WELCOME
    SCENE_1_AWAITING_HAND = "scene_1_awaiting_hand"
    SCENE_1_SHOT_2_YES_HAND_FOUND = "scene_1_shot_2_yes_hand_found"
    # INTRO
    SCENE_2_SHOT_1_HAND_STAYS_FOCUSED = "scene_2_shot_1_hand_stays_focused"
    # SCAN
    SCENE_3_SCANNING_HAND = "scanning_hand"
    SCENE_3_SHOT_1_CORRECT_HAND = "scene_3_shot_1_correct_hand"
    SCENE_3_SHOT_4_SCAN_DONE = "scene_3_shot_4_scan_done"
    # TRANSFORM
    SCENE_4_TRANSFORM = "scene_4_transform"
    # STORY
    SCENE_5_WITCH_ORIGIN_STORY = "scene_5_witch_origin_story"
    # READING
    SCENE_6_VISUAL_IMAGE_HAND = "scene_6_visual_image_hand"
    SCENE_6_SHOT_1_POINT_OUT_DETAILS = "scene_6_shot_1_point_out_details"
    SCENE_6_SHOT_2_INTERACTIVE_TASK = "scene_6_shot_2_interactive_task"
    SCENE_6_SHOT_2_1_TASK_DONE = "scene_6_shot_2_1_task_done"
    SCENE_6_SHOT_2_2_TASK_IGNORED = "scene_6_shot_2_2_task_ignored"
    SCENE_6_SHOT_3_ASSIGN_ELEMENTS = "scene_6_shot_3_assign_elements"
    SCENE_6_SHOT_4_ELEMENT_ANALYSIS = "scene_6_shot_4_element_analysis"
    SCENE_6_SHOT_5_INNER_BALANCE = "scene_6_shot_5_inner_balance"
    # END
    SCENE_7_LAST_WORDS = "scene_7_last_words"
    SCENE_7_SHOT_1_RETURN_TO_IDLE = "scene_7_shot_1_return_to_idle"
    SCENE_7_SHOT_2_DISAPPEAR = "scene_7_shot_2_disappear"
    # RESTART
    SCENE_RESTART = "scene_restart"


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


class EventDoneEvent(Event, tag="event_done"):
    pass


class AnalysisStartedEvent(Event, tag="analysis_started"):
    scene: Scene


class AnalysisResultEvent(Event, tag="analysis_result"):
    text: str


class FortuneEvent(Event, tag="fortune"):
    text: str
    sample_rate: int | None = None


class ErrorEvent(Event, tag="error"):
    message: str


WitchEvent: TypeAlias = (
    HandEvent
    | PersonEvent
    | SceneCommandEvent
    | EventDoneEvent
    | AnalysisStartedEvent
    | AnalysisResultEvent
    | FortuneEvent
    | ErrorEvent
)
