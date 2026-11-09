from __future__ import annotations

from dataclasses import dataclass, field

import msgspec

from shared.events import HandTrigger, Scene

SCAN_DURATION_S = 5.0
SCAN_RUNNING_MAX = 0.99

RING_NEUTRAL = "neutral"
RING_RED = "red"
RING_GREEN = "green"

HAND_PROMPT_SCENES = frozenset(
    {
        Scene.SCENE_4_AWAITING_HAND,
        Scene.SCENE_5_HANDSCAN,
        Scene.SCENE_6_HAND_DETECTED,
        Scene.DEBUG_HAND_TILTED,
        Scene.DEBUG_WRONG_SIDE,
        Scene.DEBUG_NOT_FULLY_IN_VIEW,
        Scene.DEBUG_HAND_PULLED_AWAY,
        Scene.DEBUG_GASLIGHT,
    }
)
SCANNING_SCENES = frozenset({Scene.SCENE_5_HANDSCAN, Scene.SCENE_6_HAND_DETECTED})
SCAN_DONE_SCENES = frozenset(
    {Scene.SCENE_7_HANDSCAN_DONE, Scene.SCENE_8_ANALYSIS, Scene.SCENE_9_OUTRO}
)


class HandVizFrame(msgspec.Struct):
    """One frame from ImageProcessing on /ws/ip-ai-handviz."""

    trigger: HandTrigger
    points: list[list[float]]


@dataclass
class HandVizComposer:
    """Turns scene + live hand status into the Unreal hand stage message `ring|scan|points`."""

    scan_started_at: float | None = field(default=None)

    def compose(
        self,
        scene: Scene,
        trigger: HandTrigger,
        points: list[list[float]],
        now: float,
    ) -> str:
        hand_ok = trigger is HandTrigger.DETECTED
        ring = self._ring(scene, hand_ok)
        scan = self._scan(scene, now)
        point_text = ";".join(f"{x:.4f},{y:.4f}" for x, y in points) if hand_ok else ""
        return f"{ring}|{scan:.3f}|{point_text}"

    @staticmethod
    def _ring(scene: Scene, hand_ok: bool) -> str:
        if scene not in HAND_PROMPT_SCENES:
            return RING_NEUTRAL
        return RING_GREEN if hand_ok else RING_RED

    def _scan(self, scene: Scene, now: float) -> float:
        if scene in SCAN_DONE_SCENES:
            self.scan_started_at = None
            return 1.0
        if scene not in SCANNING_SCENES:
            self.scan_started_at = None
            return 0.0
        if self.scan_started_at is None:
            self.scan_started_at = now
        return min(SCAN_RUNNING_MAX, (now - self.scan_started_at) / SCAN_DURATION_S)
