from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from shared.events import (
    AnalysisResultEvent,
    AnalysisStartedEvent,
    ErrorEvent,
    EventDoneEvent,
    Hand,
    HandEvent,
    HandTrigger,
    IPEvent,
    PersonEvent,
    PersonTrigger,
    Scene,
    SceneCommandEvent,
)

from .hand_analysis import (
    build_combined_analysis_prompt,
    build_scene_prompt,
    get_scene_base_text,
)
from .message_channels import EventChannel
from .speech_pipeline import SpeechPipeline
from .state_machine.state_machine import (
    ANALYSIS_SCENES,
    AUTO_ADVANCE_STATES,
    HAND_LOCKED_STATES,
    HAND_REPLAY_STATES,
    PERSON_LOCKED_STATES,
    WAIT_FOR_UNREAL_STATES,
    StateChange,
    WitchStateMachine,
    hand_condition,
)
from .websocket_server.websocket_server import WebSocketServer

logger = logging.getLogger("ai")


def _wait_for_unreal_ack() -> bool:
    return os.environ.get("WITCH_WAIT_FOR_UNREAL_ACK", "true").strip().lower() == "true"


def _gaslight_enabled() -> bool:
    return os.environ.get("WITCH_GASLIGHT", "").strip().lower() == "true"


OUTBOUND_AI3D_EVENT = (
    SceneCommandEvent | AnalysisStartedEvent | AnalysisResultEvent | ErrorEvent
)


@dataclass(frozen=True)
class PendingUnrealAck:
    event_type: str
    scene: str


class WitchRuntime:
    def __init__(
        self,
        *,
        ws_server: WebSocketServer,
        state_machine: WitchStateMachine,
        llm: Any,
        tts: Any,
    ):
        self._ws_server = ws_server
        self._state_machine = state_machine
        self._latest_hand_event: HandEvent | None = None
        self._analysis_hand_event: HandEvent | None = None
        self._forbidden_hand: Hand | None = None
        self._hand_gaslight_pending = False
        self._hand_reset_required = False
        self._gaslight_hand_name: str | None = None
        self._pending_scene_command: SceneCommandEvent | None = None
        self._speech_scene: Scene | None = None
        self._speech_task: asyncio.Task[None] | None = None
        self._speech_loop: asyncio.AbstractEventLoop | None = None
        self._pending_unreal_ack: PendingUnrealAck | None = None
        self._queued_speech: tuple[str, Scene] | None = None
        self._analysis_started = False
        self._desired_hand: Hand | None = None
        self._ip_channel = EventChannel(
            ws_server=ws_server,
            path="/ws/ip-ai",
            default_origin="ImageProcessing",
            decode_source="ip",
        )
        self._ai3d_channel = EventChannel(
            ws_server=ws_server,
            path="/ws/ai-3d",
            default_origin="ArtificialIntelligence",
            decode_source="ai-3d",
        )

        self._speech_pipeline = SpeechPipeline(
            llm=llm,
            tts=tts,
            tts_seed=int(os.environ.get("WITCH_TTS_SEED", "42")),
        )

        self._register_routes()

    def _register_routes(self) -> None:
        self._ws_server.add_route("/ws/ip-ai", self._on_ip_ai_message)
        self._ws_server.add_route("/ws/ip-ai-video", self._on_ip_ai_video_message)
        self._ws_server.add_route("/ws/ip-roi", self._on_ip_roi_message)
        self._ws_server.add_route("/ws/ai-3d", self._on_ai_3d_message)
        self._ws_server.add_route("/ws/ai-3d-video", None)
        self._ws_server.add_route("/ws/ai-3d-roi", None)

    async def _on_ip_ai_message(
        self, server: WebSocketServer, connection: Any, message: str | bytes
    ) -> None:
        event = self._ip_channel.decode(message)
        if not isinstance(event, (HandEvent, PersonEvent)):
            return

        if isinstance(event, HandEvent) and self._state_machine.manual_mode:
            return

        await self._process_ip_event(event)
        await self._ip_channel.broadcast(event)

    async def _process_ip_event(self, event: IPEvent) -> None:
        if isinstance(event, HandEvent):
            if self._state_machine.state in HAND_LOCKED_STATES:
                logger.info(
                    "Ignoring hand event after analysis lock: state=%s trigger=%s",
                    self._state_machine.state,
                    event.trigger,
                )
                return

            logger.info("Broadcasting hand event: %s", event.trigger)

            if self._hand_reset_required:
                if event.trigger in {HandTrigger.ABSENT, HandTrigger.NOT_FULLY_IN_VIEW}:
                    logger.info(
                        "Hand removed after gaslight; next insertion may continue normally"
                    )
                    self._hand_reset_required = False
                    self._latest_hand_event = event
                    if event.vector:
                        self._analysis_hand_event = event
                    return
                elif hand_condition(event) in {"present", "ready"}:
                    logger.info(
                        "Ignoring hand event until the visitor removes the gaslit hand: trigger=%s",
                        event.trigger,
                    )
                    return

            if (
                self._state_machine.state == Scene.SCENE_2_AWAITING_HAND.value
                and event.hand is not None
                and hand_condition(event) not in {"absent"}
                and _gaslight_enabled()
                and (
                    self._desired_hand is None
                    or (
                        hand_condition(event) in {"present", "ready"}
                        and event.hand is not self._desired_hand
                    )
                )
            ):
                if self._desired_hand is None:
                    self._forbidden_hand = event.hand
                    self._desired_hand = (
                        Hand.LEFT if event.hand is Hand.RIGHT else Hand.RIGHT
                    )
                self._hand_gaslight_pending = True
                self._hand_reset_required = True
                self._gaslight_hand_name = self._hand_name(event)

                change = self._state_machine.advance("ip_hand_present")
                if change is not None:
                    scene_changed = await self._emit_scene_changes([change])
                    if scene_changed:
                        await self._on_state_changed()
                        await self._start_speech_for_current_scene(restart=True)
                return

            self._latest_hand_event = event
            if event.vector:
                self._analysis_hand_event = event

            scene_changed = await self._emit_scene_changes(
                self._state_machine.hand_event(event)
            )
            if scene_changed:
                await self._on_state_changed()
                await self._start_speech_for_current_scene(restart=True)
            return

        if isinstance(event, PersonEvent):
            self._forbidden_hand = None
            self._hand_gaslight_pending = False
            self._hand_reset_required = False
            self._gaslight_hand_name = None
            self._desired_hand = None

            if self._state_machine.state in PERSON_LOCKED_STATES:
                logger.info(
                    "Ignoring person event during locked flow: state=%s trigger=%s",
                    self._state_machine.state,
                    event.trigger,
                )
                return

        scene_changed = await self._emit_scene_changes(
            self._state_machine.person_event(event)
        )
        if scene_changed:
            await self._on_state_changed()
            await self._start_speech_for_current_scene(restart=True)

    async def _emit_scene_changes(self, changes: list[StateChange]) -> bool:
        if not changes:
            return False

        self._pending_unreal_ack = None
        self._pending_scene_command = self._scene_command(changes[-1])
        return True

    def _scene_command(self, change: StateChange) -> SceneCommandEvent:
        return SceneCommandEvent(
            scene=Scene(change.dest),
            animation=change.dest,
            effects={},
            trigger=change.trigger,
        )

    def _speech_prompt_for_scene(self, scene: Scene) -> str | None:
        if scene is Scene.SCENE_2_HAND_FOUND and self._hand_gaslight_pending:
            base_text = (
                f"Du hast mir deine {self._gaslight_hand_name or 'unbekannte'} Hand gezeigt. "
                "Genau diese ist die falsche. Nimm die andere Hand."
            )
            return build_scene_prompt(scene, base_text=base_text)

        if scene is Scene.SCENE_5_HANDREAD_VISUALISATION:
            return build_combined_analysis_prompt(
                self._analysis_hand_event or self._latest_hand_event,
            )

        if scene in {Scene.SCENE_5_SHOT_1_CORE_ELEMENT, Scene.SCENE_5_SHOT_2_WEAK_ELEMENT, Scene.SCENE_5_SHOT_3_ADVICE}:
            return None

        base_text = get_scene_base_text(scene)
        if not base_text:
            return None
        return build_scene_prompt(scene, base_text=base_text)

    async def _start_speech_for_current_scene(self, *, restart: bool) -> None:
        scene = Scene(self._state_machine.state)
        prompt = self._speech_prompt_for_scene(scene)
        if prompt is None:
            await self._emit_scene_events_on_audio_start(scene, None)
            await self._advance_scene_after_speech(scene)
            return

        if self._speech_task is not None and not self._speech_task.done():
            if not restart and self._speech_scene == scene:
                return
            self._queued_speech = (prompt, scene)
            return

        loop = asyncio.get_running_loop()
        self._speech_scene = scene
        self._speech_loop = loop
        self._speech_task = loop.create_task(self._run_speech(prompt, scene))

    def _check_queued_speech(self) -> None:
        if self._queued_speech is None:
            return
        prompt, scene = self._queued_speech
        self._queued_speech = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._speech_scene = scene
        self._speech_loop = loop
        self._speech_task = loop.create_task(self._run_speech(prompt, scene))

    async def _run_speech(self, prompt: str, scene: Scene) -> None:
        try:
            text = await self._speech_pipeline.generate_text(prompt)
            await self._speech_pipeline.play_text(
                text,
                on_audio_start=lambda: self._emit_scene_events_on_audio_start(
                    scene,
                    text,
                ),
            )
            current = asyncio.current_task()
            if self._speech_task is current:
                self._speech_task = None
                self._speech_loop = None
            if self._speech_scene == scene:
                self._speech_scene = None
            await self._advance_scene_after_speech(scene)
        except asyncio.CancelledError:
            logger.info("Speech task cancelled for %s", scene.value)
            raise
        except Exception as exc:
            logger.exception("Speech failed for %s", scene.value)
            await self._emit_ai3d_event(
                ErrorEvent(message=f"Speech failed for {scene.value}: {exc}"),
            )
            await self._advance_scene_after_speech(scene)
        finally:
            current = asyncio.current_task()
            if self._speech_task is current:
                self._speech_task = None
                self._speech_loop = None
            if self._speech_scene == scene:
                self._speech_scene = None
            self._check_queued_speech()

    async def _on_ip_ai_video_message(
        self, server: WebSocketServer, connection: Any, message: str | bytes
    ) -> None:
        if isinstance(message, bytes):
            await self._ws_server.broadcast(message, path="/ws/ai-3d-video")

    async def _on_ip_roi_message(
        self, server: WebSocketServer, connection: Any, message: str | bytes
    ) -> None:
        if isinstance(message, bytes):
            await self._ws_server.broadcast(message, path="/ws/ai-3d-roi")

    async def _on_ai_3d_message(
        self, server: WebSocketServer, connection: Any, message: str | bytes
    ) -> None:
        event = self._ai3d_channel.decode(message)
        if not isinstance(event, EventDoneEvent):
            return

        await self._process_unreal_event(event)

    async def _process_unreal_event(self, event: EventDoneEvent) -> None:
        pending = self._pending_unreal_ack
        if pending is None:
            logger.info("Ignoring event_done without a pending Unreal acknowledgement")
            return

        changes = self._state_machine.event_done(pending.scene)
        self._pending_unreal_ack = None
        if not changes:
            logger.info(
                "Ignoring stale event_done for %s on %s while current state is %s",
                pending.event_type,
                pending.scene,
                self._state_machine.state,
            )
            return

        await self._emit_scene_changes(changes)
        await self._on_state_changed()
        await self._start_speech_for_current_scene(restart=True)

    async def _emit_ai3d_event(self, event: OUTBOUND_AI3D_EVENT) -> None:
        await self._ai3d_channel.broadcast(event)
        self._track_pending_unreal_ack(event)

    async def _emit_scene_events_on_audio_start(
        self,
        scene: Scene,
        text: str | None,
    ) -> None:
        pending_scene_command = self._pending_scene_command
        if pending_scene_command is not None and pending_scene_command.scene is scene:
            if text is not None:
                pending_scene_command = SceneCommandEvent(
                    scene=pending_scene_command.scene,
                    animation=pending_scene_command.animation,
                    effects=pending_scene_command.effects,
                    trigger=pending_scene_command.trigger,
                    text=text,
                )
            await self._emit_ai3d_event(pending_scene_command)
            self._pending_scene_command = None

        if scene in ANALYSIS_SCENES and text is not None:
            await self._emit_ai3d_event(
                AnalysisResultEvent(text=text, scene=scene),
            )

    def _track_pending_unreal_ack(self, event: OUTBOUND_AI3D_EVENT) -> None:
        if isinstance(event, ErrorEvent):
            return

        if isinstance(event, SceneCommandEvent):
            event_type = "scene_command"
            scene = event.scene.value
        elif isinstance(event, AnalysisStartedEvent):
            event_type = "analysis_started"
            scene = event.scene.value
        else:
            event_type = "analysis_result"
            scene = event.scene.value if event.scene is not None else None

        if scene is None:
            return
        if scene != self._state_machine.state:
            return
        if not _wait_for_unreal_ack():
            return
        if scene not in WAIT_FOR_UNREAL_STATES:
            return

        self._pending_unreal_ack = PendingUnrealAck(
            event_type=event_type,
            scene=scene,
        )

    async def simulate_hand_event(self, event: str) -> str:
        try:
            trigger_enum = HandTrigger(event)
        except ValueError:
            raise ValueError(f"Ungültiger HandTrigger: {event}")

        hand_event = HandEvent(
            trigger=trigger_enum,
            origin="ManualSimulation",
            hand=None,
            lengths={},
            vector=[],
        )

        await self._process_ip_event(hand_event)
        return self._state_machine.state

    async def simulate_person_event(self, event: str) -> str:
        try:
            trigger_enum = PersonTrigger(event)
        except ValueError:
            raise ValueError(f"Ungültiger PersonTrigger: {event}")

        person_event = PersonEvent(
            trigger=trigger_enum,
            origin="ManualSimulation",
        )

        await self._process_ip_event(person_event)
        return self._state_machine.state

    async def acknowledge_unreal_event(self) -> str:
        await self._process_unreal_event(EventDoneEvent(origin="ManualSimulation"))
        return self._state_machine.state

    @property
    def state(self) -> str:
        return self._state_machine.state

    @property
    def speech_stage(self) -> str:
        return self._speech_pipeline.stage

    @property
    def analysis_stage(self) -> str:
        return self.speech_stage

    def set_manual_mode(self, enabled: bool) -> bool:
        self._state_machine.manual_mode = enabled
        return enabled

    def _clear_runtime_state(self) -> None:
        self._latest_hand_event = None
        self._analysis_hand_event = None
        self._forbidden_hand = None
        self._hand_gaslight_pending = False
        self._hand_reset_required = False
        self._gaslight_hand_name = None
        self._pending_scene_command = None
        self._speech_scene = None
        self._pending_unreal_ack = None
        self._queued_speech = None
        self._analysis_started = False
        self._desired_hand = None
    async def _on_state_changed(self) -> None:
        if self._state_machine.state == Scene.SCENE_0_IDLE.value:
            self._latest_hand_event = None
            self._analysis_hand_event = None
            self._forbidden_hand = None
            self._hand_gaslight_pending = False
            self._hand_reset_required = False
            self._gaslight_hand_name = None
            self._pending_unreal_ack = None
            self._speech_scene = None
            self._queued_speech = None
            self._analysis_started = False
            self._desired_hand = None
            self._state_machine.manual_mode = False
            self._state_machine.previous_state = None
            return

        if (
            self._state_machine.state == Scene.SCENE_3_HANDSCAN_DONE.value
            and not self._analysis_started
        ):
            self._analysis_started = True
            await self._emit_ai3d_event(
                AnalysisStartedEvent(scene=Scene.SCENE_3_HANDSCAN_DONE),
            )

        if (
            self._state_machine.state in HAND_LOCKED_STATES
            and self._analysis_hand_event is None
            and self._latest_hand_event is not None
            and self._latest_hand_event.vector
        ):
            self._analysis_hand_event = self._latest_hand_event

    async def _advance_scene_after_speech(self, scene: Scene) -> None:
        if self._state_machine.state != scene.value:
            return

        if scene is Scene.SCENE_2_HAND_FOUND and self._hand_gaslight_pending:
            self._hand_gaslight_pending = False
            self._latest_hand_event = None
            change = self._state_machine.advance("restart_hand_prompt")
            if change is not None:
                await self._emit_scene_changes([change])
                await self._on_state_changed()
                await self._emit_scene_events_on_audio_start(
                    Scene.SCENE_2_AWAITING_HAND,
                    None,
                )
            return

        if scene.value in AUTO_ADVANCE_STATES:
            changes = self._state_machine.event_done(scene.value)
            if changes:
                await self._emit_scene_changes(changes)
                await self._on_state_changed()
                await self._start_speech_for_current_scene(restart=True)
                if (
                    self._state_machine.state == Scene.SCENE_0_IDLE.value
                    and os.environ.get("WITCH_SEAT_SENSOR_OVERRIDE", "").strip().lower()
                    == "true"
                ):
                    logger.info(
                        "Reset after outro: seat sensor override active, re-emitting person seated event"
                    )
                    await self._process_ip_event(
                        PersonEvent(trigger=PersonTrigger.DETECTED)
                    )
                return

        await self._replay_latest_hand_event()

    async def _replay_latest_hand_event(self) -> bool:
        if self._state_machine.state not in HAND_REPLAY_STATES:
            return False
        if self._state_machine.state in HAND_LOCKED_STATES:
            return False
        if self._latest_hand_event is None:
            return False

        logger.info(
            "Replaying cached hand event through process_ip_event: %s",
            self._latest_hand_event.trigger,
        )
        await self._process_ip_event(self._latest_hand_event)
        return True

    def _hand_name(self, event: HandEvent) -> str:
        if event.hand is Hand.LEFT:
            return "linke"
        if event.hand is Hand.RIGHT:
            return "rechte"
        return "unbekannte"

    @staticmethod
    def _hand_name_from_hand(hand: Hand) -> str:
        if hand is Hand.LEFT:
            return "linke"
        if hand is Hand.RIGHT:
            return "rechte"
        return "unbekannte"

    async def _cancel_speech(self) -> None:
        task = self._speech_task
        self._speech_task = None
        self._speech_loop = None
        self._speech_scene = None
        self._queued_speech = None

        self._speech_pipeline.stop_player()
        if task is None or task.done():
            return

        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

    def _cancel_speech_sync(self) -> None:
        task = self._speech_task
        loop = self._speech_loop
        self._speech_task = None
        self._speech_loop = None
        self._speech_scene = None
        self._queued_speech = None

        self._speech_pipeline.stop_player()
        if task is None or task.done():
            return

        if loop and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)
        else:
            task.cancel()

    def force_state(self, state: str) -> str:
        self._state_machine.force_state(state)
        self._clear_runtime_state()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._cancel_speech())
        except RuntimeError:
            self._cancel_speech_sync()
        return self._state_machine.state

    async def trigger_state_event(self, event: str) -> str:
        if event:
            change = self._state_machine.advance(event)
            logger.info(
                "trigger_state_event: %s -> %s (change=%s)",
                self._state_machine.state,
                event,
                change,
            )
            if event == "reset":
                if change is None:
                    self._state_machine.force_state(Scene.SCENE_0_IDLE.value)
                    logger.info("Reset: force_state fallback used")
                self._pending_scene_command = self._scene_command(
                    StateChange(
                        trigger="reset", source="", dest=Scene.SCENE_0_IDLE.value
                    ),
                )
                await self._emit_scene_events_on_audio_start(Scene.SCENE_0_IDLE, None)
                await self._on_state_changed()
                await self._cancel_speech()
                if (
                    os.environ.get("WITCH_SEAT_SENSOR_OVERRIDE", "").strip().lower()
                    == "true"
                ):
                    logger.info(
                        "Reset: seat sensor override active, re-emitting person seated event"
                    )
                    await self._process_ip_event(
                        PersonEvent(trigger=PersonTrigger.DETECTED)
                    )
        return self._state_machine.state
