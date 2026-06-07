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
from .speech_pipeline import AudioPlaybackConfig, SpeechPipeline, normalize_llm_text
from .state_machine.state_machine import (
    StateChange,
    WitchStateMachine,
)
from .websocket_server.websocket_server import WebSocketServer

logger = logging.getLogger("ai")


def _wait_for_unreal_ack() -> bool:
    return os.environ.get("WITCH_WAIT_FOR_UNREAL_ACK", "true").strip().lower() == "true"


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
        audio_config: AudioPlaybackConfig | None = None,
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
        self._seat_override_task: asyncio.Task[None] | None = None
        self._pending_person_event: PersonEvent | None = None
        self._pending_unreal_ack: PendingUnrealAck | None = None
        self._queued_speech: tuple[str, Scene] | None = None
        self._analysis_started = False
        self._desired_hand: Hand | None = None
        self._hand_was_present_in_awaiting: bool = False
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
            audio_config=audio_config,
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
            if self._state_machine._get_state_def(None).hand_locked:
                logger.info(
                    "Ignoring hand event after analysis lock: state=%s trigger=%s",
                    self._state_machine.state,
                    event.trigger,
                )
                return

            logger.info("Broadcasting hand event: %s", event.trigger)

            if self._hand_reset_required:
                if event.trigger in (HandTrigger.ABSENT, HandTrigger.NOT_FULLY_IN_VIEW):
                    logger.info(
                        "Hand removed after gaslight; next insertion may continue normally"
                    )
                    self._hand_reset_required = False
                    self._latest_hand_event = event
                    if event.vector:
                        self._analysis_hand_event = event
                    return
                elif event.trigger not in (HandTrigger.ABSENT, HandTrigger.NOT_FULLY_IN_VIEW):
                    logger.info(
                        "Ignoring hand event until the visitor removes the gaslit hand: trigger=%s",
                        event.trigger,
                    )
                    return

            if (
                self._state_machine.state == Scene.SCENE4.value
                and event.hand is not None
                and event.trigger not in (HandTrigger.ABSENT, HandTrigger.NOT_FULLY_IN_VIEW)
                and os.environ.get("WITCH_GASLIGHT", "true").strip().lower() == "true"
                and (
                    self._desired_hand is None
                    or (
                        event.trigger is HandTrigger.DETECTED
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

                changes = self._state_machine.advance(HandTrigger.DETECTED.value)
                if changes:
                    scene_changed = await self._emit_scene_changes(changes)
                    if scene_changed:
                        await self._on_state_changed()
                        await self._start_speech_for_current_scene(restart=True)
                return

            self._latest_hand_event = event
            if event.vector:
                self._analysis_hand_event = event

            if self._state_machine.state == Scene.SCENE4.value:
                if event.trigger not in (HandTrigger.ABSENT, HandTrigger.NOT_FULLY_IN_VIEW):
                    self._hand_was_present_in_awaiting = True
                elif event.trigger in (HandTrigger.ABSENT, HandTrigger.NOT_FULLY_IN_VIEW) and not self._hand_was_present_in_awaiting:
                    logger.info("Ignoring hand_absent in awaiting_hand: hand was never present")
                    return

            scene_changed = await self._emit_scene_changes(
                self._state_machine.advance(event.trigger.value)
            )
            if scene_changed:
                await self._on_state_changed()
                await self._start_speech_for_current_scene(restart=True)
            return

        if isinstance(event, PersonEvent):
            if (
                event.trigger is PersonTrigger.SEATED
                and self._state_machine.state == Scene.SCENE1.value
                and self._speech_task is not None
                and not self._speech_task.done()
            ):
                logger.info("Deferring seated event until scene 0 speech completes")
                self._pending_person_event = event
                return

            self._forbidden_hand = None
            self._hand_gaslight_pending = False
            self._hand_reset_required = False
            self._gaslight_hand_name = None
            self._desired_hand = None

            if self._state_machine._get_state_def(None).hand_locked and event.trigger is not PersonTrigger.ABSENT:
                logger.info(
                    "Ignoring person event during locked flow: state=%s trigger=%s",
                    self._state_machine.state,
                    event.trigger,
                )
                return

        scene_changed = await self._emit_scene_changes(
            self._state_machine.advance(event.trigger.value)
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
        if scene is Scene.SCENE4:
            return None

        if scene is Scene.SCENE5 and self._hand_gaslight_pending:
            base_text = (
                f"Du hast mir deine {self._gaslight_hand_name or 'unbekannte'} Hand gezeigt. "
                "Genau diese ist die falsche. Nimm die andere Hand."
            )
            return build_scene_prompt(scene, base_text=base_text)

        if scene is Scene.SCENE6:
            return build_combined_analysis_prompt(
                self._analysis_hand_event or self._latest_hand_event,
            )

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
            text = normalize_llm_text(
                await self._speech_pipeline.generate_text(prompt)
            )
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
            if self._state_machine.state == scene.value:
                logger.info("Holding %s until speech can be played", scene.value)
                await asyncio.sleep(2)
                await self._start_speech_for_current_scene(restart=True)
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

        if (
            self._state_machine._get_state_def(scene.value).is_analysis
            and text is not None
        ):
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

    async def _replay_seat_sensor_override(self) -> None:
        try:
            for trigger in (PersonTrigger.DETECTED, PersonTrigger.SEATED):
                await asyncio.sleep(10)
                event = PersonEvent(trigger=trigger)
                logger.info("Seat sensor override after reset: %s", trigger.value)
                await self._process_ip_event(event)
                await self._ip_channel.broadcast(event)
        except asyncio.CancelledError:
            logger.info("Cancelled pending seat sensor override replay")
            raise
        finally:
            if self._seat_override_task is asyncio.current_task():
                self._seat_override_task = None

    async def _restart_seat_sensor_override(self) -> None:
        task = self._seat_override_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        self._seat_override_task = asyncio.create_task(
            self._replay_seat_sensor_override()
        )

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
        self._pending_person_event = None
    async def _on_state_changed(self) -> None:
        if self._state_machine.state == Scene.SCENE0.value:
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
            self._pending_person_event = None
            self._hand_was_present_in_awaiting = False
            self._state_machine.manual_mode = False
            self._state_machine.previous_state = None
            return

        if self._state_machine.state == Scene.SCENE4.value:
            self._hand_was_present_in_awaiting = False

        if (
            self._state_machine.state == Scene.SCENE5.value
            and not self._analysis_started
        ):
            self._analysis_started = True
            await self._emit_ai3d_event(
                AnalysisStartedEvent(scene=Scene.SCENE5),
            )

        if (
            self._state_machine._get_state_def(None).hand_locked
            and self._analysis_hand_event is None
            and self._latest_hand_event is not None
            and self._latest_hand_event.vector
        ):
            self._analysis_hand_event = self._latest_hand_event

    async def _advance_scene_after_speech(self, scene: Scene) -> None:
        if self._state_machine.state != scene.value:
            return

        if scene is Scene.SCENE5 and self._hand_gaslight_pending:
            self._hand_gaslight_pending = False
            self._latest_hand_event = None
            changes = self._state_machine.advance("restart_hand_prompt")
            if changes:
                await self._emit_scene_changes(changes)
                await self._on_state_changed()
                await self._emit_scene_events_on_audio_start(
                    Scene.SCENE4,
                    None,
                )
            return

        if self._state_machine._get_state_def(scene.value).auto_trigger is not None:
            changes = self._state_machine.event_done(scene.value)
            if changes:
                await self._emit_scene_changes(changes)
                await self._on_state_changed()
                await self._start_speech_for_current_scene(restart=True)
                return

        if await self._replay_pending_person_event():
            return
        await self._replay_latest_hand_event()

    async def _replay_pending_person_event(self) -> bool:
        event = self._pending_person_event
        if event is None:
            return False
        self._pending_person_event = None
        logger.info("Processing deferred person event: %s", event.trigger.value)
        await self._process_ip_event(event)
        return True

    async def _replay_latest_hand_event(self) -> bool:
        current_state = self._state_machine.state
        if not (not self._state_machine._get_state_def(current_state).hand_locked and not current_state.startswith("scene_debug_")):
            return False
        if self._state_machine._get_state_def(None).hand_locked:
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
            changes = self._state_machine.advance(event)
            logger.info(
                "trigger_state_event: %s -> %s (changes=%s)",
                self._state_machine.state,
                event,
                changes,
            )
            if event == "reset":
                if not changes:
                    self._state_machine.force_state(Scene.SCENE0.value)
                    logger.info("Reset: force_state fallback used")
                change = changes[-1] if changes else None
                self._pending_scene_command = self._scene_command(
                    change or StateChange(
                        trigger="reset", source="", dest=Scene.SCENE0.value
                    ),
                )
                await self._emit_scene_events_on_audio_start(Scene.SCENE0, None)
                await self._on_state_changed()
                await self._cancel_speech()
                if os.getenv("WITCH_SEAT_SENSOR_OVERRIDE", "false") == "true":
                    await self._restart_seat_sensor_override()
        return self._state_machine.state
