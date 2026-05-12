from __future__ import annotations

import asyncio
import base64
import copy
import logging
from typing import Any

import httpx
import msgspec

from message_channels import EventChannel
from state_machine.state_machine import (
    SCENES_THAT_DELIVER_FORTUNE,
    SCENES_THAT_START_ANALYSIS,
    StateChange,
    WitchStateMachine,
)
from shared.events import (
    AnalysisStartedEvent,
    AnimationEvent,
    AnimationTrigger,
    ErrorEvent,
    FortuneEvent,
    FortuneRequestEvent,
    HandEvent,
    Scene,
    SceneCommandEvent,
    WitchEvent,
)
from websocket_server.websocket_server import WebSocketServer

logger = logging.getLogger(__name__)


class WitchRuntime:
    def __init__(
        self,
        *,
        ws_server: WebSocketServer,
        state_machine: WitchStateMachine,
        llm: Any,
        tts: Any,
    ):
        self.ws_server = ws_server
        self.state_machine = state_machine
        self.llm = llm
        self.tts = tts
        self._analysis_task: asyncio.Task | None = None
        self._analysis_stage = "idle"
        self._hand_data: dict | None = None
        self._pending_fortune_event: FortuneEvent | None = None
        self._ip_channel = EventChannel(
            ws_server=self.ws_server,
            path="/ws/ip-ai",
            default_origin="ImageProcessing",
            decode_source="ip",
        )
        self._ai3d_channel = EventChannel(
            ws_server=self.ws_server,
            path="/ws/ai-3d",
            default_origin="ArtificialIntelligence",
            decode_source="ai-3d",
        )
        self._ip_handlers = {
            HandEvent: self._handle_ip_hand_event,
        }
        self._unreal_handlers = {
            AnimationEvent: self._handle_unreal_animation_event,
            FortuneRequestEvent: self._handle_unreal_fortune_request,
        }

        self._register_routes()

    def _register_routes(self) -> None:
        self.ws_server.add_route("/ws/ip-ai", self._on_ip_ai_message)
        self.ws_server.add_route("/ws/ip-ai-video", self._on_ip_ai_video_message)
        self.ws_server.add_route("/ws/ip-roi", self._on_ip_roi_message)
        self.ws_server.add_route("/ws/ai-3d", self._on_ai_3d_message)
        self.ws_server.add_route("/ws/ai-3d-video", self._on_ai_3d_video_message)
        self.ws_server.add_route("/ws/ai-3d-roi", self._on_ai_3d_roi_message)

    async def _on_ip_ai_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        event = self._ip_channel.decode(message)
        if event is None:
            return
        await self.handle_ip_event(connection, event)
        await self._ip_channel.broadcast(event)

    async def _on_ip_ai_video_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self.ws_server.broadcast(message, path="/ws/ai-3d-video")

    async def _on_ip_roi_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self.ws_server.broadcast(message, path="/ws/ai-3d-roi")

    async def _on_ai_3d_roi_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self.ws_server.broadcast(message, path="/ws/ai-3d-roi")

    async def _on_ai_3d_video_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        return

    async def _on_ai_3d_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        event = self._ai3d_channel.decode(message)
        if event is None:
            return
        await self.handle_unreal(connection, event)
        await self._ai3d_channel.broadcast(event, origin="Unreal", exclude=connection)

    def state(self) -> str:
        return self.state_machine.state

    def force_state(self, state: str) -> str:
        self.state_machine.force_state(state)
        self._apply_state_effects_from_current_loop()
        return self.state_machine.state

    def trigger_state_event(self, event: str) -> str:
        if not event:
            return self.state_machine.state
        self.state_machine.advance(event)
        self._apply_state_effects_from_current_loop()
        return self.state_machine.state

    def _apply_state_effects_from_current_loop(self) -> None:
        if self.state_machine.state not in SCENES_THAT_START_ANALYSIS:
            self._analysis_stage = "idle"
            return
        if self._hand_data:
            self._start_analysis_from_current_loop()

    def _start_analysis_from_current_loop(self) -> None:
        if self._analysis_task is not None and not self._analysis_task.done():
            return
        if not self._hand_data:
            return
        self._analysis_stage = "queued"
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._pending_fortune_event = None
        self._analysis_task = loop.create_task(self._run_analysis())

    async def handle_ip_event(self, connection: Any, event: WitchEvent) -> None:
        handler = self._ip_handlers.get(type(event))
        if handler is None:
            return
        await handler(connection, event)

    async def _handle_ip_hand_event(self, connection: Any, event: HandEvent) -> None:
        raw_event = msgspec.to_builtins(event)
        self._hand_data = {
            "trigger": event.trigger.value,
            "hand": event.hand.value if event.hand else None,
            "lengths": event.lengths,
            "vector": event.vector,
        }

        transitions = self.state_machine.hand_event(event)
        for transition in transitions:
            await self._broadcast_transition(transition)

        await self._apply_transition_effects(transitions)

    async def handle_unreal(self, connection: Any, message: WitchEvent):
        handler = self._unreal_handlers.get(type(message))
        if handler is None:
            return
        await handler(connection, message)

    async def _handle_unreal_animation_event(self, connection: Any, message: AnimationEvent) -> None:
        if message.trigger is AnimationTrigger.FINISHED:
            await self._handle_animation_finished(message)

    async def _handle_unreal_fortune_request(self, connection: Any, message: FortuneRequestEvent) -> None:
        await self._handle_fortune_request(connection)

    async def _handle_fortune_request(self, connection: Any) -> None:
        if self._hand_data:
            await self._start_analysis()
            return
        await self._ai3d_channel.send_to(
            connection,
            ErrorEvent(
                message="No hand data available",
            ),
        )

    async def _handle_animation_finished(self, message: AnimationEvent) -> None:
        previous_state = self.state_machine.state
        scene = message.scene.value if message.scene else previous_state
        transitions = self.state_machine.animation_finished(scene)
        if not transitions:
            logger.debug("Ignoring animation_finished for %s while state is %s", scene, previous_state)
            return

        for transition in transitions:
            await self._broadcast_transition(transition)

        await self._apply_transition_effects(transitions)

    async def _apply_transition_effects(self, transitions: list[StateChange] | None = None) -> None:
        if (
            self._reached_state(transitions, SCENES_THAT_DELIVER_FORTUNE)
            and await self._broadcast_pending_fortune_if_ready()
        ):
            return
        if self._reached_state(transitions, SCENES_THAT_START_ANALYSIS):
            await self._start_analysis()

    def _reached_state(self, transitions: list[StateChange] | None, states: frozenset[str]) -> bool:
        if transitions is None:
            return self.state_machine.state in states
        return any(transition.dest in states for transition in transitions)

    async def _start_analysis(self) -> None:
        if self._analysis_task is not None and not self._analysis_task.done():
            return
        if not self._hand_data:
            return
        self._pending_fortune_event = None
        self._analysis_stage = "queued"
        self._analysis_task = asyncio.create_task(self._run_analysis())

    async def _run_analysis(self):
        hand_data = copy.deepcopy(self._hand_data or {})
        try:
            self._analysis_stage = "llm"
            await self._broadcast_event_to_unreal(
                AnalysisStartedEvent(
                    scene=Scene(self.state_machine.state),
                ),
            )

            fortune = await self.llm.generate_fortune(hand_data)
            logger.info("Analysis result: chars=%d", len(fortune))

            self._analysis_stage = "tts"
            tts_result = await self.tts.synthesize(fortune)
            audio, sample_rate = tts_result if tts_result is not None else (None, None)
            self._pending_fortune_event = FortuneEvent(
                text=fortune,
                audio=base64.b64encode(audio).decode() if audio else None,
                sample_rate=sample_rate,
            )
            await self._broadcast_event_to_unreal(self._pending_fortune_event)

            self._analysis_stage = "done"

        except httpx.HTTPError as e:
            self._analysis_stage = "error"
            logger.error("LLM/TTS request failed: %s", e)
            await self._broadcast_event_to_unreal(
                ErrorEvent(
                    message=f"Analysis failed: {e}",
                ),
            )
        except Exception as e:
            self._analysis_stage = "error"
            logger.exception("Analysis error")
            await self._broadcast_event_to_unreal(
                ErrorEvent(
                    message=f"Unexpected error: {e}",
                ),
            )

    async def _broadcast_transition(self, transition: StateChange) -> None:
        await self._broadcast_event_to_unreal(
            SceneCommandEvent(
                scene=Scene(transition.dest),
                animation=transition.dest,
                effects={},
                trigger=transition.trigger,
            ),
        )

    async def _broadcast_pending_fortune_if_ready(self) -> bool:
        if (
            self.state_machine.state not in SCENES_THAT_DELIVER_FORTUNE
            or self._pending_fortune_event is None
        ):
            return False
        await self._broadcast_event_to_unreal(self._pending_fortune_event)
        self._pending_fortune_event = None
        return True

    async def _broadcast_event_to_unreal(self, event: WitchEvent) -> None:
        await self._ai3d_channel.broadcast(event)
