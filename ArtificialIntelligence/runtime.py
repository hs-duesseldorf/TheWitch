from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable

from .hand_analyzer import build_prompt
from .speech_pipeline import SpeechPipeline
from .message_channels import EventChannel
from .state_machine.state_machine import (
    SCENES_THAT_DELIVER_FORTUNE,
    SCENES_THAT_START_ANALYSIS,
    StateChange,
    WitchStateMachine,
)
from .websocket_server.websocket_server import WebSocketServer
from shared.events import (
    AnalysisStartedEvent,
    ErrorEvent,
    EventDoneEvent,
    HandEvent,
    PersonEvent,
    Scene,
    SceneCommandEvent,
    WitchEvent,
)

logger = logging.getLogger(__name__)


class StateCoordinator:
    def __init__(
        self,
        state_machine: WitchStateMachine,
        analysis_engine: SpeechPipeline,
        broadcast_to_unreal: Callable[[WitchEvent], Awaitable[None]],
    ):
        self._state_machine = state_machine
        self._analysis_engine = analysis_engine
        self._broadcast_to_unreal = broadcast_to_unreal
        self._hand_event: HandEvent | None = None

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        self._state_machine.register_transition_handler("ip_scan_complete", self._on_scan_complete)
        self._state_machine.register_transition_handler("ip_hand_right", self._on_hand_scan_ready)

    async def _on_scan_complete(self, change: StateChange) -> None:
        pass

    async def _on_hand_scan_ready(self, change: StateChange) -> None:
        pass

    @property
    def state(self) -> str:
        return self._state_machine.state

    def get_hand_data(self) -> HandEvent | None:
        return self._hand_event

    async def handle_hand_event(self, event: HandEvent) -> list[StateChange]:
        self._hand_event = event

        transitions = self._state_machine.hand_event(event)
        await self._apply_transition_effects(transitions)
        return transitions

    async def handle_person_event(self, event: PersonEvent) -> list[StateChange]:
        transitions = self._state_machine.person_event(event)
        await self._apply_transition_effects(transitions)
        return transitions

    async def handle_event_done(self) -> list[StateChange]:
        previous_state = self._state_machine.state
        transitions = self._state_machine.event_done(previous_state)
        if not transitions:
            logger.debug("Ignoring event_done while state is %s", previous_state)
            return []

        await self._apply_transition_effects(transitions)
        return transitions

    async def force_state(self, state: str) -> str:
        self._state_machine.force_state(state)
        await self._apply_state_effects()
        return self._state_machine.state

    def force_state_sync(self, state: str) -> str:
        self._state_machine.force_state(state)
        return self._state_machine.state

    async def trigger_event(self, event: str) -> str:
        if not event:
            return self._state_machine.state
        self._state_machine.advance(event)
        await self._apply_state_effects()
        return self._state_machine.state

    def trigger_event_sync(self, event: str) -> str:
        if not event:
            return self._state_machine.state
        self._state_machine.advance(event)
        return self._state_machine.state

    async def _apply_state_effects(self) -> None:
        if self._state_machine.state not in SCENES_THAT_START_ANALYSIS:
            return
        if self._hand_event:
            await self._trigger_analysis()

    async def _trigger_analysis(self) -> None:
        if self._analysis_engine.is_running():
            return
        prompt = build_prompt(self._hand_event)
        await self._analysis_engine.run_analysis(
            prompt,
            Scene(self._state_machine.state),
        )

    async def _apply_transition_effects(self, transitions: list[StateChange] | None = None) -> None:
        if not transitions:
            if self._state_machine.state in SCENES_THAT_START_ANALYSIS:
                await self._trigger_analysis()
            return

        for change in transitions:
            await self._broadcast_transition(change)

            for handler in self._state_machine.get_transition_handlers(change.trigger):
                await handler(change)

        if self._reached_state(transitions, SCENES_THAT_START_ANALYSIS):
            await self._trigger_analysis()

    def _reached_state(self, transitions: list[StateChange], states: frozenset[str]) -> bool:
        return any(transition.dest in states for transition in transitions)

    async def _broadcast_transition(self, transition: StateChange) -> None:
        await self._broadcast_to_unreal(
            SceneCommandEvent(
                scene=Scene(transition.dest),
                animation=transition.dest,
                effects={},
                trigger=transition.trigger,
            ),
        )


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
            broadcast_callback=self._on_analysis_event,
            audio_callback=self._on_analysis_audio,
        )

        self._coordinator = StateCoordinator(
            state_machine=state_machine,
            analysis_engine=self._speech_pipeline,
            broadcast_to_unreal=self._broadcast_event_to_unreal,
        )

        self._register_routes()

    async def _broadcast_event_to_unreal(self, event: WitchEvent) -> None:
        await self._ai3d_channel.broadcast(event)

    def _register_routes(self) -> None:
        self._ws_server.add_route("/ws/ip-ai", self._on_ip_ai_message)
        self._ws_server.add_route("/ws/ip-ai-video", self._on_ip_ai_video_message)
        self._ws_server.add_route("/ws/ip-roi", self._on_ip_roi_message)
        self._ws_server.add_route("/ws/ai-3d", self._on_ai_3d_message)
        self._ws_server.add_route("/ws/ai-3d-audio", self._on_ai_3d_audio_message)
        self._ws_server.add_route("/ws/ai-3d-video", self._on_ai_3d_video_message)
        self._ws_server.add_route("/ws/ai-3d-roi", self._on_ai_3d_roi_message)

    async def _on_ip_ai_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        event = self._ip_channel.decode(message)
        if event is None:
            return

        if isinstance(event, HandEvent):
            transitions = await self._coordinator.handle_hand_event(event)
            for transition in transitions:
                await self._ip_channel.broadcast(event)
        elif isinstance(event, PersonEvent):
            transitions = await self._coordinator.handle_person_event(event)
            for transition in transitions:
                await self._ip_channel.broadcast(event)

    async def _on_ip_ai_video_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws_server.broadcast(message, path="/ws/ai-3d-video")

    async def _on_ip_roi_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws_server.broadcast(message, path="/ws/ai-3d-roi")

    async def _on_ai_3d_roi_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws_server.broadcast(message, path="/ws/ai-3d-roi")

    async def _on_ai_3d_video_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        return

    async def _on_ai_3d_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        event = self._ai3d_channel.decode(message)
        if event is None:
            return

        if isinstance(event, EventDoneEvent):
            await self._coordinator.handle_event_done()

    async def _on_ai_3d_audio_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        return

    async def _on_analysis_event(
        self,
        event: AnalysisStartedEvent | ErrorEvent,
    ) -> None:
        await self._ai3d_channel.broadcast(event)

    async def _on_analysis_audio(self, audio: bytes) -> None:
        logger.info(f"Audio broadcast: {len(audio)} bytes, clients: {self._ws_server.client_count('/ws/ai-3d-audio')}")
        await self._ws_server.broadcast(audio, path="/ws/ai-3d-audio")

    @property
    def state(self) -> str:
        return self._coordinator.state

    @property
    def analysis_stage(self) -> str:
        return self._speech_pipeline.stage

    def latest_tts_audio(self) -> tuple[bytes, int | None] | None:
        return None

    def force_state(self, state: str) -> str:
        return self._coordinator.force_state_sync(state)

    def trigger_state_event(self, event: str) -> str:
        return self._coordinator.trigger_event_sync(event)

    def _trigger_apply_effects(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._coordinator._apply_state_effects())
        except RuntimeError:
            pass
