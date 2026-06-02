from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from .clients.llm_client import LLMClient
from .clients.tts_client import TTSClient
from .hand_analyzer import build_prompt
from .message_channels import EventChannel
from .speech_pipeline import SpeechPipeline
from .state_machine.state_machine import (
    SCENES_THAT_DELIVER_FORTUNE,
    SCENES_THAT_START_ANALYSIS,
    StateChange,
    WitchStateMachine,
)
from .websocket_server.websocket_server import WebSocketServer
from shared.events import (
    AnalysisResultEvent,
    AnalysisStartedEvent,
    ErrorEvent,
    EventDoneEvent,
    HandEvent,
    HandTrigger,
    PersonEvent,
    PersonTrigger,
    Scene,
    SceneCommandEvent,
    WitchEvent,
)

logger = logging.getLogger("ai")

POST_SCAN_NO_WAIT_STATES = frozenset({
    Scene.SCENE_3_HANDSCAN_DONE.value,
    Scene.SCENE_4_TRANSFORMATION.value,
})


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
        self._hand_event: HandEvent | None = None
        self._pending_scene_changes: list[StateChange] = []
        self.manual_mode = False

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
            done_callback=self._on_analysis_done,
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

    async def _on_ip_ai_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        event = self._ip_channel.decode(message)
        if event is None:
            return
        
        if isinstance(event, HandEvent):
            if self.manual_mode:
                return

        if isinstance(event, HandEvent):
            logger.info("Broadcasting hand event: %s", event.trigger)
            await self._handle_hand_event(event)
            await self._ip_channel.broadcast(event)
        elif isinstance(event, PersonEvent):
            changes = self._state_machine.person_event(event)
            for change in changes:
                await self._ai3d_channel.broadcast(
                    SceneCommandEvent(
                        scene=Scene(change.dest),
                        animation=change.dest,
                        effects={},
                        trigger=change.trigger,
                    ),
                )
                if self.manual_mode:
                    asyncio.create_task(self._simulate_event_done())
            await self._ip_channel.broadcast(event)

    async def _handle_hand_event(self, event: HandEvent) -> None:
        self._hand_event = event
        changes = self._state_machine.hand_event(event)
        await self._broadcast_scene_changes(changes)

        if self._state_machine.state in POST_SCAN_NO_WAIT_STATES:
            changes = await self._advance_post_scan_without_3d_events()

        if self._state_machine.state in SCENES_THAT_START_ANALYSIS:
            await self._trigger_analysis([])

    async def _handle_person_event(self, event: PersonEvent) -> None:
        self._person_event = event
        changes = self._state_machine.person_event(event)
        await self._broadcast_scene_changes(changes)

    async def _advance_post_scan_without_3d_events(self) -> list[StateChange]:
        changes: list[StateChange] = []
        while self._state_machine.state in POST_SCAN_NO_WAIT_STATES:
            next_changes = self._state_machine.event_done(self._state_machine.state)
            if not next_changes:
                break
            changes.extend(next_changes)

        if changes:
            logger.info(
                "Auto-advanced post-scan flow without 3D event_done: %s",
                " -> ".join(change.dest for change in changes),
            )
            await self._broadcast_scene_changes(changes)
        return changes

    async def _broadcast_scene_changes(self, changes: list[StateChange]) -> None:
        for change in changes:
            await self._ai3d_channel.broadcast(
                SceneCommandEvent(
                    scene=Scene(change.dest),
                    animation=change.dest,
                    effects={},
                    trigger=change.trigger,
                ),
            )

    async def _trigger_analysis(self, pending_changes: list[StateChange]) -> None:
        if self._speech_pipeline.is_running():
            return
        self._speech_pipeline.stop_player()
        self._pending_scene_changes = pending_changes
        prompt = build_prompt(self._hand_event)
        await self._speech_pipeline.run_analysis(
            prompt,
            Scene(self._state_machine.state),
        )

    async def _on_ip_ai_video_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws_server.broadcast(message, path="/ws/ai-3d-video")

    async def _on_ip_roi_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        if isinstance(message, bytes):
            await self._ws_server.broadcast(message, path="/ws/ai-3d-roi")

    async def _on_ai_3d_message(self, server: WebSocketServer, connection: Any, message: str | bytes) -> None:
        event = self._ai3d_channel.decode(message)
        if event is None:
            return

        if isinstance(event, EventDoneEvent):
            changes = self._state_machine.event_done(self._state_machine.state)
            await self._broadcast_scene_changes(changes)
            if self._state_machine.state in SCENES_THAT_DELIVER_FORTUNE:
                await self._trigger_analysis([])

    async def _on_analysis_event(
        self,
        event: AnalysisStartedEvent | ErrorEvent,
    ) -> None:
        await self._ai3d_channel.broadcast(event)

    async def _on_analysis_done(self, text: str) -> None:
        await self._broadcast_pending_scene_changes()
        await self._ai3d_channel.broadcast(
            AnalysisResultEvent(text=text),
        )

    async def _broadcast_pending_scene_changes(self) -> None:
        if self._pending_scene_changes:
            for change in self._pending_scene_changes:
                await self._ai3d_channel.broadcast(
                    SceneCommandEvent(
                        scene=Scene(change.dest),
                        animation=change.dest,
                        effects={},
                        trigger=change.trigger,
                    ),
                )
            self._pending_scene_changes = []


    async def _simulate_event_done(self):
        await asyncio.sleep(3)

        changes = self._state_machine.event_done(self._state_machine.state)
        await self._broadcast_scene_changes(changes)

        if self._state_machine.state in SCENES_THAT_DELIVER_FORTUNE:
            await self._trigger_analysis([])

    async def simulate_hand_event(self, event:str) -> str:
        try:
            trigger_enum = HandTrigger(event)
        except ValueError:
            raise ValueError(f"Ungültiger HandTrigger: {event}")

        event = HandEvent(
            trigger=trigger_enum,
            origin="ManualSimulation",
            hand=None,
            lengths={},
            vector=[]
        )

        await self._handle_hand_event(event)

        return self._state_machine.state
    
    async def simulate_person_event(self, event:str) -> str:
        try:
            trigger_enum = PersonTrigger(event)
        except ValueError:
            raise ValueError(f"Ungültiger PersonTrigger: {event}")

        event = PersonEvent(
            trigger=trigger_enum,
            origin="ManualSimulation",
        )

        await self._handle_person_event(event)

        return self._state_machine.state

    @property
    def state(self) -> str:
        return self._state_machine.state

    @property
    def analysis_stage(self) -> str:
        return self._speech_pipeline.stage

    def force_state(self, state: str) -> str:
        self._state_machine.force_state(state)
        self._hand_event = None
        self._pending_scene_changes = []
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._speech_pipeline.cancel())
        except RuntimeError:
            pass
        return self._state_machine.state

    def trigger_state_event(self, event: str) -> str:
        if event:
            change = self._state_machine.advance(event)
            logger.info("trigger_state_event: %s -> %s (change=%s)", self._state_machine.state, event, change)
            if event == "reset" and change is None:
                self._state_machine.force_state("scene_0_idle")
                logger.info("Reset: force_state fallback used")
            if event == "reset":
                self._hand_event = None
                self._pending_scene_changes = []
                logger.info("Reset: cancelling speech pipeline")
                self._speech_pipeline.cancel_sync()
        return self._state_machine.state 
