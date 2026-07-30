from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import pedalboard

from ArtificialIntelligence.analysis.hand_analyzer import HandAnalyzer
from ArtificialIntelligence.analysis.scene_prompt_builder import (
    ScenePromptBuilder,
    ScenePromptContext,
)
from ArtificialIntelligence.clients.llm_client import LLMClient
from ArtificialIntelligence.clients.tts_client import TTSClient
from ArtificialIntelligence.speech.audio_player import AudioEffect
from ArtificialIntelligence.speech.speech_pipeline import (
    SpeechPipeline,
    normalize_llm_text,
)
from ArtificialIntelligence.state_machine.state_machine import StateMachine, Transition
from ArtificialIntelligence.ui.debug_ui import run as run_debug_ui
from ArtificialIntelligence.ui.debug_ui import set_runtime
from ArtificialIntelligence.websocket.event_channel import EventChannel
from ArtificialIntelligence.websocket.websocket_server import WebSocketServer
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

logger = logging.getLogger("ai")

OUTBOUND_AI3D_EVENT = (
    SceneCommandEvent | AnalysisStartedEvent | AnalysisResultEvent | ErrorEvent
)


@dataclass(frozen=True)
class PendingUnrealAck:
    event_type: str
    scene: str


@dataclass(frozen=True)
class SpeechRequest:
    content: str
    scene: Scene
    effects: list[AudioEffect] | None = None


class Runtime:
    def __init__(self):
        # Counts open manual debug ui websites
        self.manual_debug_clients = 0
        self.websocket_server = WebSocketServer(
            host="0.0.0.0",
            port=int(os.environ["WITCH_AI_PORT"]),
            runtime=self,
        )
        # Groups Broadcast Routes by their associated usage
        # Eases changes to the manual_debug_ui; one can choose which broadcast to silence
        self.broadcast_groups = {
            # visual input
            "camera": {
                "/ws/ip-ai-video",
                "/ws/ip-roi",
            },
            # visual output
            "unity": {
                "/ws/ai-3d",
                "/ws/ai-3d-video",
                "/ws/ai-3d-roi",
            },
            "events": {
                "/ws/ip-ai",
            },
        }
        # Name all Groups here that should be disabled during the manual debug ui use
        # set() for none (Why would you ever do that? Just use the regular Debug UI dummy!)
        # This makes it possible to allow broadcasts of f.e. the events, but to disable the camera stream and Unity trigger
        self.disabled_broadcasts_on_manual_debug_ui = {"camera", "unity", "events"}

        self.state_machine = StateMachine()
        self.hand_analyzer = HandAnalyzer()
        self.scene_prompt_builder = ScenePromptBuilder()
        self.speech_pipeline = SpeechPipeline(
            llm=LLMClient(
                base_url=self._llm_url(),
                model=os.environ["WITCH_OLLAMA_MODEL"].strip(),
            ),
            tts=TTSClient(base_url=self._tts_url()),
            prebuffer_seconds=float(os.environ["WITCH_AUDIO_PREBUFFER_SECONDS"]),
            speaker_delay_seconds=float(os.environ["WITCH_SPEAKER_DELAY_SECONDS"]),
        )

        self._accepted_hand_event: HandEvent | None = None
        self.forbidden_hand: Hand | None = None
        self.desired_hand: Hand | None = None
        self._gaslight_correct_hand: Hand | None = None
        self.gaslight_pending = False
        self.gaslight_hand_name: str | None = None

        self.pending_unreal_ack: PendingUnrealAck | None = None
        self.queued_speech: SpeechRequest | None = None
        self.speech_task: asyncio.Task[None] | None = None
        self.speech_loop: asyncio.AbstractEventLoop | None = None
        self.speech_scene: Scene | None = None
        self.analysis_started = False
        self.debug_ui_started = False
        self._debug_return_state: str | None = None
        self._debug_cooldown_until: float = 0.0

        self.ip_channel = EventChannel(
            self.websocket_server,
            "/ws/ip-ai",
            "ImageProcessing",
            "ip",
        )
        self.ai3d_channel = EventChannel(
            self.websocket_server,
            "/ws/ai-3d",
            "ArtificialIntelligence",
            "ai-3d",
        )

        self.state_handlers = {
            Scene.SCENE_0_IDLE: self.enter_idle,
            Scene.SCENE_1_WELCOME: self.enter_welcome,
            Scene.SCENE_2_SEATED: self.enter_seated,
            Scene.SCENE_3_INTRO: self.enter_intro,
            Scene.SCENE_4_AWAITING_HAND: self.enter_awaiting_hand,
            Scene.SCENE_5_HANDSCAN: self.enter_handscan,
            Scene.SCENE_7_HANDSCAN_DONE: self.enter_handscan_complete,
            Scene.SCENE_8_ANALYSIS: self.enter_analysis,
            Scene.SCENE_9_OUTRO: self.enter_outro,
            Scene.DEBUG_HAND_PULLED_AWAY: self.enter_debug_hand_pulled_away,
            Scene.DEBUG_HAND_TILTED: self.enter_debug_hand_tilted,
            Scene.DEBUG_WRONG_SIDE: self.enter_debug_wrong_side,
            Scene.DEBUG_NOT_FULLY_IN_VIEW: self.enter_debug_not_fully_in_view,
            Scene.DEBUG_GASLIGHT: self.enter_debug_gaslight,
        }
        self.register_routes()
        set_runtime(self.websocket_server, self.state_machine, self)

    def run(self):
        self.start_debug_ui()
        asyncio.run(self.websocket_server.start())

    def start_debug_ui(self):
        if self.debug_ui_started:
            return
        self.debug_ui_started = True
        threading.Thread(target=run_debug_ui, daemon=True).start()

    # Checks how many Routes are open to manual_debug_ui.html and 
    # enters restricted broadcast mode aslong as at least 1 is open
    def manual_debug_active(self):
        return self.manual_debug_clients > 0

    # Silences all Broadcasts defined to be in innit
    def broadcast_allowed(self, path):
        if self.manual_debug_active():
            for group_name in self.disabled_broadcasts_on_manual_debug_ui:
                if path in self.broadcast_groups[group_name]:
                    return False
        return True

    def register_routes(self):
        self.websocket_server.add_route("/ws/ip-ai", self._on_ip_ai_message)
        self.websocket_server.add_route("/ws/ip-ai-video", self._on_ip_ai_video_message)
        self.websocket_server.add_route("/ws/ip-roi", self._on_ip_roi_message)
        self.websocket_server.add_route("/ws/ai-3d", self._on_ai_3d_message)
        self.websocket_server.add_route("/ws/ai-3d-video", None)
        self.websocket_server.add_route("/ws/ai-3d-roi", None)
        self.websocket_server.add_route("/ws/manual-debug", None)


    async def _on_ip_ai_message(
        self, _server: WebSocketServer, _connection: Any, message: str | bytes
    ):
        event = self.ip_channel.decode(message)
        if not isinstance(event, (HandEvent, PersonEvent)):
            return
        await self.ip_channel.broadcast(event)
        if isinstance(event, HandEvent) and self.state_machine.manual_mode:
            return
        await self.handle_image_processing_event(event)

    async def _on_ip_ai_video_message(
        self, _server: WebSocketServer, _connection: Any, message: str | bytes
    ):
        # In manual mode, do NOT broadcast video frames
        # Else it will result in constant error messages, as no link is established
        if self.state_machine.manual_mode:
            return
        if isinstance(message, bytes):
            await self.websocket_server.broadcast(message, path="/ws/ai-3d-video")

    async def _on_ip_roi_message(
        self, _server: WebSocketServer, _connection: Any, message: str | bytes
    ):
        if isinstance(message, bytes):
            await self.websocket_server.broadcast(message, path="/ws/ai-3d-roi")

    async def _on_ai_3d_message(
        self, _server: WebSocketServer, _connection: Any, message: str | bytes
    ):
        event = self.ai3d_channel.decode(message)
        # In manual mode, do NOT broadcast video frames
        # Else it will result in constant error messages, as no link is established
        if self.state_machine.manual_mode:
            return
        if isinstance(event, EventDoneEvent):
            await self.handle_unreal_event(event)

    async def handle_image_processing_event(self, event: IPEvent):
        if isinstance(event, HandEvent):
            await self.handle_hand_event(event)
        else:
            await self.handle_person_event(event)

    async def handle_hand_event(self, event: HandEvent):
        if time.monotonic() < self._debug_cooldown_until:
            return
        scene = self.state_machine.current_scene

        if scene is Scene.SCENE_4_AWAITING_HAND:
            if event.trigger is HandTrigger.ABSENT:
                return
            if self._should_reject_for_gaslight(event):
                self._start_gaslight(event)
                await self.trigger_state("gaslight_reject", cancel_speech=False)
                return
            self._accepted_hand_event = event
            await self.trigger_state(event.trigger.value, cancel_speech=True)
            return

        if scene is Scene.SCENE_5_HANDSCAN:
            await self.trigger_state(event.trigger.value, cancel_speech=True)

        if scene is Scene.SCENE_6_HAND_DETECTED:
            await self.trigger_state(event.trigger.value, cancel_speech=True)

    async def handle_person_event(self, event: PersonEvent):
        if event.trigger is not PersonTrigger.ABSENT:
            self._clear_hand_selection()
        await self.trigger_state(event.trigger.value, cancel_speech=False)

    async def handle_unreal_event(self, _event: EventDoneEvent):
        pending = self.pending_unreal_ack
        if pending is None:
            return
        self.pending_unreal_ack = None
        current = self.state_machine.state
        if current.startswith("scene_debug_") and getattr(
            self, "_debug_return_state", None
        ):
            dest = self._debug_return_state
            self._debug_return_state = None
            change = self.state_machine._set_state("debug_done", dest)
            await self.apply_state_change(change, cancel_speech=False)
            return
        changes = self.state_machine.event_done(pending.scene)
        if changes:
            await self.apply_state_change(changes[-1], cancel_speech=False)

    async def trigger_state(self, trigger: str, *, cancel_speech: bool = False):
        change = self.state_machine.trigger(trigger)
        if change is not None:
            await self.apply_state_change(change, cancel_speech=cancel_speech)

    async def apply_state_change(self, change: Transition, *, cancel_speech: bool):
        self.pending_unreal_ack = None
        if change.dest.startswith("scene_debug_"):
            self._debug_return_state = (
                Scene.SCENE_5_HANDSCAN.value
                if change.source == Scene.SCENE_6_HAND_DETECTED.value
                else change.source
            )
        if (
            cancel_speech
            and self.speech_task is not None
            and not self.speech_task.done()
        ):
            logger.info("Queuing speech replacement at next sentence boundary")
        handler = self.state_handlers.get(Scene(change.dest))
        if handler is not None:
            await handler(change)

    async def enter_idle(self, _change: Transition):
        self.reset_cycle_state()

    async def enter_welcome(self, _change: Transition):
        await self.speak_scene(Scene.SCENE_1_WELCOME)

    async def enter_seated(self, _change: Transition):
        await self.speak_scene(Scene.SCENE_2_SEATED)

    async def enter_intro(self, _change: Transition):
        await self.speak_scene(Scene.SCENE_3_INTRO)

    async def enter_awaiting_hand(self, change: Transition):
        if change.source.startswith("scene_debug_"):
            self._debug_cooldown_until = time.monotonic() + float(
                os.environ["WITCH_DEBUG_COOLDOWN_SECONDS"]
            )

    async def enter_handscan(self, _change: Transition):
        self._clear_hand_selection()
        if not self.analysis_started:
            self.analysis_started = True
            try:
                await self.emit_ai3d_event(
                    AnalysisStartedEvent(scene=Scene.SCENE_5_HANDSCAN)
                )
            except Exception as exc:
                logger.error("Failed to send analysis started event: %s", exc)
        await self.speak_scene(Scene.SCENE_5_HANDSCAN)

    async def enter_handscan_complete(self, _change: Transition):
        await self.speak_scene(Scene.SCENE_7_HANDSCAN_DONE)

    async def enter_analysis(self, _change: Transition):
        prompt = self.hand_analyzer.build_prompt(self._accepted_hand_event)
        if prompt is None:
            prompt = (
                self.scene_prompt_builder.build_prompt(Scene.SCENE_8_ANALYSIS) or ""
            )

        pitch_shifter = pedalboard.Pedalboard([pedalboard.PitchShift(-4)])
        effects: list[AudioEffect] = [
            lambda data, sr: data + pitch_shifter(data, sr) * 0.25
        ]
        await self.start_speech(
            SpeechRequest(prompt, Scene.SCENE_8_ANALYSIS, effects=effects)
        )

    async def enter_outro(self, _change: Transition):
        await self.speak_scene(Scene.SCENE_9_OUTRO)

    async def enter_debug_hand_pulled_away(self, _change: Transition):
        await self.speak_scene(Scene.DEBUG_HAND_PULLED_AWAY)

    async def enter_debug_hand_tilted(self, _change: Transition):
        await self.speak_scene(Scene.DEBUG_HAND_TILTED)

    async def enter_debug_wrong_side(self, _change: Transition):
        await self.speak_scene(Scene.DEBUG_WRONG_SIDE)

    async def enter_debug_not_fully_in_view(self, _change: Transition):
        await self.speak_scene(Scene.DEBUG_NOT_FULLY_IN_VIEW)

    async def enter_debug_gaslight(self, _change: Transition):
        await self.speak_scene(Scene.DEBUG_GASLIGHT)

    async def speak_scene(self, scene: Scene, *, base_scene: Scene | None = None):
        prompt = self.scene_prompt_builder.build_prompt(
            base_scene or scene,
            ScenePromptContext(
                gaslight_pending=self.gaslight_pending,
                gaslight_hand_name=self.gaslight_hand_name,
            ),
        )
        if prompt is None:
            await self.advance_scene_after_speech(scene)
            return
        await self.start_speech(SpeechRequest(prompt, scene))

    async def start_speech(self, request: SpeechRequest):
        if self.speech_task is not None and not self.speech_task.done():
            self.queued_speech = request
            return
        loop = asyncio.get_running_loop()
        self.speech_scene = request.scene
        self.speech_loop = loop
        self.speech_task = loop.create_task(self.run_speech(request))

    async def run_speech(self, request: SpeechRequest):
        scene = request.scene
        try:
            text = request.content
            text = await self.speech_pipeline.generate_text(text)
            text = normalize_llm_text(text)
            await self.emit_ai3d_event(
                SceneCommandEvent(
                    scene=scene,
                    animation=scene.value,
                    effects={},
                    text=text,
                )
            )
            if scene is Scene.SCENE_8_ANALYSIS:
                await self.emit_ai3d_event(AnalysisResultEvent(text=text, scene=scene))
            await self.speech_pipeline.play_text(text, effects=request.effects)
            await self.advance_scene_after_speech(scene)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            try:
                await self.emit_ai3d_event(
                    ErrorEvent(message=f"Speech failed for {scene.value}: {exc}")
                )
            except Exception as inner:
                logger.error("Failed to send error event to Unreal: %s", inner)
            if self.state_machine.state == scene.value:
                await asyncio.sleep(2)
                await self.speak_scene(scene)
        finally:
            if self.speech_task is asyncio.current_task():
                self.speech_task = None
                self.speech_loop = None
            if self.speech_scene == scene:
                self.speech_scene = None
            queued = self.queued_speech
            self.queued_speech = None
            if queued is not None:
                with suppress(RuntimeError):
                    await self.start_speech(queued)

    async def emit_ai3d_event(self, event: OUTBOUND_AI3D_EVENT):
        await self.ai3d_channel.broadcast(event)
        self.track_pending_unreal_ack(event)

    def track_pending_unreal_ack(self, event: OUTBOUND_AI3D_EVENT):
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
        if scene == self.state_machine.state and self.wait_for_unreal_ack:
            self.pending_unreal_ack = PendingUnrealAck(event_type, scene)

    async def advance_scene_after_speech(self, scene: Scene):
        if self.state_machine.state != scene.value:
            return
        if scene.value.startswith("scene_debug_") and getattr(
            self, "_debug_return_state", None
        ):
            dest = self._debug_return_state
            self._debug_return_state = None
            change = self.state_machine._set_state("debug_done", dest)
            await self.apply_state_change(change, cancel_speech=False)
            return
        changes = self.state_machine.event_done(scene.value)
        if changes:
            await self.apply_state_change(changes[-1], cancel_speech=False)

    async def cancel_speech(self):
        task = self.speech_task
        self.speech_task = None
        self.speech_loop = None
        self.speech_scene = None
        self.queued_speech = None
        if task is None or task.done():
            return
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

    async def simulate_hand_event(self, event: str) -> str:
        ev = HandEvent(trigger=HandTrigger(event), origin="ManualSimulation")
        await self.ip_channel.broadcast(ev)
        await self.handle_hand_event(ev)
        return self.state

    async def simulate_person_event(self, event: str) -> str:
        ev = PersonEvent(trigger=PersonTrigger(event), origin="ManualSimulation")
        await self.ip_channel.broadcast(ev)
        await self.handle_person_event(ev)
        return self.state

    async def acknowledge_unreal_event(self) -> str:
        await self.handle_unreal_event(EventDoneEvent(origin="ManualSimulation"))
        return self.state

    async def trigger_state_event(self, event: str) -> str:
        if event == "reset":
            await self.cancel_speech()
        await self.trigger_state(event, cancel_speech=event == "reset")
        return self.state

    def force_state(self, state: str) -> str:
        self.state_machine.force_state(state)
        self.reset_cycle_state()
        return self.state

    def set_manual_mode(self, enabled: bool) -> bool:
        self.state_machine.manual_mode = enabled
        return enabled

    def reset_cycle_state(self):
        self._accepted_hand_event = None
        self._clear_hand_selection()
        self.pending_unreal_ack = None
        self.queued_speech = None
        self.analysis_started = False
        # Would break manual_debug_ui if it gets set to false after a cycle has been simulated
        # Should only be changed via the manual_debug_ui.html button!
        #self.state_machine.manual_mode = False
        self._debug_return_state = None
        self._debug_cooldown_until = 0.0
        self._gaslight_correct_hand = None

    @property
    def state(self) -> str:
        return self.state_machine.state

    @property
    def speech_stage(self) -> str:
        return self.speech_pipeline.stage

    @property
    def analysis_stage(self) -> str:
        return self.speech_stage

    @property
    def wait_for_unreal_ack(self) -> bool:
        return os.environ["WITCH_WAIT_FOR_UNREAL_ACK"].strip().lower() == "true"

    def _should_reject_for_gaslight(self, event: HandEvent) -> bool:
        if os.environ["WITCH_GASLIGHT"].strip().lower() != "true":
            return False
        if event.trigger is not HandTrigger.DETECTED or event.hand is None:
            return False
        if self._gaslight_correct_hand is None:
            return True
        return event.hand is not self._gaslight_correct_hand

    def _start_gaslight(self, event: HandEvent):
        if event.hand is None:
            return
        if self.desired_hand is None:
            self.forbidden_hand = event.hand
            self.desired_hand = Hand.LEFT if event.hand is Hand.RIGHT else Hand.RIGHT
        if self._gaslight_correct_hand is None:
            self._gaslight_correct_hand = (
                Hand.LEFT if event.hand is Hand.RIGHT else Hand.RIGHT
            )
        self.gaslight_pending = True
        self.gaslight_hand_name = self._hand_name(event)

    def _clear_hand_selection(self):
        self.forbidden_hand = None
        self.desired_hand = None
        self.gaslight_pending = False
        self.gaslight_hand_name = None

    def _hand_name(self, event: HandEvent) -> str:
        if event.hand is Hand.LEFT:
            return "linke"
        if event.hand is Hand.RIGHT:
            return "rechte"
        return "unbekannte"

    def _llm_url(self) -> str:
        return f"http://{os.environ['WITCH_LLM_HOST']}:{os.environ['WITCH_LLM_PORT']}"

    def _tts_url(self) -> str:
        return f"http://{os.environ['WITCH_TTS_HOST']}:{os.environ['WITCH_TTS_PORT']}"
