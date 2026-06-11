from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parents[2]
load_dotenv(ROOT / ".env")

VOICE_NAME = os.environ["WITCH_TTS_VOICE_NAME"]
VOICE_DESCRIPTION = os.environ["WITCH_TTS_VOICE_DESCRIPTION"]
CUSTOM_VOICE_DIR = SERVER_DIR / "custom_voices"
CUSTOM_VOICE_MANIFEST = CUSTOM_VOICE_DIR / "custom_voice_manifest.json"
CUSTOM_VOICE_PROFILE = CUSTOM_VOICE_DIR / f"{VOICE_NAME}.safetensors"
VOICE_ANCHOR = CUSTOM_VOICE_DIR / "voice_anchor.wav"
VOICE_ANCHOR_TEXT = os.environ["WITCH_TTS_ANCHOR_TEXT"]

BASE_MODEL = os.environ["WITCH_TTS_BASE_MODEL"]
VOICE_DESIGN_MODEL = os.environ["WITCH_TTS_VOICE_DESIGN_MODEL"]
LANGUAGE = os.environ["WITCH_TTS_LANGUAGE"]
INSTRUCTIONS = VOICE_DESCRIPTION


@dataclass(frozen=True)
class TTSProfile:
    model: str
    task_type: str
    voice: str | None
    instructions: str | None
    uses_anchor: bool


def profile() -> TTSProfile:
    if CUSTOM_VOICE_MANIFEST.is_file() and CUSTOM_VOICE_PROFILE.is_file():
        return TTSProfile(
            model=BASE_MODEL,
            task_type="Base",
            voice=VOICE_NAME,
            instructions=None,
            uses_anchor=True,
        )
    return TTSProfile(
        model=VOICE_DESIGN_MODEL,
        task_type="VoiceDesign",
        voice=None,
        instructions=INSTRUCTIONS,
        uses_anchor=False,
    )
