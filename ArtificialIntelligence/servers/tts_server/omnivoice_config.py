from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Runtime settings for the OmniVoice TTS server
# we're doing this here to keep both .env and the servers main.py more clean
DEVICE = "cuda:0"
NUM_STEP = 32
SAMPLE_RATE = 24000
LANGUAGE = "de"


# Below the audio sample is hardcoded right now, we'll change this later
# because once we need two different voices we cant have it like this but for now its fine

# Voice cloning reference
# Put your German reference clip at this path, or change REF_AUDIO here
# Recommended: clean roughly 3-10 second long WAV clip
REF_AUDIO = str(BASE_DIR / "voice_refs" / "witch_reference.wav")

# This must match what is spoken in REF_AUDIO!!
REF_TEXT = """
Es ist wichtig sich selbst die Zeit zu geben, die man braucht, um zu heilen und die eigenen Gefühle zu verstehen.
""".strip()
