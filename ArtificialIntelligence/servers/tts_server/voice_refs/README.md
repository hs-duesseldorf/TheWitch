# OmniVoice reference audio

For now the setup is only for one Voice!

Put your voice-cloning reference file here:

```text
witch_reference.wav
```

Then edit `../omnivoice_config.py` and set `REF_TEXT` to the exact German transcript of the audio.

Recommended reference length: 3-10 seconds, clean speech, little/no background noise, but little bit longer audio is also fine

Later on when we want to have the two different voices dependent on the current state, we'll have both references/samples here. Will prob work with a check which state we are currently in to decide which voice to use.

