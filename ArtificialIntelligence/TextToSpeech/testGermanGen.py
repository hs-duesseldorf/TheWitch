from TTS.api import TTS

# Load XTTS v2 model
tts = TTS("tts_models/de/thorsten/tacotron2-DDC", gpu=False)

# Reference voice file
speaker_wav = "sample.wav"

# Output settings
output_file = "output.wav"

print("Enter text to synthesize (or type 'exit'):\n")

while True:
    text = input("Text: ")

    if text.lower() == "exit":
        break

    print("Generating audio...")

    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path=output_file
    )

    print(f"Saved: {output_file}\n")