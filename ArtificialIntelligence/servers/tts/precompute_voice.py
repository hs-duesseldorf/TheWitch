from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from vllm_omni.utils.custom_voice_io import safe_voice_stem

SPEAKER_SAMPLE_RATE = 24_000
SPEAKER_NUM_MELS = 128


def _resolve_model_dir(model: str) -> str:
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def _read_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    wav, sample_rate = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return np.asarray(wav, dtype=np.float32), int(sample_rate)


def _load_speaker_encoder(model_dir: str, device: torch.device) -> tuple[Any, torch.nn.Module]:
    from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSSpeakerEncoder

    config = Qwen3TTSConfig.from_pretrained(model_dir)
    encoder = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(
            {name for key, name in index.get("weight_map", {}).items() if key.startswith("speaker_encoder.")}
        )
        shard_paths = [Path(model_dir) / name for name in shard_names]
    else:
        shard_paths = sorted(Path(model_dir).glob("model*.safetensors"))

    state: dict[str, torch.Tensor] = {}
    for shard in shard_paths:
        with safe_open(str(shard), framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key.startswith("speaker_encoder."):
                    state[key.removeprefix("speaker_encoder.")] = tensors.get_tensor(key)
    if not state:
        raise RuntimeError("The Base checkpoint does not contain speaker_encoder weights")

    encoder.load_state_dict(state)
    encoder.to(device=device, dtype=torch.bfloat16).eval()
    return config, encoder


def _speaker_embedding(
    encoder: torch.nn.Module,
    wav: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> torch.Tensor:
    from vllm.multimodal.audio import AudioResampler
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import mel_spectrogram

    if sample_rate != SPEAKER_SAMPLE_RATE:
        wav = AudioResampler(target_sr=SPEAKER_SAMPLE_RATE).resample(wav, orig_sr=sample_rate)
    wav_tensor = torch.from_numpy(wav).to(device=device, dtype=torch.float32)
    mels = mel_spectrogram(
        wav_tensor.unsqueeze(0),
        n_fft=1024,
        num_mels=SPEAKER_NUM_MELS,
        sampling_rate=SPEAKER_SAMPLE_RATE,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12_000,
    ).transpose(1, 2)
    with torch.inference_mode():
        return encoder(mels.to(dtype=torch.bfloat16))[0].float().cpu().contiguous()


def _reference_code(model_dir: str, wav: np.ndarray, sample_rate: int, device: torch.device) -> torch.Tensor:
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(Path(model_dir) / "speech_tokenizer"),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    try:
        del tokenizer.model.decoder
        tokenizer.model.decoder = None
        tokenizer.model.encoder.to(device)
        tokenizer.device = device
    except Exception:
        tokenizer.device = device
    with torch.inference_mode():
        encoded = tokenizer.encode(wav, sr=sample_rate, return_dict=True)
    codes = encoded.audio_codes[0] if isinstance(encoded.audio_codes, list) else encoded.audio_codes
    if codes.ndim == 3:
        codes = codes[0]
    return codes.to(dtype=torch.int32, device="cpu").contiguous()


def precompute_voice(
    *,
    model: str,
    voice_name: str,
    ref_audio: Path,
    ref_text: str,
    speaker_description: str,
    output_dir: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = _resolve_model_dir(model)
    config, encoder = _load_speaker_encoder(model_dir, device)
    wav, sample_rate = _read_audio_mono(ref_audio)
    if wav.size < 1024:
        raise ValueError("Reference audio is too short")

    tensors = {
        "speaker_embedding": _speaker_embedding(encoder, wav, sample_rate, device),
        "ref_code": _reference_code(model_dir, wav, sample_rate, device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_voice_stem(voice_name)}.safetensors"
    save_file(tensors, str(output_dir / filename))

    manifest = {
        "schema_version": 1,
        "model_type": "qwen3_tts",
        "model": model,
        "hidden_size": int(config.talker_config.hidden_size),
        "voices": {
            voice_name: {
                "name": voice_name,
                "file": filename,
                "mode": "icl",
                "embedding_dim": int(tensors["speaker_embedding"].numel()),
                "ref_code_length": int(tensors["ref_code"].shape[0]),
                "ref_text": ref_text,
                "speaker_description": speaker_description,
            }
        },
    }
    (output_dir / "custom_voice_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
