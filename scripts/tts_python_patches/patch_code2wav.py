from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)


def _sequence_counts(input_ids: torch.Tensor, kwargs: dict[str, object]) -> list[int]:
    seq_token_counts = kwargs.get("seq_token_counts")
    if isinstance(seq_token_counts, (list, tuple)):
        counts: list[int] = []
        for count in seq_token_counts:
            try:
                counts.append(int(count))
            except (TypeError, ValueError):
                return [int(input_ids.numel())]
        return counts or [int(input_ids.numel())]
    return [int(input_ids.numel())]


def _clamp_oversized_left_context(
    runtime_additional_information,
    *,
    input_ids: torch.Tensor,
    num_quantizers: int,
    kwargs: dict[str, object],
):
    if not isinstance(runtime_additional_information, list):
        return runtime_additional_information

    counts = _sequence_counts(input_ids, kwargs)
    if len(counts) != len(runtime_additional_information):
        return runtime_additional_information

    patched_information = []
    changed = False

    for info, token_count in zip(runtime_additional_information, counts, strict=False):
        if not isinstance(info, dict):
            patched_information.append(info)
            continue

        new_info = dict(info)
        try:
            left_context_size = int(new_info.get("left_context_size", 0) or 0)
        except (TypeError, ValueError):
            patched_information.append(info)
            continue

        decoded_frames = max(0, int(token_count) // max(1, num_quantizers))
        if left_context_size >= decoded_frames > 0:
            logger.warning(
                "Code2Wav left_context_size %d >= decoded frame count %d; clamping to 0.",
                left_context_size,
                decoded_frames,
            )
            new_info["left_context_size"] = 0
            changed = True

        patched_information.append(new_info)

    return patched_information if changed else runtime_additional_information


def _patch_qwen3_tts_code2wav() -> bool:
    """Patch Qwen3TTSCode2Wav to handle malformed async chunk input gracefully."""
    try:
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav
    except Exception:
        return False

    if getattr(Qwen3TTSCode2Wav.forward, "_witch_malformed_chunk_patch", False):
        return True

    original_forward = Qwen3TTSCode2Wav.forward

    @functools.wraps(original_forward)
    def patched_forward(self, input_ids=None, positions=None, intermediate_tensors=None,
                        inputs_embeds=None, runtime_additional_information=None, **kwargs):
        if input_ids is None or input_ids.numel() == 0:
            from vllm_omni.model_executor.models.output_templates import OmniOutput

            sr_tensor = torch.tensor(24000, dtype=torch.int32)
            empty = torch.zeros((0,), dtype=torch.float32)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        q = int(self._num_quantizers)
        n = ids.numel()

        if n > 0 and n % q != 0:
            logger.warning(
                "Code2Wav input_ids length %d not divisible by num_quantizers %d; padding with zeros.",
                n, q,
            )
            pad_len = q - (n % q)
            padding = torch.zeros(pad_len, dtype=torch.long, device=input_ids.device)
            input_ids = torch.cat([ids, padding])
            if "seq_token_counts" in kwargs:
                kwargs["seq_token_counts"] = [input_ids.numel()]

        runtime_additional_information = _clamp_oversized_left_context(
            runtime_additional_information,
            input_ids=input_ids,
            num_quantizers=q,
            kwargs=kwargs,
        )

        return original_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            runtime_additional_information=runtime_additional_information,
            **kwargs,
        )

    Qwen3TTSCode2Wav.forward = patched_forward
    Qwen3TTSCode2Wav.forward._witch_malformed_chunk_patch = True
    logger.info("Patched Qwen3TTSCode2Wav malformed async chunk handling")
    return True


_patch_qwen3_tts_code2wav()
