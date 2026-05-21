from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)


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
