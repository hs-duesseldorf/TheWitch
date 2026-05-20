from __future__ import annotations

import functools
import inspect


def _patch_transformers_masking_utils() -> None:
    try:
        from transformers import masking_utils
    except Exception:
        return

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        func = getattr(masking_utils, name, None)
        if func is None or getattr(func, "_witch_input_embeds_compat", False):
            continue

        params = inspect.signature(func).parameters
        if "input_embeds" in params or "inputs_embeds" not in params:
            continue
        unsupported_kwargs = {"cache_position"} - set(params)

        @functools.wraps(func)
        def wrapper(*args, __func=func, **kwargs):
            if "input_embeds" in kwargs and "inputs_embeds" not in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            for key in unsupported_kwargs:
                kwargs.pop(key, None)
            return __func(*args, **kwargs)

        wrapper._witch_input_embeds_compat = True
        setattr(masking_utils, name, wrapper)


_patch_transformers_masking_utils()

# Patch Qwen3-TTS code2wav to handle malformed async chunk input
try:
    from .patch_code2wav import _patch_qwen3_tts_code2wav
    _patch_qwen3_tts_code2wav()
except Exception:
    pass
