from __future__ import annotations

import builtins
import functools
import importlib
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

_ORIGINAL_IMPORT = builtins.__import__
_ORIGINAL_IMPORT_MODULE = importlib.import_module
_QWEN3_CODE2WAV_MODULE = "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav"


def _patch_qwen3_code2wav_when_available() -> bool:
    try:
        from patch_code2wav import _patch_qwen3_tts_code2wav

        return bool(_patch_qwen3_tts_code2wav())
    except Exception:
        return False


def _import_module_with_qwen3_patch(name: str, package: str | None = None):
    module = _ORIGINAL_IMPORT_MODULE(name, package)
    if name == _QWEN3_CODE2WAV_MODULE:
        _patch_qwen3_code2wav_when_available()
    return module


def _import_with_qwen3_patch(name, globals=None, locals=None, fromlist=(), level=0):
    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if name == _QWEN3_CODE2WAV_MODULE or (
        name == "vllm_omni.model_executor.models.qwen3_tts"
        and "qwen3_tts_code2wav" in fromlist
    ):
        _patch_qwen3_code2wav_when_available()
    return module


# At sitecustomize time the model module may not be loaded yet, so keep an
# import hook active until the target module appears.
try:
    if not _patch_qwen3_code2wav_when_available():
        builtins.__import__ = _import_with_qwen3_patch
        importlib.import_module = _import_module_with_qwen3_patch
except Exception:
    pass
