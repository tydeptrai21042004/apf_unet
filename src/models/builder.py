from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional

from .registry import create_model


def _to_dict(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return deepcopy(config)
    return deepcopy(dict(config))


ALIASES = {
    "proposal": "proposal_fourier_unet",
    "fourier": "proposal_fourier_unet",
    "fourier_unet": "proposal_fourier_unet",
    "plain_fourier_unet": "proposal_fourier_unet",
    "urf": "proposal_urf_unet",
    "urf_unet": "proposal_urf_unet",
    "uncertainty_routed_fourier_unet": "proposal_urf_unet",
    # Backward-compatible model names used by earlier repository versions.
    "proposal_apf_unet": "proposal_fourier_unet",
    "fourier_unet_plain": "proposal_fourier_unet",
    "apf": "proposal_fourier_unet",
    "apf_unet": "proposal_fourier_unet",
    "amplitude_phase_fourier": "proposal_fourier_unet",
    "apf_amplitude_only": "fourier_unet_amplitude_only",
    "apf_phase_only": "fourier_unet_phase_only",
    "proposal_apf_unet_at_encoder1": "fourier_unet_at_encoder1",
    "apf_encoder1": "fourier_unet_at_encoder1",
}

# Old APF configuration keys are translated so archived YAML files and
# notebooks remain usable with the new canonical Fourier implementation.
LEGACY_CONFIG_KEYS = {
    "apf_alpha": "fourier_alpha",
    "apf_alpha_start": "fourier_alpha_start",
    "apf_alpha_warmup_epochs": "fourier_alpha_warmup_epochs",
    "apf_expansion": "fourier_expansion",
    "apf_dropout": "fourier_dropout",
    "apf_block_norm": "fourier_block_norm",
    "apf_block_act": "fourier_block_act",
    "apf_init_hw": "fourier_init_hw",
    "apf_amplitude_scale": "fourier_amplitude_scale",
    "apf_phase_max": "fourier_phase_max",
    "apf_use_amplitude": "fourier_use_amplitude",
    "apf_use_phase": "fourier_use_phase",
    "apf_zero_init_output": "fourier_zero_init_output",
    "apf_stage_index": "fourier_stage_index",
}


def _translate_legacy_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    translated = dict(cfg)
    for old_key, new_key in LEGACY_CONFIG_KEYS.items():
        if old_key in translated:
            if new_key in translated:
                raise ValueError(
                    f"Both legacy key {old_key!r} and canonical key "
                    f"{new_key!r} were provided"
                )
            translated[new_key] = translated.pop(old_key)
    return translated


def build_model(
    name: str,
    config: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
):
    cfg = _to_dict(config)
    cfg.pop("name", None)
    cfg.update(overrides)
    cfg.pop("name", None)
    cfg = _translate_legacy_config(cfg)
    model_name = ALIASES.get(name.lower(), name.lower())
    return create_model(model_name, **cfg)


__all__ = ["build_model"]
