from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.decoder import UNetDecoder
from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model


class AmplitudePhaseFourierBottleneck(nn.Module):
    """Bounded amplitude-phase Fourier residual correction.

    The block computes a real FFT, modulates amplitude and phase separately,
    reconstructs only the *spectral difference*, and adds that correction to
    the input. There is no feature-dependent gate.

    At initialization the final projection is exactly zero, so the block is an
    identity map while the small non-zero spectral logits keep the adapter
    trainable from the first optimization step.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.5,
        alpha: float = 0.5,
        dropout: float = 0.0,
        norm: str = "gn",
        act: str = "gelu",
        init_hw: Sequence[int] = (22, 22),
        amplitude_scale: float = 0.10,
        phase_max: float = math.pi / 4.0,
        use_amplitude: bool = True,
        use_phase: bool = True,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if len(init_hw) != 2 or min(int(init_hw[0]), int(init_hw[1])) <= 0:
            raise ValueError("init_hw must contain two positive integers")
        if not (use_amplitude or use_phase):
            raise ValueError("At least one of use_amplitude/use_phase must be enabled")

        hidden = max(channels, int(round(channels * float(expansion))))
        self.channels = int(channels)
        self.hidden_channels = int(hidden)
        self.alpha = float(alpha)
        self.amplitude_scale = float(amplitude_scale)
        self.phase_max = float(phase_max)
        self.use_amplitude = bool(use_amplitude)
        self.use_phase = bool(use_phase)
        self.zero_init_output = bool(zero_init_output)

        self.pre = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pre_norm = self._make_norm(norm, hidden)
        self.pre_act = self._make_act(act)

        h0, w0 = int(init_hw[0]), int(init_hw[1])
        # rfft2 stores W//2+1 frequency columns.
        rw0 = w0 // 2 + 1
        self.amplitude_logits = nn.Parameter(torch.empty(1, hidden, h0, rw0))
        self.phase_logits = nn.Parameter(torch.empty(1, hidden, h0, rw0))

        self.dropout = nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity()
        self.post = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        self.reset_parameters()

    @staticmethod
    def _make_norm(kind: str, channels: int) -> nn.Module:
        key = str(kind).lower()
        if key in {"none", "identity", ""}:
            return nn.Identity()
        if key == "bn":
            return nn.BatchNorm2d(channels)
        if key == "gn":
            groups = min(32, channels)
            while channels % groups != 0 and groups > 1:
                groups -= 1
            return nn.GroupNorm(groups, channels)
        raise ValueError(f"Unsupported norm: {kind}")

    @staticmethod
    def _make_act(kind: str) -> nn.Module:
        key = str(kind).lower()
        if key in {"none", "identity", ""}:
            return nn.Identity()
        if key == "relu":
            return nn.ReLU(inplace=True)
        if key == "gelu":
            return nn.GELU()
        if key in {"silu", "swish"}:
            return nn.SiLU(inplace=True)
        raise ValueError(f"Unsupported activation: {kind}")

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.pre.weight, mode="fan_out", nonlinearity="relu")
        # Small non-zero spectral perturbations avoid a dead zero-initialized
        # adapter while the zero post projection preserves exact identity.
        nn.init.normal_(self.amplitude_logits, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.phase_logits, mean=0.0, std=1.0e-3)
        if self.zero_init_output:
            nn.init.zeros_(self.post.weight)
            nn.init.zeros_(self.post.bias)
        else:
            nn.init.kaiming_normal_(self.post.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(self.post.bias)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _resize_logits(self, tensor: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor
        return F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_act(self.pre_norm(self.pre(x)))
        spatial_size = z.shape[-2:]
        spectrum = torch.fft.rfft2(z, dim=(-2, -1), norm="ortho")

        amplitude = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        freq_size = spectrum.shape[-2:]

        if self.use_amplitude:
            amp_logits = self._resize_logits(self.amplitude_logits, freq_size)
            amp_factor = 1.0 + self.amplitude_scale * torch.tanh(amp_logits)
            mod_amplitude = amplitude * amp_factor
        else:
            mod_amplitude = amplitude

        if self.use_phase:
            phase_logits = self._resize_logits(self.phase_logits, freq_size)
            mod_phase = phase + self.phase_max * torch.tanh(phase_logits)
        else:
            mod_phase = phase

        mod_spectrum = torch.polar(mod_amplitude, mod_phase)
        delta_spectrum = mod_spectrum - spectrum
        delta_spatial = torch.fft.irfft2(
            delta_spectrum,
            s=spatial_size,
            dim=(-2, -1),
            norm="ortho",
        )
        correction = self.post(self.dropout(delta_spatial))
        return x + self.alpha * correction

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        amp = self.amplitude_scale * torch.tanh(self.amplitude_logits)
        phase = self.phase_max * torch.tanh(self.phase_logits)
        return {
            "apf/alpha": float(self.alpha),
            "apf/amplitude_delta_abs_mean": float(amp.abs().mean().item()),
            "apf/amplitude_delta_abs_max": float(amp.abs().max().item()),
            "apf/phase_delta_abs_mean": float(phase.abs().mean().item()),
            "apf/phase_delta_abs_max": float(phase.abs().max().item()),
        }


class _APFUNetBase(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        apf_alpha: float = 0.5,
        apf_alpha_start: float = 0.0,
        apf_alpha_warmup_epochs: int = 10,
        apf_expansion: float = 1.5,
        apf_dropout: float = 0.0,
        apf_block_norm: str = "gn",
        apf_block_act: str = "gelu",
        apf_init_hw: Sequence[int] = (22, 22),
        apf_amplitude_scale: float = 0.10,
        apf_phase_max: float = math.pi / 4.0,
        apf_use_amplitude: bool = True,
        apf_use_phase: bool = True,
        apf_zero_init_output: bool = True,
        norm: str = "bn",
        act: str = "relu",
        decoder_use_cbam: bool = False,
        apf_stage_index: int = -1,
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            channels=channels,
            block="double",
            norm=norm,
            act=act,
        )
        self.apf_alpha_target = float(apf_alpha)
        self.apf_stage_index = int(apf_stage_index)
        if self.apf_stage_index < 0:
            self.apf_stage_index += len(channels)
        if not 0 <= self.apf_stage_index < len(channels):
            raise ValueError(
                f"apf_stage_index must select one of {len(channels)} encoder features, "
                f"got {apf_stage_index}"
            )
        self.apf_alpha_start = float(apf_alpha_start)
        self.apf_alpha_warmup_epochs = int(apf_alpha_warmup_epochs)

        self.apf_bottleneck = AmplitudePhaseFourierBottleneck(
            channels=channels[self.apf_stage_index],
            expansion=apf_expansion,
            alpha=apf_alpha,
            dropout=apf_dropout,
            norm=apf_block_norm,
            act=apf_block_act,
            init_hw=apf_init_hw,
            amplitude_scale=apf_amplitude_scale,
            phase_max=apf_phase_max,
            use_amplitude=apf_use_amplitude,
            use_phase=apf_use_phase,
            zero_init_output=apf_zero_init_output,
        )
        self.decoder = UNetDecoder(
            channels=channels,
            norm=norm,
            act=act,
            use_cbam=decoder_use_cbam,
        )
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        init_weights(self)
        # Global initialization touches Conv2d layers, so restore the defining
        # identity-preserving APF initialization afterwards.
        self.apf_bottleneck.reset_parameters()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.apf_alpha_warmup_epochs <= 0:
            alpha = self.apf_alpha_target
        else:
            progress = min(max(float(epoch), 0.0) / float(self.apf_alpha_warmup_epochs), 1.0)
            alpha = self.apf_alpha_start + (
                self.apf_alpha_target - self.apf_alpha_start
            ) * progress
        self.apf_bottleneck.set_alpha(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feats[self.apf_stage_index] = self.apf_bottleneck(
            feats[self.apf_stage_index]
        )
        dec = self.decoder(feats)
        return self.seg_head(dec)

    def auxiliary_regularization(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def diagnostic_metrics(self) -> dict[str, float]:
        return self.apf_bottleneck.diagnostics()


@register_model("proposal_apf_unet")
class APFUNet(_APFUNetBase):
    """Main amplitude-phase Fourier U-Net proposal, without a gate."""




@register_model("proposal_apf_unet_at_encoder1")
class APFUNetAtEncoder1(_APFUNetBase):
    """APF-U-Net with the APF block after encoder stage 1.

    Encoder stage 1 is the second encoder feature map (``feats[1]``).
    This placement preserves more spatial detail than the bottleneck while
    keeping the same no-gate amplitude--phase Fourier correction operator.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["apf_stage_index"] = 1
        super().__init__(*args, **kwargs)


@register_model("apf_amplitude_only")
class APFAmplitudeOnlyUNet(_APFUNetBase):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["apf_use_amplitude"] = True
        kwargs["apf_use_phase"] = False
        super().__init__(*args, **kwargs)


@register_model("apf_phase_only")
class APFPhaseOnlyUNet(_APFUNetBase):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["apf_use_amplitude"] = False
        kwargs["apf_use_phase"] = True
        super().__init__(*args, **kwargs)


@register_model("fourier_unet_plain")
class PlainFourierUNet(_APFUNetBase):
    """Fourier-only no-gate control with unrestricted larger corrections."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("apf_amplitude_scale", 1.0)
        kwargs.setdefault("apf_phase_max", math.pi)
        super().__init__(*args, **kwargs)
