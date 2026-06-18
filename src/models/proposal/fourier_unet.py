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


class FourierSpectralBottleneck(nn.Module):
    """Plain Fourier residual mixer for real-valued feature maps.

    The block projects the input to a hidden channel space, applies an
    orthonormal real 2-D FFT, learns a frequency-dependent amplitude gain and
    phase shift, optionally mixes channels at every frequency, and transforms
    the result back to the spatial domain.

    By default the block predicts only the spectral correction and adds it to
    the input through a residual connection.  The amplitude response is
    parameterized with an exponential gain, so it always remains positive.
    The spectral channel mixer is initialized to the identity.  Together with
    a zero-initialized output projection, the residual version starts as an
    exact identity map while remaining trainable.
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
        amplitude_scale: float = 1.0,
        phase_max: float = math.pi,
        use_amplitude: bool = True,
        use_phase: bool = True,
        use_channel_mixing: bool = True,
        residual: bool = True,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if len(init_hw) != 2 or min(int(init_hw[0]), int(init_hw[1])) <= 0:
            raise ValueError("init_hw must contain two positive integers")
        if not (use_amplitude or use_phase or use_channel_mixing):
            raise ValueError(
                "At least one spectral operation must be enabled: amplitude, "
                "phase, or channel mixing"
            )
        if not residual and zero_init_output:
            raise ValueError(
                "zero_init_output=True is only valid for the residual block; "
                "a non-residual block would otherwise start as the zero map"
            )
        if amplitude_scale < 0:
            raise ValueError("amplitude_scale must be non-negative")
        if phase_max < 0:
            raise ValueError("phase_max must be non-negative")

        hidden = max(channels, int(round(channels * float(expansion))))
        self.channels = int(channels)
        self.hidden_channels = int(hidden)
        self.alpha = float(alpha)
        self.amplitude_scale = float(amplitude_scale)
        self.phase_max = float(phase_max)
        self.use_amplitude = bool(use_amplitude)
        self.use_phase = bool(use_phase)
        self.use_channel_mixing = bool(use_channel_mixing)
        self.residual = bool(residual)
        self.zero_init_output = bool(zero_init_output)

        self.pre = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pre_norm = self._make_norm(norm, hidden)
        self.pre_act = self._make_act(act)

        h0, w0 = int(init_hw[0]), int(init_hw[1])
        rfft_width = w0 // 2 + 1
        self.amplitude_logits = nn.Parameter(
            torch.empty(1, hidden, h0, rfft_width)
        )
        self.phase_logits = nn.Parameter(
            torch.empty(1, hidden, h0, rfft_width)
        )

        if self.use_channel_mixing:
            # The same real-valued channel matrix is applied to the real and
            # imaginary components.  This preserves a valid real-valued output
            # after irfft2 while allowing cross-channel spectral interaction.
            self.spectral_channel_mixer: nn.Module = nn.Conv2d(
                hidden,
                hidden,
                kernel_size=1,
                bias=False,
            )
        else:
            self.spectral_channel_mixer = nn.Identity()

        self.dropout = (
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        )
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
        nn.init.kaiming_normal_(
            self.pre.weight,
            mode="fan_out",
            nonlinearity="relu",
        )
        # Small non-zero logits avoid a perfectly flat spectral branch while
        # keeping the initial transform extremely close to the identity.
        nn.init.normal_(self.amplitude_logits, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.phase_logits, mean=0.0, std=1.0e-3)

        if isinstance(self.spectral_channel_mixer, nn.Conv2d):
            nn.init.zeros_(self.spectral_channel_mixer.weight)
            eye = torch.eye(
                self.hidden_channels,
                dtype=self.spectral_channel_mixer.weight.dtype,
            )
            self.spectral_channel_mixer.weight.data[:, :, 0, 0].copy_(eye)

        if self.zero_init_output:
            nn.init.zeros_(self.post.weight)
            nn.init.zeros_(self.post.bias)
        else:
            nn.init.kaiming_normal_(
                self.post.weight,
                mode="fan_in",
                nonlinearity="linear",
            )
            nn.init.zeros_(self.post.bias)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    @staticmethod
    def _resize_parameter(
        tensor: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor
        return F.interpolate(
            tensor,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _spectral_response(
        self,
        spectrum: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frequency_size = spectrum.shape[-2:]

        if self.use_amplitude:
            amplitude_logits = self._resize_parameter(
                self.amplitude_logits,
                frequency_size,
            )
            log_gain = self.amplitude_scale * torch.tanh(amplitude_logits)
            amplitude_gain = torch.exp(log_gain)
        else:
            amplitude_gain = torch.ones_like(spectrum.real)

        if self.use_phase:
            phase_logits = self._resize_parameter(
                self.phase_logits,
                frequency_size,
            )
            phase_shift = self.phase_max * torch.tanh(phase_logits)
        else:
            phase_shift = torch.zeros_like(spectrum.real)

        response = torch.polar(amplitude_gain, phase_shift)
        transformed = spectrum * response

        if self.use_channel_mixing:
            transformed = torch.complex(
                self.spectral_channel_mixer(transformed.real),
                self.spectral_channel_mixer(transformed.imag),
            )

        return transformed, amplitude_gain, phase_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_act(self.pre_norm(self.pre(x)))
        spatial_size = z.shape[-2:]
        spectrum = torch.fft.rfft2(
            z,
            dim=(-2, -1),
            norm="ortho",
        )
        transformed_spectrum, _, _ = self._spectral_response(spectrum)

        if self.residual:
            spatial_signal = torch.fft.irfft2(
                transformed_spectrum - spectrum,
                s=spatial_size,
                dim=(-2, -1),
                norm="ortho",
            )
            correction = self.post(self.dropout(spatial_signal))
            return x + self.alpha * correction

        spatial_signal = torch.fft.irfft2(
            transformed_spectrum,
            s=spatial_size,
            dim=(-2, -1),
            norm="ortho",
        )
        return self.post(self.dropout(spatial_signal))

    @torch.no_grad()
    def diagnostics(self) -> dict[str, float]:
        log_gain = self.amplitude_scale * torch.tanh(self.amplitude_logits)
        gain = torch.exp(log_gain)
        phase = self.phase_max * torch.tanh(self.phase_logits)

        if isinstance(self.spectral_channel_mixer, nn.Conv2d):
            matrix = self.spectral_channel_mixer.weight[:, :, 0, 0]
            identity = torch.eye(
                matrix.shape[0],
                device=matrix.device,
                dtype=matrix.dtype,
            )
            mixer_delta = (matrix - identity).abs()
            mixer_delta_mean = float(mixer_delta.mean().item())
            mixer_delta_max = float(mixer_delta.max().item())
        else:
            mixer_delta_mean = 0.0
            mixer_delta_max = 0.0

        return {
            "fourier/alpha": float(self.alpha),
            "fourier/amplitude_gain_mean": float(gain.mean().item()),
            "fourier/amplitude_gain_min": float(gain.min().item()),
            "fourier/amplitude_gain_max": float(gain.max().item()),
            "fourier/phase_shift_abs_mean": float(phase.abs().mean().item()),
            "fourier/phase_shift_abs_max": float(phase.abs().max().item()),
            "fourier/channel_mixer_delta_abs_mean": mixer_delta_mean,
            "fourier/channel_mixer_delta_abs_max": mixer_delta_max,
        }


class _FourierUNetBase(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        fourier_alpha: float = 0.5,
        fourier_alpha_start: float = 0.5,
        fourier_alpha_warmup_epochs: int = 0,
        fourier_expansion: float = 1.5,
        fourier_dropout: float = 0.1,
        fourier_block_norm: str = "gn",
        fourier_block_act: str = "gelu",
        fourier_init_hw: Sequence[int] = (22, 22),
        fourier_amplitude_scale: float = 1.0,
        fourier_phase_max: float = math.pi,
        fourier_use_amplitude: bool = True,
        fourier_use_phase: bool = True,
        fourier_use_channel_mixing: bool = True,
        fourier_residual: bool = True,
        fourier_zero_init_output: bool = True,
        norm: str = "bn",
        act: str = "relu",
        decoder_use_cbam: bool = False,
        fourier_stage_index: int = -1,
    ) -> None:
        super().__init__()
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            channels=channels,
            block="double",
            norm=norm,
            act=act,
        )

        self.fourier_alpha_target = float(fourier_alpha)
        self.fourier_alpha_start = float(fourier_alpha_start)
        self.fourier_alpha_warmup_epochs = int(fourier_alpha_warmup_epochs)

        self.fourier_stage_index = int(fourier_stage_index)
        if self.fourier_stage_index < 0:
            self.fourier_stage_index += len(channels)
        if not 0 <= self.fourier_stage_index < len(channels):
            raise ValueError(
                "fourier_stage_index must select one of "
                f"{len(channels)} encoder features, got {fourier_stage_index}"
            )

        self.fourier_bottleneck = FourierSpectralBottleneck(
            channels=channels[self.fourier_stage_index],
            expansion=fourier_expansion,
            alpha=fourier_alpha,
            dropout=fourier_dropout,
            norm=fourier_block_norm,
            act=fourier_block_act,
            init_hw=fourier_init_hw,
            amplitude_scale=fourier_amplitude_scale,
            phase_max=fourier_phase_max,
            use_amplitude=fourier_use_amplitude,
            use_phase=fourier_use_phase,
            use_channel_mixing=fourier_use_channel_mixing,
            residual=fourier_residual,
            zero_init_output=fourier_zero_init_output,
        )
        self.decoder = UNetDecoder(
            channels=channels,
            norm=norm,
            act=act,
            use_cbam=decoder_use_cbam,
        )
        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        init_weights(self)
        # init_weights touches every Conv2d layer. Restore the Fourier-specific
        # identity initialization after the global network initialization.
        self.fourier_bottleneck.reset_parameters()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if self.fourier_alpha_warmup_epochs <= 0:
            alpha = self.fourier_alpha_target
        else:
            progress = min(
                max(float(epoch), 0.0)
                / float(self.fourier_alpha_warmup_epochs),
                1.0,
            )
            alpha = self.fourier_alpha_start + (
                self.fourier_alpha_target - self.fourier_alpha_start
            ) * progress
        self.fourier_bottleneck.set_alpha(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features[self.fourier_stage_index] = self.fourier_bottleneck(
            features[self.fourier_stage_index]
        )
        decoded = self.decoder(features)
        return self.seg_head(decoded)

    def auxiliary_regularization(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def diagnostic_metrics(self) -> dict[str, float]:
        return self.fourier_bottleneck.diagnostics()

    # Compatibility properties for old analysis scripts.
    @property
    def apf_bottleneck(self) -> FourierSpectralBottleneck:
        return self.fourier_bottleneck

    @property
    def apf_stage_index(self) -> int:
        return self.fourier_stage_index


@register_model("proposal_fourier_unet")
class FourierUNet(_FourierUNetBase):
    """Proposed plain Fourier U-Net at the deepest encoder feature."""


@register_model("fourier_unet_bounded")
class BoundedFourierUNet(_FourierUNetBase):
    """Ablation using the former conservative APF response bounds."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_amplitude_scale"] = 0.10
        kwargs["fourier_phase_max"] = math.pi / 4.0
        super().__init__(*args, **kwargs)


@register_model("fourier_unet_amplitude_only")
class FourierAmplitudeOnlyUNet(_FourierUNetBase):
    """Ablation that learns amplitude gains but no phase shifts."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_amplitude"] = True
        kwargs["fourier_use_phase"] = False
        super().__init__(*args, **kwargs)


@register_model("fourier_unet_phase_only")
class FourierPhaseOnlyUNet(_FourierUNetBase):
    """Ablation that learns phase shifts but no amplitude gains."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_amplitude"] = False
        kwargs["fourier_use_phase"] = True
        super().__init__(*args, **kwargs)


@register_model("fourier_unet_no_channel_mix")
class FourierNoChannelMixUNet(_FourierUNetBase):
    """Ablation without cross-channel mixing in the spectral domain."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_use_channel_mixing"] = False
        super().__init__(*args, **kwargs)


@register_model("fourier_unet_no_residual")
class FourierNoResidualUNet(_FourierUNetBase):
    """Ablation that replaces the feature instead of adding a correction."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_residual"] = False
        kwargs["fourier_zero_init_output"] = False
        super().__init__(*args, **kwargs)


@register_model("fourier_unet_at_encoder1")
class FourierUNetAtEncoder1(_FourierUNetBase):
    """Placement ablation applying the Fourier block after encoder stage 1."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["fourier_stage_index"] = 1
        super().__init__(*args, **kwargs)


__all__ = [
    "FourierSpectralBottleneck",
    "FourierUNet",
    "BoundedFourierUNet",
    "FourierAmplitudeOnlyUNet",
    "FourierPhaseOnlyUNet",
    "FourierNoChannelMixUNet",
    "FourierNoResidualUNet",
    "FourierUNetAtEncoder1",
]
