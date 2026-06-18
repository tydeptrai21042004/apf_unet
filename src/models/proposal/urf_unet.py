from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.encoder import PyramidEncoder
from ..common.utils import init_weights
from ..registry import register_model
from .fourier_unet import FourierSpectralBottleneck


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


def _radial_basis(
    height: int,
    width: int,
    num_bands: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_center: float = 0.0,
) -> torch.Tensor:
    """Return smooth normalized radial bands for an rFFT grid.

    The returned tensor has shape ``[1, K, H, W//2+1]`` and is generated at
    runtime, so the spectral modules remain resolution independent.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if num_bands <= 0:
        raise ValueError("num_bands must be positive")

    fy = torch.fft.fftfreq(height, device=device, dtype=dtype).abs()
    fx = torch.fft.rfftfreq(width, device=device, dtype=dtype).abs()
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    radius = radius / radius.max().clamp_min(torch.finfo(dtype).eps)

    centers = torch.linspace(
        float(min_center),
        1.0,
        num_bands,
        device=device,
        dtype=dtype,
    )
    sigma = max((1.0 - float(min_center)) / max(num_bands - 1, 1), 0.18)
    basis = torch.exp(
        -0.5 * ((radius[None, ...] - centers[:, None, None]) / sigma).square()
    )
    basis = basis / basis.sum(dim=0, keepdim=True).clamp_min(
        torch.finfo(dtype).eps
    )
    return basis.unsqueeze(0)


class AdaptiveRadialFourierBottleneck(nn.Module):
    """Image-conditioned global Fourier mixer with smooth radial responses.

    Unlike the static frequency map used by :class:`FourierSpectralBottleneck`,
    this block predicts a small set of per-image radial band coefficients.  The
    coefficients are expanded into a smooth frequency response and applied to
    the bottleneck spectrum.  The block starts as an identity mapping through a
    zero-initialized residual projection.
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 1.25,
        alpha: float = 0.25,
        dropout: float = 0.05,
        norm: str = "gn",
        act: str = "gelu",
        num_bands: int = 4,
        amplitude_scale: float = 0.35,
        phase_max: float = 0.0,
        use_channel_mixing: bool = True,
        zero_init_output: bool = True,
        context_reduction: int = 4,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if amplitude_scale < 0 or phase_max < 0:
            raise ValueError("response limits must be non-negative")

        hidden = max(channels, int(round(channels * float(expansion))))
        context_channels = max(hidden // max(int(context_reduction), 1), 16)

        self.channels = int(channels)
        self.hidden_channels = int(hidden)
        self.alpha = float(alpha)
        self.num_bands = int(num_bands)
        self.amplitude_scale = float(amplitude_scale)
        self.phase_max = float(phase_max)
        self.use_channel_mixing = bool(use_channel_mixing)
        self.zero_init_output = bool(zero_init_output)

        self.pre = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pre_norm = _make_norm(norm, hidden)
        self.pre_act = _make_act(act)

        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, context_channels, kernel_size=1),
            _make_act(act),
        )
        self.amplitude_predictor = nn.Conv2d(
            context_channels,
            self.num_bands,
            kernel_size=1,
        )
        self.phase_predictor = (
            nn.Conv2d(context_channels, self.num_bands, kernel_size=1)
            if self.phase_max > 0
            else None
        )

        if self.use_channel_mixing:
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
        self._last_diagnostics: dict[str, float] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.pre.weight, mode="fan_out", nonlinearity="relu")
        for module in self.context.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # A tiny non-zero response is required when the residual output
        # projection is zero-initialized.  An exactly unit response together
        # with a zero post projection would make both layers receive zero
        # gradient on the first optimization step.
        nn.init.normal_(self.amplitude_predictor.weight, mean=0.0, std=1.0e-3)
        nn.init.normal_(self.amplitude_predictor.bias, mean=0.0, std=1.0e-3)
        if self.phase_predictor is not None:
            nn.init.normal_(self.phase_predictor.weight, mean=0.0, std=1.0e-3)
            nn.init.normal_(self.phase_predictor.bias, mean=0.0, std=1.0e-3)

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
            nn.init.kaiming_normal_(self.post.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(self.post.bias)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pre_act(self.pre_norm(self.pre(x)))
        height, width = z.shape[-2:]
        spectrum = torch.fft.rfft2(z, dim=(-2, -1), norm="ortho")
        basis = _radial_basis(
            height,
            width,
            self.num_bands,
            device=z.device,
            dtype=z.dtype,
        )

        context = self.context(z)
        amplitude_weights = torch.tanh(self.amplitude_predictor(context))
        log_gain = self.amplitude_scale * torch.sum(
            amplitude_weights * basis,
            dim=1,
            keepdim=True,
        )
        gain = torch.exp(log_gain)

        if self.phase_predictor is not None:
            phase_weights = torch.tanh(self.phase_predictor(context))
            phase = self.phase_max * torch.sum(
                phase_weights * basis,
                dim=1,
                keepdim=True,
            )
        else:
            phase = torch.zeros_like(gain)

        response = torch.polar(gain, phase)
        transformed = spectrum * response
        if self.use_channel_mixing:
            transformed = torch.complex(
                self.spectral_channel_mixer(transformed.real),
                self.spectral_channel_mixer(transformed.imag),
            )

        correction_spectrum = transformed - spectrum
        correction = torch.fft.irfft2(
            correction_spectrum,
            s=(height, width),
            dim=(-2, -1),
            norm="ortho",
        )
        correction = self.post(self.dropout(correction))

        with torch.no_grad():
            self._last_diagnostics = {
                "urf/global_alpha": float(self.alpha),
                "urf/global_gain_mean": float(gain.mean().item()),
                "urf/global_gain_min": float(gain.min().item()),
                "urf/global_gain_max": float(gain.max().item()),
                "urf/global_phase_abs_mean": float(phase.abs().mean().item()),
            }
        return x + self.alpha * correction

    def diagnostics(self) -> dict[str, float]:
        return dict(self._last_diagnostics)


class UncertaintyRoutedLocalFourierRefiner(nn.Module):
    """Windowed high-frequency correction routed by prediction uncertainty."""

    def __init__(
        self,
        channels: int,
        alpha: float = 0.20,
        window_size: int = 8,
        num_bands: int = 3,
        response_scale: float = 0.35,
        routing_floor: float = 0.10,
        use_uncertainty: bool = True,
        dropout: float = 0.05,
        norm: str = "gn",
        act: str = "gelu",
        context_reduction: int = 4,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if window_size <= 1:
            raise ValueError("window_size must be greater than one")
        if num_bands <= 0:
            raise ValueError("num_bands must be positive")
        if not 0.0 <= routing_floor <= 1.0:
            raise ValueError("routing_floor must be in [0, 1]")

        hidden = int(channels)
        context_channels = max(hidden // max(int(context_reduction), 1), 16)
        self.channels = int(channels)
        self.alpha = float(alpha)
        self.window_size = int(window_size)
        self.num_bands = int(num_bands)
        self.response_scale = float(response_scale)
        self.routing_floor = float(routing_floor)
        self.use_uncertainty = bool(use_uncertainty)
        self.zero_init_output = bool(zero_init_output)

        self.pre = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.pre_norm = _make_norm(norm, hidden)
        self.pre_act = _make_act(act)
        self.band_predictor = nn.Sequential(
            nn.Linear(hidden, context_channels),
            _make_act(act),
            nn.Linear(context_channels, self.num_bands),
        )
        self.dropout = (
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        )
        self.post = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self._last_uncertainty: torch.Tensor | None = None
        self._last_diagnostics: dict[str, float] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.pre.weight, mode="fan_out", nonlinearity="relu")
        for module in self.band_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        if self.zero_init_output:
            nn.init.zeros_(self.post.weight)
            nn.init.zeros_(self.post.bias)
        else:
            nn.init.kaiming_normal_(self.post.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(self.post.bias)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def _windowize(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int, int]:
        batch, channels, height, width = x.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        padded_h, padded_w = padded.shape[-2:]
        rows = padded_h // ws
        cols = padded_w // ws
        windows = (
            padded.view(batch, channels, rows, ws, cols, ws)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(batch * rows * cols, channels, ws, ws)
        )
        return windows, rows, cols, height, width

    def _dewindowize(
        self,
        windows: torch.Tensor,
        batch: int,
        rows: int,
        cols: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        channels = windows.shape[1]
        ws = self.window_size
        x = (
            windows.view(batch, rows, cols, channels, ws, ws)
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(batch, channels, rows * ws, cols * ws)
        )
        return x[..., :height, :width]

    def forward(
        self,
        x: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        z = self.pre_act(self.pre_norm(self.pre(x)))
        batch = z.shape[0]
        windows, rows, cols, height, width = self._windowize(z)
        ws = self.window_size

        spectrum = torch.fft.rfft2(windows, dim=(-2, -1), norm="ortho")
        basis = _radial_basis(
            ws,
            ws,
            self.num_bands,
            device=z.device,
            dtype=z.dtype,
            min_center=0.35,
        )
        pooled = windows.mean(dim=(-2, -1))
        weights = torch.tanh(self.band_predictor(pooled))
        log_gain = self.response_scale * torch.sum(
            weights[:, :, None, None] * basis.squeeze(0),
            dim=1,
            keepdim=True,
        )
        gain = torch.exp(log_gain)
        transformed = spectrum * gain
        correction_windows = torch.fft.irfft2(
            transformed - spectrum,
            s=(ws, ws),
            dim=(-2, -1),
            norm="ortho",
        )

        if self.use_uncertainty:
            probability = torch.sigmoid(coarse_logits)
            uncertainty = 4.0 * probability * (1.0 - probability)
            uncertainty = self.routing_floor + (
                1.0 - self.routing_floor
            ) * uncertainty
        else:
            uncertainty = torch.ones_like(coarse_logits)

        uncertainty_windows, _, _, _, _ = self._windowize(uncertainty)
        window_gate = uncertainty_windows.mean(dim=(-2, -1), keepdim=True)
        correction_windows = correction_windows * window_gate
        correction = self._dewindowize(
            correction_windows,
            batch,
            rows,
            cols,
            height,
            width,
        )
        correction = self.post(self.dropout(correction))

        self._last_uncertainty = uncertainty
        with torch.no_grad():
            self._last_diagnostics = {
                "urf/local_alpha": float(self.alpha),
                "urf/local_gate_mean": float(window_gate.mean().item()),
                "urf/local_gate_min": float(window_gate.min().item()),
                "urf/local_gate_max": float(window_gate.max().item()),
                "urf/local_gain_mean": float(gain.mean().item()),
                "urf/local_gain_min": float(gain.min().item()),
                "urf/local_gain_max": float(gain.max().item()),
            }
        return x + self.alpha * correction

    @property
    def last_uncertainty(self) -> torch.Tensor | None:
        return self._last_uncertainty

    def diagnostics(self) -> dict[str, float]:
        return dict(self._last_diagnostics)


class _URFUNetBase(nn.Module):
    """U-Net with adaptive global and uncertainty-routed local Fourier blocks."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        channels: tuple[int, ...] = (32, 64, 128, 256, 512),
        norm: str = "bn",
        act: str = "relu",
        # Global adaptive Fourier branch.
        urf_global_mode: str = "adaptive",
        urf_global_alpha: float = 0.25,
        urf_global_alpha_start: float = 0.0,
        urf_global_alpha_warmup_epochs: int = 10,
        urf_global_expansion: float = 1.25,
        urf_global_dropout: float = 0.05,
        urf_global_num_bands: int = 4,
        urf_global_amplitude_scale: float = 0.35,
        urf_global_phase_max: float = 0.0,
        urf_global_use_channel_mixing: bool = True,
        # Plain Fourier settings used by the no-dynamic-global ablation.
        fourier_init_hw: Sequence[int] = (22, 22),
        fourier_amplitude_scale: float = 0.35,
        fourier_phase_max: float = math.pi / 4.0,
        # Local uncertainty-routed branch.
        urf_use_local_refiner: bool = True,
        urf_use_uncertainty: bool = True,
        urf_local_decoder_index: int = 1,
        urf_local_alpha: float = 0.20,
        urf_local_alpha_start: float = 0.0,
        urf_local_alpha_warmup_epochs: int = 10,
        urf_local_window_size: int = 8,
        urf_local_num_bands: int = 3,
        urf_local_response_scale: float = 0.35,
        urf_routing_floor: float = 0.10,
        urf_local_dropout: float = 0.05,
        urf_return_boundary: bool = True,
    ) -> None:
        super().__init__()
        channels = tuple(int(value) for value in channels)
        if len(channels) < 3:
            raise ValueError("URF-U-Net requires at least three encoder stages")

        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            channels=channels,
            block="double",
            norm=norm,
            act=act,
        )

        mode = str(urf_global_mode).lower()
        if mode == "adaptive":
            self.global_fourier: nn.Module = AdaptiveRadialFourierBottleneck(
                channels=channels[-1],
                expansion=urf_global_expansion,
                alpha=urf_global_alpha,
                dropout=urf_global_dropout,
                num_bands=urf_global_num_bands,
                amplitude_scale=urf_global_amplitude_scale,
                phase_max=urf_global_phase_max,
                use_channel_mixing=urf_global_use_channel_mixing,
            )
        elif mode == "plain":
            self.global_fourier = FourierSpectralBottleneck(
                channels=channels[-1],
                expansion=urf_global_expansion,
                alpha=urf_global_alpha,
                dropout=urf_global_dropout,
                init_hw=fourier_init_hw,
                amplitude_scale=fourier_amplitude_scale,
                phase_max=fourier_phase_max,
                use_amplitude=True,
                use_phase=True,
                use_channel_mixing=urf_global_use_channel_mixing,
                residual=True,
                zero_init_output=True,
            )
        elif mode == "identity":
            self.global_fourier = nn.Identity()
        else:
            raise ValueError(
                "urf_global_mode must be one of {'adaptive', 'plain', 'identity'}"
            )
        self.urf_global_mode = mode

        from ..common.decoder import UNetDecoder

        self.decoder = UNetDecoder(channels=channels, norm=norm, act=act)
        self.urf_use_local_refiner = bool(urf_use_local_refiner)
        self.urf_use_uncertainty = bool(urf_use_uncertainty)
        self.urf_return_boundary = bool(urf_return_boundary)
        self.urf_local_decoder_index = int(urf_local_decoder_index)
        if not 0 <= self.urf_local_decoder_index < len(self.decoder.blocks):
            raise ValueError(
                "urf_local_decoder_index must select a decoder block, got "
                f"{urf_local_decoder_index}"
            )

        local_channels = channels[-2 - self.urf_local_decoder_index]
        self.coarse_head = nn.Conv2d(local_channels, num_classes, kernel_size=1)
        self.local_refiner = (
            UncertaintyRoutedLocalFourierRefiner(
                channels=local_channels,
                alpha=urf_local_alpha,
                window_size=urf_local_window_size,
                num_bands=urf_local_num_bands,
                response_scale=urf_local_response_scale,
                routing_floor=urf_routing_floor,
                use_uncertainty=urf_use_uncertainty,
                dropout=urf_local_dropout,
            )
            if self.urf_use_local_refiner
            else None
        )

        self.seg_head = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        boundary_hidden = max(channels[0] // 2, 8)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(channels[0], boundary_hidden, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, boundary_hidden),
            _make_act(act),
            nn.Conv2d(boundary_hidden, num_classes, kernel_size=1),
        )

        self.urf_global_alpha_target = float(urf_global_alpha)
        self.urf_global_alpha_start = float(urf_global_alpha_start)
        self.urf_global_alpha_warmup_epochs = int(urf_global_alpha_warmup_epochs)
        self.urf_local_alpha_target = float(urf_local_alpha)
        self.urf_local_alpha_start = float(urf_local_alpha_start)
        self.urf_local_alpha_warmup_epochs = int(urf_local_alpha_warmup_epochs)

        init_weights(self)
        if hasattr(self.global_fourier, "reset_parameters"):
            self.global_fourier.reset_parameters()
        if self.local_refiner is not None:
            self.local_refiner.reset_parameters()
        self.set_epoch(0)

    @staticmethod
    def _warmup_value(
        epoch: int,
        start: float,
        target: float,
        warmup_epochs: int,
    ) -> float:
        if warmup_epochs <= 0:
            return float(target)
        progress = min(max(float(epoch), 0.0) / float(warmup_epochs), 1.0)
        return float(start + (target - start) * progress)

    def set_epoch(self, epoch: int) -> None:
        global_alpha = self._warmup_value(
            epoch,
            self.urf_global_alpha_start,
            self.urf_global_alpha_target,
            self.urf_global_alpha_warmup_epochs,
        )
        if hasattr(self.global_fourier, "set_alpha"):
            self.global_fourier.set_alpha(global_alpha)

        if self.local_refiner is not None:
            local_alpha = self._warmup_value(
                epoch,
                self.urf_local_alpha_start,
                self.urf_local_alpha_target,
                self.urf_local_alpha_warmup_epochs,
            )
            self.local_refiner.set_alpha(local_alpha)

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        features[-1] = self.global_fourier(features[-1])

        decoded = features[-1]
        skips = list(reversed(features[:-1]))
        coarse_logits: torch.Tensor | None = None
        for index, (block, skip) in enumerate(zip(self.decoder.blocks, skips)):
            decoded = block(decoded, skip)
            if index == self.urf_local_decoder_index:
                coarse_logits = self.coarse_head(decoded)
                if self.local_refiner is not None:
                    decoded = self.local_refiner(decoded, coarse_logits)

        main = self.seg_head(decoded)
        boundary = self.boundary_head(decoded) if self.urf_return_boundary else None
        if coarse_logits is None:
            raise RuntimeError("The configured local decoder stage was not reached")
        coarse_up = F.interpolate(
            coarse_logits,
            size=main.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        output = {
            "main": main,
            "aux": [coarse_up],
        }
        if boundary is not None:
            output["boundary"] = boundary
        if self.local_refiner is not None and self.local_refiner.last_uncertainty is not None:
            output["uncertainty"] = F.interpolate(
                self.local_refiner.last_uncertainty,
                size=main.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return output

    def auxiliary_regularization(self) -> torch.Tensor:
        return torch.zeros((), device=next(self.parameters()).device)

    def diagnostics(self) -> dict[str, float]:
        values: dict[str, float] = {
            "urf/use_local_refiner": float(self.urf_use_local_refiner),
            "urf/use_uncertainty": float(self.urf_use_uncertainty),
        }
        if hasattr(self.global_fourier, "diagnostics"):
            values.update(self.global_fourier.diagnostics())
        if self.local_refiner is not None:
            values.update(self.local_refiner.diagnostics())
        return values

    def diagnostic_metrics(self) -> dict[str, float]:
        return self.diagnostics()


@register_model("proposal_urf_unet")
class URFUNet(_URFUNetBase):
    """Full Uncertainty-Routed Local--Global Fourier U-Net."""


@register_model("urf_unet_dynamic_global_only")
class URFDynamicGlobalOnly(_URFUNetBase):
    """Ablation retaining only the adaptive global Fourier branch."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["urf_use_local_refiner"] = False
        super().__init__(*args, **kwargs)


@register_model("urf_unet_no_uncertainty")
class URFNoUncertainty(_URFUNetBase):
    """Ablation using local Fourier refinement without uncertainty routing."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["urf_use_uncertainty"] = False
        super().__init__(*args, **kwargs)


@register_model("urf_unet_no_dynamic_global")
class URFNoDynamicGlobal(_URFUNetBase):
    """Ablation replacing the adaptive global response with plain Fourier."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["urf_global_mode"] = "plain"
        super().__init__(*args, **kwargs)


@register_model("urf_unet_no_coarse_supervision")
class URFNoCoarseSupervision(_URFUNetBase):
    """Architecture-identical control trained without coarse auxiliary loss."""



@register_model("urf_unet_no_boundary_supervision")
class URFNoBoundarySupervision(_URFUNetBase):
    """Architecture-identical control trained without boundary supervision."""


__all__ = [
    "AdaptiveRadialFourierBottleneck",
    "UncertaintyRoutedLocalFourierRefiner",
    "URFUNet",
    "URFDynamicGlobalOnly",
    "URFNoUncertainty",
    "URFNoDynamicGlobal",
    "URFNoCoarseSupervision",
    "URFNoBoundarySupervision",
]
