"""Backward-compatible imports for the former APF model names.

The canonical implementation is now :mod:`fourier_unet` and the proposed model
is ``proposal_fourier_unet``.  These Python aliases keep older notebooks from
breaking; new experiments should use the Fourier names.
"""

from .fourier_unet import (
    BoundedFourierUNet,
    FourierAmplitudeOnlyUNet,
    FourierNoChannelMixUNet,
    FourierNoResidualUNet,
    FourierPhaseOnlyUNet,
    FourierSpectralBottleneck,
    FourierUNet,
    FourierUNetAtEncoder1,
)

AmplitudePhaseFourierBottleneck = FourierSpectralBottleneck
APFUNet = FourierUNet
APFUNetAtEncoder1 = FourierUNetAtEncoder1
APFAmplitudeOnlyUNet = FourierAmplitudeOnlyUNet
APFPhaseOnlyUNet = FourierPhaseOnlyUNet
PlainFourierUNet = FourierUNet

__all__ = [
    "AmplitudePhaseFourierBottleneck",
    "APFUNet",
    "APFUNetAtEncoder1",
    "APFAmplitudeOnlyUNet",
    "APFPhaseOnlyUNet",
    "PlainFourierUNet",
    "BoundedFourierUNet",
    "FourierNoChannelMixUNet",
    "FourierNoResidualUNet",
]
