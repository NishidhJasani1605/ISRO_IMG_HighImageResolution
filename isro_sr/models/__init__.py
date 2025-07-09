"""
Model architectures for super-resolution
"""

from .transformer_sr import SatelliteSR
from .dual_encoder_gan import DualEncoderGANSR

__all__ = ['SatelliteSR', 'DualEncoderGANSR'] 