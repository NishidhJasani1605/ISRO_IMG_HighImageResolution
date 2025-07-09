"""
Utility functions for image processing and preprocessing
"""

import cv2
import numpy as np
from .image_preprocessor import SatelliteImagePreprocessor
from .enhanced_preprocessor import EnhancedSatellitePreprocessor
from .image_utils import align_images, compute_quality_metrics

# Create convenience functions
def normalize_image(image):
    """Convenience function to normalize an image"""
    preprocessor = SatelliteImagePreprocessor()
    return preprocessor.normalize_image(image)

def preprocess_image(img1, img2):
    """Convenience function to preprocess an image pair"""
    preprocessor = SatelliteImagePreprocessor()
    return preprocessor.prepare_image_pair(img1, img2)

def enhance_image(image):
    """Convenience function to enhance an image"""
    # Clone the image for the second parameter
    # EnhancedSatellitePreprocessor.enhance_image_pair expects two images
    image_copy = np.copy(image)
    preprocessor = EnhancedSatellitePreprocessor()
    # Return only the first enhanced image
    enhanced_img1, _, _ = preprocessor.enhance_image_pair(image, image_copy)
    return enhanced_img1

def equalize_histogram(image):
    """Convenience function to equalize histogram"""
    return cv2.equalizeHist(image) if len(image.shape) == 2 else image

__all__ = [
    "SatelliteImagePreprocessor", "EnhancedSatellitePreprocessor",
    "preprocess_image", "normalize_image", 
    "enhance_image", "equalize_histogram",
    "align_images", "compute_quality_metrics"
] 