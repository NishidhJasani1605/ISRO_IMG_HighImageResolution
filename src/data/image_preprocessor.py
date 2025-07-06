import cv2
import numpy as np
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

class SatelliteImagePreprocessor:
    def __init__(self, registration_method: str = 'ecc'):
        """
        Initialize the preprocessor with specified registration method.
        Args:
            registration_method: 'ecc' or 'phase_correlation'
        """
        self.registration_method = registration_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def register_images(self, reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register target image to reference image using specified method.
        """
        if self.registration_method == 'ecc':
            return self._ecc_registration(reference, target)
        else:
            return self._phase_correlation_registration(reference, target)

    def _ecc_registration(self, reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform ECC registration with sub-pixel accuracy.
        """
        # Convert images to grayscale if they're not already
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference
            target_gray = target

        # Define warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Define ECC termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-8)

        try:
            # Find transformation matrix
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, target_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
            
            # Apply transformation
            height, width = reference.shape[:2]
            registered = cv2.warpAffine(target, warp_matrix, (width, height),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            # Calculate registration confidence map
            confidence_map = self._calculate_registration_confidence(ref_gray, target_gray, warp_matrix)
            
            return registered, confidence_map
        except cv2.error:
            print("Warning: ECC registration failed, returning original images")
            return target, np.ones_like(ref_gray, dtype=np.float32)

    def _phase_correlation_registration(self, reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform phase correlation based registration.
        """
        # Convert to grayscale if needed
        if len(reference.shape) == 3:
            ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference
            target_gray = target

        # Calculate phase correlation
        shift, response = cv2.phaseCorrelate(np.float32(ref_gray), np.float32(target_gray))
        
        # Create transformation matrix
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        
        # Apply transformation
        height, width = reference.shape[:2]
        registered = cv2.warpAffine(target, M, (width, height))
        
        # Calculate confidence map based on local correlation
        confidence_map = self._calculate_registration_confidence(ref_gray, target_gray, M)
        
        return registered, confidence_map

    def _calculate_registration_confidence(self, ref_gray: np.ndarray, target_gray: np.ndarray, 
                                        warp_matrix: np.ndarray, window_size: int = 16) -> np.ndarray:
        """
        Calculate registration confidence map using local correlation.
        """
        height, width = ref_gray.shape
        confidence_map = np.zeros_like(ref_gray, dtype=np.float32)
        
        for y in range(0, height - window_size, window_size):
            for x in range(0, width - window_size, window_size):
                ref_patch = ref_gray[y:y+window_size, x:x+window_size]
                target_patch = target_gray[y:y+window_size, x:x+window_size]
                
                correlation = cv2.matchTemplate(ref_patch, target_patch, cv2.TM_CCOEFF_NORMED)
                confidence_map[y:y+window_size, x:x+window_size] = correlation[0, 0]
        
        # Smooth confidence map
        confidence_map = cv2.GaussianBlur(confidence_map, (5, 5), 0)
        return confidence_map

    def detect_clouds(self, image: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """
        Detect clouds using brightness and texture analysis.
        Returns a binary mask where 1 indicates clouds.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Brightness thresholding
        _, bright_mask = cv2.threshold(gray, int(255 * threshold), 255, cv2.THRESH_BINARY)
        
        # Texture analysis using local standard deviation
        kernel_size = 5
        local_std = cv2.blur(np.float32(gray), (kernel_size, kernel_size))
        texture_mask = local_std > np.mean(local_std) + np.std(local_std)
        
        # Combine masks
        cloud_mask = ((bright_mask > 0) & texture_mask).astype(np.uint8)
        
        # Clean up mask using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
        
        return cloud_mask

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using contrast limited adaptive histogram equalization (CLAHE).
        """
        # Convert to LAB color space if RGB
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels
            normalized = cv2.merge([cl, a, b])
            normalized = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE directly to grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(image)
            
        return normalized

    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced denoising using Non-local Means Denoising.
        """
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return denoised

    def prepare_image_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete pipeline for preparing an image pair for super-resolution.
        Returns: (registered_img, reference_img, quality_mask)
        """
        # Normalize and denoise both images
        img1_processed = self.denoise_image(self.normalize_image(img1))
        img2_processed = self.denoise_image(self.normalize_image(img2))
        
        # Register images
        registered_img, confidence_map = self.register_images(img1_processed, img2_processed)
        
        # Detect clouds in both images
        clouds1 = self.detect_clouds(img1_processed)
        clouds2 = self.detect_clouds(registered_img)
        
        # Combine cloud masks and confidence map into quality mask
        quality_mask = ((1 - clouds1) * (1 - clouds2) * confidence_map).astype(np.float32)
        
        return registered_img, img1_processed, quality_mask 