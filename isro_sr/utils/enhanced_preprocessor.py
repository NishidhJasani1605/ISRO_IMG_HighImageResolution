import numpy as np
import cv2
from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage import measure, morphology
from skimage.restoration import denoise_tv_chambolle


class EnhancedSatellitePreprocessor:
    """Enhanced preprocessing for satellite data to maximize accuracy"""
    
    def __init__(self):
        self.registration_accuracy = 'subpixel'
        self.denoise_strength = 0.1
        
    def robust_registration(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform robust sub-pixel image registration
        """
        # Convert to float32 for processing
        img1_f = img1.astype(np.float32)
        img2_f = img2.astype(np.float32)
        
        # Normalize for better registration
        img1_norm = (img1_f - img1_f.mean()) / (img1_f.std() + 1e-8)
        img2_norm = (img2_f - img2_f.mean()) / (img2_f.std() + 1e-8)
        
        # Use phase correlation for initial alignment
        f1 = np.fft.fft2(img1_norm)
        f2 = np.fft.fft2(img2_norm)
        
        # Cross power spectrum
        cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-8)
        correlation = np.real(np.fft.ifft2(cross_power))
        
        # Find peak with sub-pixel accuracy
        y_peak, x_peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        
        # Sub-pixel refinement using quadratic interpolation
        if 1 <= y_peak < correlation.shape[0] - 1 and 1 <= x_peak < correlation.shape[1] - 1:
            # Quadratic fit around peak
            y_offset = self._quadratic_peak(
                correlation[y_peak-1, x_peak],
                correlation[y_peak, x_peak],
                correlation[y_peak+1, x_peak]
            )
            x_offset = self._quadratic_peak(
                correlation[y_peak, x_peak-1],
                correlation[y_peak, x_peak],
                correlation[y_peak, x_peak+1]
            )
            
            dy = y_peak + y_offset
            dx = x_peak + x_offset
        else:
            dy, dx = y_peak, x_peak
        
        # Convert to displacement
        if dy > correlation.shape[0] // 2:
            dy -= correlation.shape[0]
        if dx > correlation.shape[1] // 2:
            dx -= correlation.shape[1]
        
        # Apply sub-pixel shift using Fourier domain
        registered_img2 = self._fourier_shift(img2_f, -dy, -dx)
        
        return img1_f, registered_img2
    
    def _quadratic_peak(self, y1: float, y2: float, y3: float) -> float:
        """Find sub-pixel peak using quadratic interpolation"""
        denom = 2 * (y1 - 2*y2 + y3)
        if abs(denom) < 1e-8:
            return 0
        return (y1 - y3) / denom
    
    def _fourier_shift(self, img: np.ndarray, dy: float, dx: float) -> np.ndarray:
        """Apply sub-pixel shift using Fourier domain"""
        rows, cols = img.shape
        
        # Create coordinate grids
        y = np.arange(rows).reshape(-1, 1)
        x = np.arange(cols).reshape(1, -1)
        
        # Shift in frequency domain
        fy = np.fft.fftfreq(rows).reshape(-1, 1)
        fx = np.fft.fftfreq(cols).reshape(1, -1)
        
        # Phase shift
        phase_shift = np.exp(-2j * np.pi * (fy * dy + fx * dx))
        
        # Apply shift
        img_fft = np.fft.fft2(img)
        shifted_fft = img_fft * phase_shift
        shifted_img = np.real(np.fft.ifft2(shifted_fft))
        
        return shifted_img
    
    def advanced_quality_assessment(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Generate advanced quality mask considering multiple factors
        """
        h, w = img1.shape
        quality_mask = np.ones((h, w), dtype=np.float32)
        
        # 1. Cloud detection using spectral and textural features
        cloud_mask = self._detect_clouds(img1, img2)
        quality_mask *= (1 - cloud_mask)
        
        # 2. Motion blur detection
        blur_mask = self._detect_motion_blur(img1, img2)
        quality_mask *= (1 - blur_mask)
        
        # 3. Noise assessment
        noise_mask = self._assess_noise_level(img1, img2)
        quality_mask *= (1 - noise_mask)
        
        # 4. Edge consistency
        edge_consistency = self._compute_edge_consistency(img1, img2)
        quality_mask *= edge_consistency
        
        # 5. Local contrast assessment
        contrast_quality = self._assess_local_contrast(img1, img2)
        quality_mask *= contrast_quality
        
        # Smooth the quality mask
        quality_mask = ndimage.gaussian_filter(quality_mask, sigma=2.0)
        
        # Ensure minimum quality threshold
        quality_mask = np.clip(quality_mask, 0.1, 1.0)
        
        return quality_mask
    
    def _detect_clouds(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Detect clouds based on high reflectance and temporal consistency"""
        # High reflectance threshold
        high_reflectance = (img1 > np.percentile(img1, 90)) | (img2 > np.percentile(img2, 90))
        
        # Temporal inconsistency (potential clouds)
        temporal_diff = np.abs(img1 - img2)
        temporal_threshold = np.percentile(temporal_diff, 75)
        temporal_inconsistent = temporal_diff > temporal_threshold
        
        # Combine indicators
        cloud_mask = (high_reflectance & temporal_inconsistent).astype(np.float32)
        
        # Morphological operations to refine
        cloud_mask = morphology.binary_opening(cloud_mask, morphology.disk(3))
        cloud_mask = morphology.binary_closing(cloud_mask, morphology.disk(5))
        
        return cloud_mask.astype(np.float32)
    
    def _detect_motion_blur(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Detect motion blur using gradient analysis"""
        # Compute gradients
        grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        grad1_mag = np.sqrt(grad1_x**2 + grad1_y**2)
        grad2_mag = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Low gradient indicates potential blur
        low_grad_threshold = np.percentile(np.maximum(grad1_mag, grad2_mag), 25)
        blur_mask = (grad1_mag < low_grad_threshold) | (grad2_mag < low_grad_threshold)
        
        return blur_mask.astype(np.float32)
    
    def _assess_noise_level(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Assess local noise level"""
        # Estimate noise using Laplacian variance
        laplacian1 = cv2.Laplacian(img1, cv2.CV_64F)
        laplacian2 = cv2.Laplacian(img2, cv2.CV_64F)
        
        # Local variance of Laplacian (indicates noise)
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        local_var1 = cv2.filter2D(laplacian1**2, -1, kernel) - cv2.filter2D(laplacian1, -1, kernel)**2
        local_var2 = cv2.filter2D(laplacian2**2, -1, kernel) - cv2.filter2D(laplacian2, -1, kernel)**2
        
        # High variance indicates noise
        noise_threshold = np.percentile(np.maximum(local_var1, local_var2), 75)
        noise_mask = (local_var1 > noise_threshold) | (local_var2 > noise_threshold)
        
        return noise_mask.astype(np.float32)
    
    def _compute_edge_consistency(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Compute edge consistency between images"""
        # Detect edges
        edges1 = cv2.Canny((img1 * 255).astype(np.uint8), 50, 150)
        edges2 = cv2.Canny((img2 * 255).astype(np.uint8), 50, 150)
        
        # Edge consistency
        edge_agreement = (edges1 == edges2).astype(np.float32)
        
        # Smooth the consistency map
        consistency = cv2.GaussianBlur(edge_agreement, (5, 5), 1.0)
        
        return consistency
    
    def _assess_local_contrast(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Assess local contrast quality"""
        kernel_size = 9
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        
        # Local mean and standard deviation
        mean1 = cv2.filter2D(img1, -1, kernel)
        mean2 = cv2.filter2D(img2, -1, kernel)
        
        std1 = np.sqrt(cv2.filter2D(img1**2, -1, kernel) - mean1**2)
        std2 = np.sqrt(cv2.filter2D(img2**2, -1, kernel) - mean2**2)
        
        # Good contrast areas have higher standard deviation
        min_contrast = 0.01
        contrast_quality = np.minimum(std1, std2) / (np.maximum(std1, std2) + 1e-8)
        contrast_quality = np.clip(contrast_quality, min_contrast, 1.0)
        
        return contrast_quality
    
    def enhance_image_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Main preprocessing pipeline for maximum accuracy
        """
        # 1. Robust registration
        reg_img1, reg_img2 = self.robust_registration(img1, img2)
        
        # 2. Denoising
        denoised_img1 = denoise_tv_chambolle(reg_img1, weight=self.denoise_strength)
        denoised_img2 = denoise_tv_chambolle(reg_img2, weight=self.denoise_strength)
        
        # 3. Advanced quality assessment
        quality_mask = self.advanced_quality_assessment(denoised_img1, denoised_img2)
        
        # 4. Histogram matching for consistency
        matched_img2 = self._histogram_matching(denoised_img2, denoised_img1)
        
        return denoised_img1, matched_img2, quality_mask
    
    def _histogram_matching(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match histogram of source to reference"""
        # Flatten images
        src_flat = source.flatten()
        ref_flat = reference.flatten()
        
        # Compute histograms
        src_hist, src_bins = np.histogram(src_flat, bins=256, density=True)
        ref_hist, ref_bins = np.histogram(ref_flat, bins=256, density=True)
        
        # Compute CDFs
        src_cdf = np.cumsum(src_hist)
        ref_cdf = np.cumsum(ref_hist)
        
        # Normalize CDFs
        src_cdf = src_cdf / src_cdf[-1]
        ref_cdf = ref_cdf / ref_cdf[-1]
        
        # Create mapping
        src_values = src_bins[:-1]
        ref_values = ref_bins[:-1]
        
        # Interpolate to create mapping function
        mapping = np.interp(src_cdf, ref_cdf, ref_values)
        
        # Apply mapping
        matched = np.interp(source.flatten(), src_values, mapping)
        
        return matched.reshape(source.shape)
