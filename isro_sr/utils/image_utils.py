import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import gaussian_filter

def align_images(img1, img2, warp_mode=cv2.MOTION_EUCLIDEAN, max_iterations=5000, termination_eps=1e-7):
    """
    Align two images using ECC (Enhanced Correlation Coefficient) method
    
    Args:
        img1 (numpy.ndarray): First image (reference)
        img2 (numpy.ndarray): Second image to be aligned
        warp_mode: Type of transformation (MOTION_EUCLIDEAN, MOTION_AFFINE, etc.)
        max_iterations: Maximum number of iterations
        termination_eps: Termination criteria
        
    Returns:
        tuple: (aligned_img1, aligned_img2) - Both images aligned to a common reference
    """
    # Convert images to grayscale
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Find size of images
    height, width = img1_gray.shape
    
    # Define the transformation matrix
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, termination_eps)
    
    try:
        # Run the ECC algorithm
        _, warp_matrix = cv2.findTransformECC(
            img1_gray, img2_gray, warp_matrix, warp_mode, criteria, None, 1
        )
        
        # Warp the second image to match the first
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned_img2 = cv2.warpPerspective(
                img2, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        else:
            aligned_img2 = cv2.warpAffine(
                img2, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            
        return img1, aligned_img2
    
    except cv2.error as e:
        # Fallback to phase correlation if ECC fails
        print(f"ECC alignment failed, using phase correlation instead: {e}")
        return phase_correlation_align(img1, img2)

def phase_correlation_align(img1, img2):
    """
    Align images using phase correlation for sub-pixel alignment
    
    Args:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
    
    Returns:
        tuple: (aligned_img1, aligned_img2) - Both images aligned
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Get image dimensions
    height, width = img1_gray.shape
    
    # Compute the phase correlation to find translation
    shift, error, _ = cv2.phaseCorrelate(
        np.float32(img1_gray), np.float32(img2_gray)
    )
    
    # Create the transformation matrix for the translation
    M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
    
    # Apply the translation to align the second image
    aligned_img2 = cv2.warpAffine(img2, M, (width, height), flags=cv2.INTER_LINEAR)
    
    return img1, aligned_img2

def compute_niqe(img):
    """
    Compute NIQE (Natural Image Quality Evaluator) score
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        float: NIQE score (lower is better)
    """
    try:
        from piq import niqe
        import torch
        
        # Convert to RGB if BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        # Calculate NIQE
        score = niqe(img_tensor).item()
        return score
    except ImportError:
        # Fallback method if piq is not available
        return simple_niqe_estimate(img)

def compute_brisque(img):
    """
    Compute BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        float: BRISQUE score (lower is better)
    """
    try:
        from piq import brisque
        import torch
        
        # Convert to RGB if BGR
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        
        # Calculate BRISQUE
        score = brisque(img_tensor).item()
        return score
    except ImportError:
        # Fallback method if piq is not available
        return simple_brisque_estimate(img)

def simple_niqe_estimate(img):
    """
    A simple estimation of image quality when piq is not available
    This is not a true NIQE implementation but a fallback
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        float: Estimated quality score
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Compute local variance
    local_var = np.std(gray)
    
    # Compute gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Compute approximate score (lower is better)
    score = 10.0 - (local_var * np.mean(gradient_magnitude)) / 1000.0
    return max(0.0, min(score, 10.0))  # Clamp between 0 and 10

def simple_brisque_estimate(img):
    """
    A simple estimation of image quality when piq is not available
    This is not a true BRISQUE implementation but a fallback
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        float: Estimated quality score
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur
    blurred = gaussian_filter(gray, sigma=1.0)
    
    # Compute difference
    diff = gray.astype(np.float32) - blurred
    
    # Compute statistics
    mean_val = np.mean(np.abs(diff))
    std_val = np.std(diff)
    
    # Compute approximate score (lower is better)
    score = 50.0 * (mean_val / (std_val + 1e-6))
    return max(0.0, min(score, 100.0))  # Clamp between 0 and 100

def compute_quality_metrics(img):
    """
    Compute multiple quality metrics for an image
    
    Args:
        img (numpy.ndarray): Input image
        
    Returns:
        dict: Dictionary of quality metrics
    """
    metrics = {
        'NIQE': compute_niqe(img),
        'BRISQUE': compute_brisque(img)
    }
    
    return metrics 