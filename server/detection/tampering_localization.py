"""
Document Tampering Localization Module
======================================
Hybrid approach combining Deep Learning (Grad-CAM) with Classical Image Forensics
for detecting and localizing manipulated regions in document images.

Author: DocuForge Team
Date: October 2025
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import filters, feature, transform
from skimage.util import img_as_float
from scipy import ndimage
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. Using custom Grad-CAM implementation.")

# Import enhanced blur detection (specialized for Forgery.py artifacts)
try:
    from detection.enhanced_blur_detection import (
        detect_blur_regions,
        detect_motion_blur,
        detect_text_overlay_artifacts,
        detect_splice_boundaries,
        detect_illumination_inconsistency,
        comprehensive_forgery_detection
    )
    ENHANCED_BLUR_AVAILABLE = True
except ImportError:
    try:
        from enhanced_blur_detection import (
            detect_blur_regions,
            detect_motion_blur,
            detect_text_overlay_artifacts,
            detect_splice_boundaries,
            detect_illumination_inconsistency,
            comprehensive_forgery_detection
        )
        ENHANCED_BLUR_AVAILABLE = True
    except ImportError:
        ENHANCED_BLUR_AVAILABLE = False
        print("Note: Enhanced blur detection module not available. Using standard methods.")


class DocumentTamperingDetector:
    """
    Hybrid tampering detection system combining CNN-based and classical forensics.
    """
    
    def __init__(self, model, device="cuda", output_dir="tampering_results"):
        """
        Initialize the tampering detector.
        
        Args:
            model: Pre-trained PyTorch model for forgery detection
            device: Device to run inference on ('cuda' or 'cpu')
            output_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.model.eval()
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def preprocess_image(self, image_path):
        """Load and preprocess image for model input."""
        # Load image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Keep original for visualization
        original_img = img_rgb.copy()
        
        # Resize for model (assuming ImageNet-style input)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_normalized - mean) / std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return img_bgr, img_rgb, original_img, img_tensor
    
    def compute_gradcam(self, img_tensor, target_layer=None):
        """
        Compute Grad-CAM heatmap for forgery localization.
        
        Args:
            img_tensor: Preprocessed image tensor
            target_layer: Target layer for Grad-CAM (auto-detected if None)
        
        Returns:
            Grad-CAM heatmap normalized to [0, 1]
        """
        img_tensor = img_tensor.to(self.device)
        
        if GRADCAM_AVAILABLE and target_layer is not None:
            # Use pytorch-grad-cam library
            try:
                cam = GradCAMPlusPlus(model=self.model, target_layers=[target_layer])
                grayscale_cam = cam(input_tensor=img_tensor, targets=None)
                return grayscale_cam[0]
            except Exception as e:
                print(f"pytorch-grad-cam failed: {e}. Using custom implementation.")
        
        # Custom Grad-CAM implementation
        return self._custom_gradcam(img_tensor)
    
    def _custom_gradcam(self, img_tensor):
        """Custom Grad-CAM implementation."""
        # Forward pass
        self.model.zero_grad()
        
        # Hook to capture gradients and activations
        activations = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            activations['value'] = output
        
        def backward_hook(module, grad_in, grad_out):
            gradients['value'] = grad_out[0]
        
        # Register hooks on the last convolutional layer
        target_layer = self._get_target_layer()
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        
        try:
            # Forward pass
            output = self.model(img_tensor)
            
            # Get prediction
            if output.dim() == 1:
                score = output
            else:
                score = output[:, 1] if output.shape[1] > 1 else output[:, 0]
            
            # Backward pass
            self.model.zero_grad()
            score.backward()
            
            # Get activations and gradients
            act = activations['value'].detach()
            grad = gradients['value'].detach()
            
            # Compute weights
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            
            # Compute weighted combination
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze().cpu().numpy()
            cam = cv2.resize(cam, (224, 224))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            return cam
        
        finally:
            forward_handle.remove()
            backward_handle.remove()
    
    def _get_target_layer(self):
        """Auto-detect the last convolutional layer."""
        # For ResNet architectures
        if hasattr(self.model, 'layer4'):
            return self.model.layer4[-1]
        elif hasattr(self.model, 'features'):
            # For VGG or similar
            for module in reversed(list(self.model.features.children())):
                if isinstance(module, torch.nn.Conv2d):
                    return module
        else:
            # Fallback: find last Conv2d layer
            conv_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Conv2d)]
            if conv_layers:
                return conv_layers[-1]
        
        raise ValueError("Could not auto-detect target layer for Grad-CAM")
    
    def error_level_analysis(self, img_bgr, quality=90):
        """
        Error Level Analysis (ELA) - detects compression inconsistencies.
        
        Args:
            img_bgr: Input image in BGR format
            quality: JPEG quality for re-compression
        
        Returns:
            ELA heatmap normalized to [0, 1]
        """
        # Save and reload with JPEG compression
        temp_path = os.path.join(self.output_dir, "temp_ela.jpg")
        cv2.imwrite(temp_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imread(temp_path)
        
        # Compute difference
        ela = cv2.absdiff(img_bgr, compressed).astype(np.float32)
        ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
        
        # Normalize
        ela_gray = (ela_gray - ela_gray.min()) / (ela_gray.max() - ela_gray.min() + 1e-8)
        
        # Enhance contrast
        ela_enhanced = np.power(ela_gray, 0.7)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return ela_enhanced
    
    def noise_inconsistency_map(self, img_rgb):
        """
        Detect noise inconsistencies using local variance analysis.
        
        Args:
            img_rgb: Input image in RGB format
        
        Returns:
            Noise inconsistency heatmap normalized to [0, 1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Compute local standard deviation
        kernel_size = 7
        mean = cv2.blur(gray, (kernel_size, kernel_size))
        mean_sq = cv2.blur(gray ** 2, (kernel_size, kernel_size))
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)  # Ensure non-negative
        std_dev = np.sqrt(variance)
        
        # Apply median filter to get expected noise level
        # Note: medianBlur requires uint8 or uint16, so we convert and scale
        kernel_size_large = 15
        std_dev_scaled = (std_dev * 255 / (std_dev.max() + 1e-8)).astype(np.uint8)
        median_std_scaled = cv2.medianBlur(std_dev_scaled, kernel_size_large)
        median_std = median_std_scaled.astype(np.float32) * (std_dev.max() + 1e-8) / 255.0
        
        # Compute inconsistency
        inconsistency = np.abs(std_dev - median_std)
        
        # Normalize
        inconsistency = (inconsistency - inconsistency.min()) / (inconsistency.max() - inconsistency.min() + 1e-8)
        
        return inconsistency
    
    def edge_artifact_map(self, img_rgb):
        """
        Detect edge artifacts using high-frequency Laplacian filter.
        
        Args:
            img_rgb: Input image in RGB format
        
        Returns:
            Edge artifact heatmap normalized to [0, 1]
        """
        # Convert to grayscale (keep as uint8 for Laplacian)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply Laplacian filter (input must be uint8)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian = np.abs(laplacian).astype(np.float32)
        
        # Apply Gaussian blur to reduce noise
        laplacian_smooth = cv2.GaussianBlur(laplacian, (5, 5), 0)
        
        # Compute local statistics
        kernel_size = 9
        local_mean = cv2.blur(laplacian_smooth, (kernel_size, kernel_size))
        local_std = cv2.blur(laplacian_smooth ** 2, (kernel_size, kernel_size)) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_std, 0))
        
        # High standard deviation indicates inconsistent edges
        artifact_map = local_std
        
        # Normalize
        artifact_map = (artifact_map - artifact_map.min()) / (artifact_map.max() - artifact_map.min() + 1e-8)
        
        return artifact_map
    
    def jpeg_block_artifact_analysis(self, img_rgb):
        """
        Detect JPEG block artifacts (8x8 DCT grid inconsistencies).
        
        Args:
            img_rgb: Input image in RGB format
        
        Returns:
            Block artifact heatmap normalized to [0, 1]
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        h, w = gray.shape
        block_size = 8
        
        # Pad image to be divisible by block_size
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Compute block-wise DCT coefficients
        h_blocks = padded.shape[0] // block_size
        w_blocks = padded.shape[1] // block_size
        
        artifact_map = np.zeros_like(padded)
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                y, x = i * block_size, j * block_size
                block = padded[y:y+block_size, x:x+block_size]
                
                # Compute DCT
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                
                # High-frequency coefficients indicate artifacts
                high_freq = np.abs(dct_block[4:, 4:]).sum()
                artifact_map[y:y+block_size, x:x+block_size] = high_freq
        
        # Remove padding
        artifact_map = artifact_map[:h, :w]
        
        # Apply Gaussian smoothing
        artifact_map = cv2.GaussianBlur(artifact_map, (9, 9), 0)
        
        # Normalize
        artifact_map = (artifact_map - artifact_map.min()) / (artifact_map.max() - artifact_map.min() + 1e-8)
        
        return artifact_map
    
    def copy_move_detection(self, img_rgb, threshold=0.8):
        """
        Optional: Detect copy-move forgery using ORB keypoint matching.
        
        Args:
            img_rgb: Input image in RGB format
            threshold: Matching threshold
        
        Returns:
            Copy-move heatmap normalized to [0, 1]
        """
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=2000)
        
        # Detect keypoints and descriptors
        kp, des = orb.detectAndCompute(gray, None)
        
        if des is None or len(kp) < 10:
            return np.zeros(gray.shape, dtype=np.float32)
        
        # Match descriptors with themselves
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=2)
        
        # Filter good matches (excluding self-matches)
        suspicious_points = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair
                # Exclude self-matches and apply ratio test
                if m.queryIdx != m.trainIdx and m.distance < threshold * n.distance:
                    pt1 = kp[m.queryIdx].pt
                    pt2 = kp[m.trainIdx].pt
                    # Check if points are sufficiently far apart
                    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
                    if dist > 20:  # Minimum distance threshold
                        suspicious_points.append((pt1, pt2))
        
        # Create heatmap
        heatmap = np.zeros(gray.shape, dtype=np.float32)
        for pt1, pt2 in suspicious_points:
            cv2.circle(heatmap, (int(pt1[0]), int(pt1[1])), 15, 1.0, -1)
            cv2.circle(heatmap, (int(pt2[0]), int(pt2[1])), 15, 1.0, -1)
        
        # Smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def enhanced_blur_detection(self, img_rgb, enable_all=True):
        """
        Enhanced blur/smudge detection specifically tuned for Forgery.py artifacts.
        Detects:
        - Local Gaussian blur regions
        - Motion blur (horizontal/vertical)
        - Text overlay artifacts
        - Copy-paste splice boundaries
        - Illumination inconsistencies
        
        Args:
            img_rgb: Input image in RGB format
            enable_all: If True, run all detection methods; if False, only blur
        
        Returns:
            Dictionary with detection maps or single combined map
        """
        if ENHANCED_BLUR_AVAILABLE and enable_all:
            # Use comprehensive detection from enhanced module
            results = comprehensive_forgery_detection(img_rgb)
            return results['combined'], results
        elif ENHANCED_BLUR_AVAILABLE:
            # Just blur detection
            blur_map = detect_blur_regions(img_rgb, window_size=31, threshold_factor=1.5)
            motion_map = detect_motion_blur(img_rgb, num_angles=16)
            combined = 0.6 * blur_map + 0.4 * motion_map
            return combined, {'blur': blur_map, 'motion_blur': motion_map, 'combined': combined}
        else:
            # Fallback: basic blur detection using Laplacian variance
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # Compute Laplacian variance (blur measure - input must be uint8)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3).astype(np.float32)
            laplacian_var = cv2.blur(laplacian ** 2, (15, 15)) - cv2.blur(laplacian, (15, 15)) ** 2
            laplacian_var = np.maximum(laplacian_var, 0)
            
            # Invert (low variance = blur)
            blur_map = 1.0 / (1.0 + laplacian_var / (laplacian_var.mean() + 1e-8))
            
            # Normalize
            blur_map = (blur_map - blur_map.min()) / (blur_map.max() - blur_map.min() + 1e-8)
            blur_map = cv2.GaussianBlur(blur_map, (21, 21), 0)
            
            return blur_map, {'blur': blur_map, 'combined': blur_map}
    
    def combine_maps_fusion(self, gradcam_map, classical_maps, weights=None):
        """
        Fuse Grad-CAM and classical forensic maps into a single suspicion score.
        
        Args:
            gradcam_map: Grad-CAM heatmap [0, 1]
            classical_maps: List of classical forensic heatmaps
            weights: Fusion weights (default: 0.4 Grad-CAM, 0.6 classical)
        
        Returns:
            Fused heatmap normalized to [0, 1]
        """
        if weights is None:
            weight_gradcam = 0.4
            weight_classical = 0.6
        else:
            weight_gradcam = weights.get('gradcam', 0.4)
            weight_classical = weights.get('classical', 0.6)
        
        # Normalize Grad-CAM to image size
        h, w = classical_maps[0].shape
        gradcam_resized = cv2.resize(gradcam_map, (w, h))
        
        # Combine classical maps (average)
        classical_combined = np.mean(np.stack(classical_maps, axis=0), axis=0)
        
        # Weighted fusion
        fused = weight_gradcam * gradcam_resized + weight_classical * classical_combined
        
        # Normalize to [0, 1]
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        
        return fused
    
    def apply_heatmap_overlay(self, original_img, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Apply heatmap overlay on original image.
        
        Args:
            original_img: Original image (RGB)
            heatmap: Heatmap [0, 1]
            alpha: Transparency factor
            colormap: OpenCV colormap
        
        Returns:
            Overlaid image
        """
        # Resize heatmap to match image
        h, w = original_img.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert to uint8 and apply colormap
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original
        overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def extract_bounding_boxes_from_mask(self, mask, min_area=100):
        """
        Extract bounding boxes from binary mask.
        
        Args:
            mask: Binary mask (0 or 255)
            min_area: Minimum contour area to consider
        
        Returns:
            List of bounding boxes [(x, y, w, h), ...]
        """
        # Ensure mask is uint8
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                bboxes.append((x, y, w, h))
        
        return bboxes
    
    def visualize_results(self, original_img, heatmap, mask, bboxes, filename="tampering_visualization.png"):
        """
        Create comprehensive visualization of tampering detection results.
        
        Args:
            original_img: Original image (RGB)
            heatmap: Fused heatmap
            mask: Binary mask
            bboxes: List of bounding boxes
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Heatmap overlay
        overlay = self.apply_heatmap_overlay(original_img, heatmap, alpha=0.6)
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title("Tampering Heatmap Overlay")
        axes[0, 1].axis('off')
        
        # Binary mask
        axes[1, 0].imshow(mask, cmap='gray')
        axes[1, 0].set_title("Binary Tampering Mask")
        axes[1, 0].axis('off')
        
        # Bounding boxes
        img_with_boxes = original_img.copy()
        for (x, y, w, h) in bboxes:
            cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (255, 0, 0), 2)
        axes[1, 1].imshow(img_with_boxes)
        axes[1, 1].set_title(f"Detected Regions ({len(bboxes)} areas)")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {output_path}")


def detect_tampering_hybrid(image_path, model, device="cuda", save_results=True, 
                           sensitivity=0.5, return_intermediate_maps=False):
    """
    Detect tampered regions using hybrid Deep Learning + Classical Forensics.
    
    This function combines:
    - Grad-CAM from a pre-trained CNN forgery detector
    - Classical forensic techniques (ELA, noise analysis, edge artifacts, JPEG blocks)
    - Weighted fusion to produce final tampering localization
    
    Args:
        image_path (str): Path to document image
        model (torch.nn.Module): Pre-trained forgery detection model
        device (str): Device for inference ('cuda' or 'cpu')
        save_results (bool): Whether to save visualization results
        sensitivity (float): Threshold for binary mask (0-1, default 0.5)
        return_intermediate_maps (bool): Return individual forensic maps
    
    Returns:
        dict: Dictionary containing:
            - 'heatmap': Final tampering heatmap (RGB image)
            - 'mask': Binary tampering mask (0 or 1)
            - 'bboxes': List of bounding boxes [(x, y, w, h), ...]
            - 'probability': Overall tampering confidence score (0-1)
            - 'fused_map': Raw fused heatmap [0, 1]
            - 'intermediate_maps': (optional) Individual forensic maps
    
    Example:
        >>> result = detect_tampering_hybrid(
        ...     "sample_passport.jpg",
        ...     model=resnet_model,
        ...     device="cuda"
        ... )
        >>> cv2.imshow("Heatmap", result["heatmap"])
        >>> print(f"Tampering probability: {result['probability']:.2%}")
    """
    
    # Initialize detector
    detector = DocumentTamperingDetector(model, device=device)
    
    print(f"🔍 Analyzing document: {image_path}")
    
    # Step 1: Load and preprocess image
    try:
        img_bgr, img_rgb, original_img, img_tensor = detector.preprocess_image(image_path)
        print("✓ Image loaded and preprocessed")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")
    
    # Step 2: Deep Learning Branch - Grad-CAM
    print("🧠 Computing Grad-CAM heatmap...")
    try:
        gradcam_map = detector.compute_gradcam(img_tensor)
        print("✓ Grad-CAM completed")
    except Exception as e:
        print(f"⚠ Grad-CAM failed: {e}. Using fallback.")
        gradcam_map = np.zeros((224, 224), dtype=np.float32)
    
    # Step 3: Classical Forensics Branch
    print("🔬 Running classical forensic analysis...")
    
    classical_maps = []
    intermediate_results = {}
    
    # Error Level Analysis
    try:
        ela_map = detector.error_level_analysis(img_bgr)
        classical_maps.append(ela_map)
        intermediate_results['ela'] = ela_map
        print("  ✓ ELA completed")
    except Exception as e:
        print(f"  ⚠ ELA failed: {e}")
    
    # Noise Inconsistency
    try:
        noise_map = detector.noise_inconsistency_map(img_rgb)
        classical_maps.append(noise_map)
        intermediate_results['noise'] = noise_map
        print("  ✓ Noise analysis completed")
    except Exception as e:
        print(f"  ⚠ Noise analysis failed: {e}")
    
    # Edge Artifacts
    try:
        edge_map = detector.edge_artifact_map(img_rgb)
        classical_maps.append(edge_map)
        intermediate_results['edge'] = edge_map
        print("  ✓ Edge artifact detection completed")
    except Exception as e:
        print(f"  ⚠ Edge artifact detection failed: {e}")
    
    # JPEG Block Artifacts
    try:
        jpeg_map = detector.jpeg_block_artifact_analysis(img_rgb)
        classical_maps.append(jpeg_map)
        intermediate_results['jpeg'] = jpeg_map
        print("  ✓ JPEG block analysis completed")
    except Exception as e:
        print(f"  ⚠ JPEG block analysis failed: {e}")
    
    # Enhanced Blur Detection (SPECIALIZED FOR FORGERY.PY ARTIFACTS)
    try:
        print("  🎯 Enhanced blur/smudge detection (Forgery.py artifacts)...")
        blur_combined, blur_results = detector.enhanced_blur_detection(img_rgb, enable_all=True)
        classical_maps.append(blur_combined)
        
        # Add individual blur detection maps to intermediate results
        if return_intermediate_maps:
            for key, value in blur_results.items():
                intermediate_results[f'blur_{key}'] = value
        
        if ENHANCED_BLUR_AVAILABLE:
            print("  ✓ Enhanced blur detection completed (5 techniques)")
        else:
            print("  ✓ Blur detection completed (fallback method)")
    except Exception as e:
        print(f"  ⚠ Blur detection failed: {e}")
    
    # Optional: Copy-Move Detection
    try:
        copymove_map = detector.copy_move_detection(img_rgb)
        if copymove_map.max() > 0:
            classical_maps.append(copymove_map)
            intermediate_results['copymove'] = copymove_map
            print("  ✓ Copy-move detection completed")
    except Exception as e:
        print(f"  ⚠ Copy-move detection failed: {e}")
    
    if not classical_maps:
        raise RuntimeError("All classical forensic methods failed")
    
    # Step 4: Fusion Layer
    print("🔗 Fusing detection maps...")
    fused_map = detector.combine_maps_fusion(gradcam_map, classical_maps)
    print("✓ Fusion completed")
    
    # Step 5: Generate binary mask and extract regions
    print("📊 Generating tampering mask...")
    binary_mask = (fused_map > sensitivity).astype(np.uint8)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    
    # Extract bounding boxes
    bboxes = detector.extract_bounding_boxes_from_mask(binary_mask, min_area=100)
    print(f"✓ Found {len(bboxes)} suspicious region(s)")
    
    # Step 6: Calculate tampering probability
    tampering_probability = float(np.mean(fused_map))
    print(f"📈 Overall tampering probability: {tampering_probability:.2%}")
    
    # Step 7: Create visualizations
    print("🎨 Creating visualizations...")
    
    # Heatmap overlay
    heatmap_overlay = detector.apply_heatmap_overlay(
        original_img, fused_map, alpha=0.5, colormap=cv2.COLORMAP_JET
    )
    
    # Save results if requested
    if save_results:
        # Extract filename
        base_name = Path(image_path).stem
        
        # Comprehensive visualization
        detector.visualize_results(
            original_img, fused_map, binary_mask, bboxes,
            filename=f"{base_name}_tampering_analysis.png"
        )
        
        # Save individual outputs
        cv2.imwrite(
            os.path.join(detector.output_dir, f"{base_name}_heatmap.png"),
            cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(
            os.path.join(detector.output_dir, f"{base_name}_mask.png"),
            binary_mask * 255
        )
        
        # Save intermediate maps if requested
        if return_intermediate_maps:
            for name, map_data in intermediate_results.items():
                map_uint8 = (map_data * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(detector.output_dir, f"{base_name}_{name}.png"),
                    map_uint8
                )
        
        print(f"✓ Results saved to: {detector.output_dir}")
    
    # Prepare output dictionary
    result = {
        'heatmap': heatmap_overlay,
        'mask': binary_mask,
        'bboxes': bboxes,
        'probability': tampering_probability,
        'fused_map': fused_map,
    }
    
    if return_intermediate_maps:
        result['intermediate_maps'] = intermediate_results
        result['gradcam'] = gradcam_map
    
    print("✅ Tampering detection completed!\n")
    
    return result


# Utility functions for standalone use

def visualize_detection_result(result, display=True):
    """
    Quick visualization of detection results.
    
    Args:
        result: Output dictionary from detect_tampering_hybrid()
        display: Whether to display images (requires GUI)
    """
    if display:
        cv2.imshow("Tampering Heatmap", cv2.cvtColor(result['heatmap'], cv2.COLOR_RGB2BGR))
        cv2.imshow("Tampering Mask", result['mask'] * 255)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n📊 Detection Summary:")
    print(f"  - Tampering Probability: {result['probability']:.2%}")
    print(f"  - Suspicious Regions: {len(result['bboxes'])}")
    if result['bboxes']:
        print(f"  - Bounding Boxes:")
        for i, (x, y, w, h) in enumerate(result['bboxes'], 1):
            print(f"    {i}. x={x}, y={y}, width={w}, height={h}")


def batch_detect_tampering(image_paths, model, device="cuda", output_csv="tampering_results.csv"):
    """
    Process multiple images and save results to CSV.
    
    Args:
        image_paths: List of image paths
        model: Pre-trained model
        device: Device for inference
        output_csv: Output CSV filename
    
    Returns:
        DataFrame with results
    """
    import pandas as pd
    
    results_list = []
    
    for img_path in image_paths:
        try:
            result = detect_tampering_hybrid(
                img_path, model, device=device, save_results=True
            )
            
            results_list.append({
                'image': img_path,
                'tampering_probability': result['probability'],
                'num_regions': len(result['bboxes']),
                'status': 'suspicious' if result['probability'] > 0.5 else 'authentic'
            })
        except Exception as e:
            print(f"❌ Failed to process {img_path}: {e}")
            results_list.append({
                'image': img_path,
                'tampering_probability': None,
                'num_regions': 0,
                'status': 'error'
            })
    
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Batch results saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    """
    Example usage and testing
    """
    print("=" * 70)
    print("Document Tampering Localization - Hybrid Detection System")
    print("=" * 70)
    print("\nThis module requires a pre-trained PyTorch model.")
    print("Example usage:\n")
    print("from tampering_localization import detect_tampering_hybrid")
    print("from your_model import load_model\n")
    print("model = load_model('best_model.pth')")
    print("result = detect_tampering_hybrid('document.jpg', model)")
    print("print(result['probability'])")
    print("\n" + "=" * 70)
