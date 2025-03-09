import cv2
import numpy as np
from PIL import Image
import io
from scipy.fftpack import dct
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)

class PhotoshopDetector:
    """Advanced detector for image manipulations using multiple techniques."""
    
    def __init__(self):
        self.methods = {
            "ela": self._error_level_analysis,
            "noise": self._noise_analysis,
            "dct": self._dct_analysis,
            "clone": self._clone_detection
        }
    
    async def detect(self, img):
        """
        Detect image manipulations using multiple advanced techniques.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Detection results
        """
        try:
            # Convert PIL Image to OpenCV format
            img_cv = self._pil_to_cv2(img)
            
            # Run all detection methods
            ela_result = await self._error_level_analysis(img)
            noise_result = await self._noise_analysis(img_cv)
            dct_result = await self._dct_analysis(img_cv)
            clone_result = await self._clone_detection(img_cv)
            jpeg_ghost_result = await self._jpeg_ghost_detection(img)
            
            # Combine results 
            all_regions = []
            all_regions.extend(ela_result.get("regions", []))
            all_regions.extend(noise_result.get("regions", []))
            all_regions.extend(clone_result.get("regions", []))
            
            # Calculate weighted score based on all methods
            weights = {
                "ela": 0.3,
                "noise": 0.2,
                "dct": 0.2,
                "clone": 0.2,
                "jpeg_ghost": 0.1
            }
            
            scores = {
                "ela": ela_result.get("score", 0),
                "noise": noise_result.get("score", 0),
                "dct": dct_result.get("score", 0),
                "clone": clone_result.get("score", 0),
                "jpeg_ghost": jpeg_ghost_result.get("score", 0)
            }
            
            weighted_score = sum(scores[method] * weights[method] for method in weights)
            
            # Calculate manipulation probability
            manipulation_probability = min(100, weighted_score)
            
            # Invert score for consistency (100 = authentic, 0 = manipulated)
            authenticity_score = 100 - manipulation_probability
            
            return {
                "score": authenticity_score,
                "manipulation_probability": manipulation_probability,
                "manipulated_regions": all_regions,
                "detection_results": {
                    "ela": ela_result,
                    "noise": noise_result,
                    "dct": dct_result,
                    "clone": clone_result,
                    "jpeg_ghost": jpeg_ghost_result
                },
                "techniques_used": list(self.methods.keys()) + ["jpeg_ghost"]
            }
            
        except Exception as e:
            logger.error(f"Manipulation detection error: {str(e)}")
            return {
                "score": 50,  # Neutral score on error
                "error": str(e),
                "manipulation_probability": 0,
                "manipulated_regions": []
            }
    
    def _pil_to_cv2(self, pil_img):
        """Convert PIL Image to OpenCV format."""
        # Convert to RGB if it's in RGBA
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        
        # Convert to numpy array
        img_np = np.array(pil_img)
        
        # Convert RGB to BGR (OpenCV format)
        return cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    async def _error_level_analysis(self, img):
        """
        Perform Error Level Analysis (ELA) to detect manipulated regions.
        
        ELA works by saving the image at a known quality level (e.g., 90%), 
        then comparing it to the original. Areas with higher error levels 
        often indicate manipulation.
        """
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Quality level for ELA
            quality = 90
            
            # Save to a temporary JPEG with known quality
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=quality)
            temp_io.seek(0)
            
            # Load the saved image
            saved_img = Image.open(temp_io)
            
            # Initialize error image
            ela_img = Image.new('RGB', img.size, (0, 0, 0))
            
            # Compare original with resaved
            for x in range(img.width):
                for y in range(img.height):
                    orig_pixel = img.getpixel((x, y))
                    saved_pixel = saved_img.getpixel((x, y))
                    
                    # Calculate difference for each channel
                    diff_r = abs(orig_pixel[0] - saved_pixel[0]) * 10
                    diff_g = abs(orig_pixel[1] - saved_pixel[1]) * 10
                    diff_b = abs(orig_pixel[2] - saved_pixel[2]) * 10
                    
                    # Scale for visibility
                    ela_pixel = (min(diff_r, 255), min(diff_g, 255), min(diff_b, 255))
                    ela_img.putpixel((x, y), ela_pixel)
            
            # Convert to numpy array for analysis
            ela_np = np.array(ela_img)
            
            # Threshold to find suspicious regions
            gray_ela = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray_ela, 50, 255, cv2.THRESH_BINARY)
            
            # Find contours of suspicious regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours
            min_area = img.width * img.height * 0.001  # 0.1% of image area
            suspicious_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    suspicious_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(area),
                        "detection_method": "ELA"
                    })
            
            # Calculate manipulation score based on suspicious regions
            if suspicious_regions:
                total_suspicious_area = sum(region["area"] for region in suspicious_regions)
                image_area = img.width * img.height
                area_percentage = (total_suspicious_area / image_area) * 100
                
                # Cap at 90% to avoid absolute certainty
                manipulation_score = min(90, area_percentage * 3)  # Scale for sensitivity
            else:
                manipulation_score = 0
            
            return {
                "score": manipulation_score,
                "regions": suspicious_regions,
                "analysis_method": "Error Level Analysis"
            }
            
        except Exception as e:
            logger.error(f"ELA error: {str(e)}")
            return {
                "score": 0,
                "regions": [],
                "error": str(e)
            }
    
    async def _noise_analysis(self, img_cv):
        """
        Detect inconsistent noise patterns, which can indicate manipulation.
        
        Different parts of an authentic image typically have consistent noise patterns.
        Inconsistencies can reveal splicing or other manipulations.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply median blur to estimate noise-free image
            median = cv2.medianBlur(gray, 5)
            
            # Calculate residual noise
            residual = cv2.absdiff(gray, median)
            
            # Enhance noise for better visualization
            residual_enhanced = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply adaptive thresholding to find inconsistent noise regions
            thresh = cv2.adaptiveThreshold(
                residual_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours of suspicious regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours
            min_area = img_cv.shape[0] * img_cv.shape[1] * 0.005  # 0.5% of image area
            suspicious_regions = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    suspicious_regions.append({
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": int(area),
                        "detection_method": "Noise Inconsistency"
                    })
            
            # Calculate manipulation score
            if suspicious_regions:
                total_suspicious_area = sum(region["area"] for region in suspicious_regions)
                image_area = img_cv.shape[0] * img_cv.shape[1]
                area_percentage = (total_suspicious_area / image_area) * 100
                
                # Cap at 80%
                manipulation_score = min(80, area_percentage * 2)
            else:
                manipulation_score = 0
            
            return {
                "score": manipulation_score,
                "regions": suspicious_regions,
                "analysis_method": "Noise Inconsistency"
            }
            
        except Exception as e:
            logger.error(f"Noise analysis error: {str(e)}")
            return {
                "score": 0,
                "regions": [],
                "error": str(e)
            }
    
    async def _dct_analysis(self, img_cv):
        """
        Analyze Discrete Cosine Transform (DCT) coefficients for manipulation detection.
        
        DCT analysis can reveal inconsistencies in JPEG compression patterns,
        which often occur in manipulated regions.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Analyze blocks of 8x8 pixels (standard JPEG block size)
            block_size = 8
            height, width = gray.shape
            block_counts = ((height // block_size), (width // block_size))
            
            # Array to store DCT energy for each block
            dct_energy = np.zeros((block_counts[0], block_counts[1]))
            
            # Process each block
            for i in range(block_counts[0]):
                for j in range(block_counts[1]):
                    # Extract block
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    
                    # Apply DCT
                    block_dct = dct(dct(block.T, norm='ortho').T, norm='ortho')
                    
                    # Calculate energy (excluding DC component)
                    energy = np.sum(np.abs(block_dct[1:, 1:]))
                    dct_energy[i, j] = energy
            
            # Normalize energy values
            dct_energy_norm = cv2.normalize(dct_energy, None, 0, 1, cv2.NORM_MINMAX)
            
            # Smooth for better visualization
            dct_energy_norm = cv2.resize(dct_energy_norm, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # Threshold to find inconsistent regions
            _, thresh = cv2.threshold(dct_energy_norm, 0.7, 1, cv2.THRESH_BINARY)
            
            # Convert to 8-bit for contour finding
            thresh_8bit = (thresh * 255).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(thresh_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate manipulation score based on DCT inconsistencies
            if len(contours) > 0:
                # More contours indicate more inconsistencies
                score = min(70, len(contours) * 5)
            else:
                score = 0
                
            return {
                "score": score,
                "dct_inconsistency_level": float(np.std(dct_energy_norm)),
                "analysis_method": "DCT Analysis"
            }
            
        except Exception as e:
            logger.error(f"DCT analysis error: {str(e)}")
            return {
                "score": 0,
                "error": str(e)
            }
    
    async def _clone_detection(self, img_cv):
        """
        Detect copy-pasted (cloned) regions in an image using feature matching.
        
        This identifies areas that have been duplicated within the same image,
        a common technique in image manipulation.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Use SIFT for feature detection
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            # If not enough keypoints, return early
            if descriptors is None or len(keypoints) < 10:
                return {
                    "score": 0,
                    "regions": [],
                    "analysis_method": "Clone Detection"
                }
            
            # Match features to themselves (to find duplicates)
            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(descriptors, descriptors, k=2)
            
            # Filter good matches (similar features but different locations)
            clone_matches = []
            for i, (m, n) in enumerate(matches):
                # Check if match is good and not the same point
                if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                    # Get coordinates
                    query_idx = m.queryIdx
                    train_idx = m.trainIdx
                    
                    p1 = keypoints[query_idx].pt
                    p2 = keypoints[train_idx].pt
                    
                    # Calculate distance between points
                    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    
                    # Only consider matches with significant distance
                    if distance > 50:  # Minimum distance threshold
                        clone_matches.append((keypoints[query_idx], keypoints[train_idx]))
            
            # Group nearby matches to find regions
            cloned_regions = []
            
            if len(clone_matches) > 5:  # At least 5 matches needed to consider cloning
                # Extract point coordinates
                points = []
                for kp1, kp2 in clone_matches:
                    points.append((int(kp1.pt[0]), int(kp1.pt[1])))
                    points.append((int(kp2.pt[0]), int(kp2.pt[1])))
                
                # Convert to numpy array
                points_array = np.array(points, dtype=np.int32)
                
                # Use DBSCAN clustering to group points
                if len(points_array) > 0:
                    # Scale the points to improve clustering
                    scaled_points = points_array.astype(np.float32)
                    scaled_points[:, 0] /= img_cv.shape[1]
                    scaled_points[:, 1] /= img_cv.shape[0]
                    
                    # Apply DBSCAN clustering
                    clustering = DBSCAN(eps=0.05, min_samples=3).fit(scaled_points)
                    
                    # Extract clusters
                    labels = clustering.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    # For each cluster, find bounding box
                    for i in range(n_clusters):
                        cluster_points = points_array[labels == i]
                        
                        if len(cluster_points) >= 3:  # At least 3 points
                            x, y, w, h = cv2.boundingRect(cluster_points)
                            
                            # Only add if significant size
                            min_area = img_cv.shape[0] * img_cv.shape[1] * 0.001
                            area = w * h
                            
                            if area > min_area:
                                cloned_regions.append({
                                    "x": int(x),
                                    "y": int(y),
                                    "width": int(w),
                                    "height": int(h),
                                    "area": int(area),
                                    "detection_method": "Clone Detection"
                                })
            
            # Calculate manipulation score
            manipulation_score = 0
            if cloned_regions:
                match_ratio = len(clone_matches) / len(keypoints)
                region_ratio = len(cloned_regions)
                
                # Combine both factors
                manipulation_score = min(85, (match_ratio * 50) + (region_ratio * 10))
                
            return {
                "score": manipulation_score,
                "regions": cloned_regions,
                "match_count": len(clone_matches),
                "analysis_method": "Clone Detection"
            }
            
        except Exception as e:
            logger.error(f"Clone detection error: {str(e)}")
            return {
                "score": 0,
                "regions": [],
                "error": str(e)
            }
    
    async def _jpeg_ghost_detection(self, img):
        """
        Detect JPEG Ghost artifacts that appear in double-compressed images.
        
        This technique identifies regions that have been inserted from another
        JPEG image with different compression parameters.
        """
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save original image at multiple JPEG qualities
            qualities = [65, 75, 85, 95]
            ghosts = []
            
            for quality in qualities:
                # Save at current quality
                temp_io = io.BytesIO()
                img.save(temp_io, 'JPEG', quality=quality)
                temp_io.seek(0)
                
                # Load saved image
                saved_img = Image.open(temp_io)
                saved_np = np.array(saved_img)
                
                # Calculate difference with original
                original_np = np.array(img)
                diff = np.abs(original_np.astype(np.float32) - saved_np.astype(np.float32))
                
                # Convert to grayscale if color
                if len(diff.shape) == 3:
                    diff = np.mean(diff, axis=2)
                
                # Normalize difference
                diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Apply threshold
                _, thresh = cv2.threshold(diff_norm, 30, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate ghost score for this quality
                if contours:
                    ghost_area = sum(cv2.contourArea(c) for c in contours)
                    image_area = img.width * img.height
                    ghost_percentage = (ghost_area / image_area) * 100
                    
                    ghosts.append({
                        "quality": quality,
                        "ghost_percentage": ghost_percentage
                    })
            
            # Calculate maximum ghost effect
            if ghosts:
                max_ghost = max(ghosts, key=lambda x: x["ghost_percentage"])
                ghost_score = min(75, max_ghost["ghost_percentage"] * 3)
            else:
                ghost_score = 0
                
            return {
                "score": ghost_score,
                "ghosts": ghosts,
                "analysis_method": "JPEG Ghost"
            }
            
        except Exception as e:
            logger.error(f"JPEG ghost detection error: {str(e)}")
            return {
                "score": 0,
                "error": str(e)
            }