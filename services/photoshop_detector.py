import cv2
import numpy as np
import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class PhotoshopDetector:
    """Detects Photoshop manipulation using Error Level Analysis and other techniques."""
    
    def __init__(self):
        self.ela_quality = 90  # Quality level for ELA
    
    async def detect(self, img):
        """
        Detect image manipulation using Error Level Analysis.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Detection results
        """
        try:
            # Convert PIL Image to OpenCV format
            img_cv = self._pil_to_cv2(img)
            
            # Perform Error Level Analysis
            ela_result = self._error_level_analysis(img)
            
            # Detect inconsistent noise patterns
            noise_result = self._detect_noise_inconsistency(img_cv)
            
            # Detect copy-paste regions
            clone_result = self._detect_cloned_regions(img_cv)
            
            # Analyze manipulation probability
            regions = []
            
            if ela_result["suspicious_regions"]:
                regions.extend(ela_result["suspicious_regions"])
            
            if noise_result["suspicious_regions"]:
                regions.extend(noise_result["suspicious_regions"])
            
            if clone_result["cloned_regions"]:
                regions.extend(clone_result["cloned_regions"])
            
            # Calculate overall probability
            manipulation_probability = max(
                ela_result["manipulation_score"],
                noise_result["manipulation_score"],
                clone_result["manipulation_score"]
            )
            
            # Invert for consistency (100 = not manipulated)
            authenticity_score = 100 - manipulation_probability
            
            return {
                "score": authenticity_score,
                "manipulation_probability": manipulation_probability,
                "manipulated_regions": regions,
                "ela_result": ela_result,
                "noise_result": noise_result,
                "clone_result": clone_result,
                "techniques_used": [
                    "Error Level Analysis (ELA)",
                    "Noise Inconsistency Detection",
                    "Clone Detection"
                ]
            }
            
        except Exception as e:
            logger.error(f"Photoshop detection error: {str(e)}")
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
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_cv
    
    def _error_level_analysis(self, img):
        """
        Perform Error Level Analysis to detect manipulated regions.
        ELA identifies areas that have different compression levels,
        which can indicate manipulation.
        """
        try:
            # Save to a temporary JPEG with known quality
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=self.ela_quality)
            temp_io.seek(0)
            
            # Load the saved image
            saved_img = Image.open(temp_io)
            
            # Calculate the difference
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
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "area": area,
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
                "manipulation_score": manipulation_score,
                "suspicious_regions": suspicious_regions,
                "analysis_method": "Error Level Analysis"
            }
            
        except Exception as e:
            logger.error(f"ELA error: {str(e)}")
            return {
                "manipulation_score": 0,
                "suspicious_regions": [],
                "error": str(e)
            }
    
    def _detect_noise_inconsistency(self, img_cv):
        """
        Detect inconsistent noise patterns, which can indicate manipulation.
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply median blur to reduce noise
            median = cv2.medianBlur(gray, 5)
            
            # Calculate residual noise
            residual = cv2.absdiff(gray, median)
            
            # Apply adaptive thresholding to find inconsistent noise regions
            thresh = cv2.adaptiveThreshold(
                residual, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
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
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "area": area,
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
                "manipulation_score": manipulation_score,
                "suspicious_regions": suspicious_regions,
                "analysis_method": "Noise Inconsistency"
            }
            
        except Exception as e:
            logger.error(f"Noise analysis error: {str(e)}")
            return {
                "manipulation_score": 0,
                "suspicious_regions": [],
                "error": str(e)
            }
    
    def _detect_cloned_regions(self, img_cv):
        """
        Detect copy-pasted (cloned) regions in an image.
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
                    "manipulation_score": 0,
                    "cloned_regions": [],
                    "analysis_method": "Clone Detection"
                }
            
            # Match features to themselves
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors, descriptors, k=2)
            
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
                    from sklearn.cluster import DBSCAN
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
                "manipulation_score": manipulation_score,
                "cloned_regions": cloned_regions,
                "match_count": len(clone_matches),
                "analysis_method": "Clone Detection"
            }
            
        except Exception as e:
            logger.error(f"Clone detection error: {str(e)}")
            return {
                "manipulation_score": 0,
                "cloned_regions": [],
                "error": str(e)
            }
