import cv2
import numpy as np
import io
import logging
from PIL import Image

logger = logging.getLogger(__name__)

class PhotoshopDetector:
    """Detects Photoshop manipulation using Error Level Analysis and other techniques."""
    
    def __init__(self):
        self.ela_quality = 90  
        logger.info("PhotoshopDetector initialized with ELA quality %d", self.ela_quality)
    
    async def detect(self, img):
        """
        Detect image manipulation using Error Level Analysis.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Detection results
        """
        logger.info("Starting Photoshop detection")
        try:
            # Convert PIL Image to OpenCV format
            img_cv = self._pil_to_cv2(img)
            logger.debug("Converted PIL image to OpenCV format")
            
            # Perform Error Level Analysis
            logger.info("Performing Error Level Analysis (ELA)")
            ela_result = self._error_level_analysis(img)
            logger.debug("ELA result: %s", ela_result)
            
            # Detect inconsistent noise patterns
            logger.info("Detecting noise inconsistencies")
            noise_result = self._detect_noise_inconsistency(img_cv)
            logger.debug("Noise analysis result: %s", noise_result)
            
            # Detect copy-paste regions
            logger.info("Detecting cloned regions")
            clone_result = self._detect_cloned_regions(img_cv)
            logger.debug("Clone detection result: %s", clone_result)
            
            # Analyze manipulation probability by combining techniques
            regions = []
            if ela_result.get("suspicious_regions"):
                regions.extend(ela_result["suspicious_regions"])
            if noise_result.get("suspicious_regions"):
                regions.extend(noise_result["suspicious_regions"])
            if clone_result.get("cloned_regions"):
                regions.extend(clone_result["cloned_regions"])
            
            # Calculate overall probability based on highest score from each technique
            manipulation_probability = max(
                ela_result.get("manipulation_score", 0),
                noise_result.get("manipulation_score", 0),
                clone_result.get("manipulation_score", 0)
            )
            logger.info("Calculated manipulation probability: %f", manipulation_probability)
            
            # Invert for consistency (100 = not manipulated)
            authenticity_score = 100 - manipulation_probability
            logger.info("Final authenticity score: %f", authenticity_score)
            
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
            logger.error("Photoshop detection error: %s", str(e))
            return {
                "score": 50,  # Neutral score on error
                "error": str(e),
                "manipulation_probability": 0,
                "manipulated_regions": []
            }
    
    def _pil_to_cv2(self, pil_img):
        """Convert PIL Image to OpenCV format."""
        logger.info("Converting PIL image to OpenCV format")
        try:
            # Convert to RGB if in RGBA
            if pil_img.mode == 'RGBA':
                logger.debug("Image mode is RGBA; converting to RGB")
                pil_img = pil_img.convert('RGB')
            
            # Convert to numpy array and then to BGR format
            img_np = np.array(pil_img)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            logger.info("Successfully converted image to OpenCV format")
            return img_cv
        except Exception as e:
            logger.error("Error in _pil_to_cv2: %s", str(e))
            raise e
    
    def _error_level_analysis(self, img):
        """
        Perform Error Level Analysis to detect manipulated regions.
        ELA identifies areas that have different compression levels,
        which can indicate manipulation.
        """
        logger.info("Starting Error Level Analysis (ELA)")
        try:
            # Convert to RGB if image has alpha channel (RGBA)
            if img.mode == 'RGBA':
                logger.debug("Image mode is RGBA; converting to RGB for ELA")
                img = img.convert('RGB')
            
            # Save to a temporary JPEG with known quality
            temp_io = io.BytesIO()
            img.save(temp_io, 'JPEG', quality=self.ela_quality)
            temp_io.seek(0)
            logger.debug("Image saved temporarily with quality %d for ELA", self.ela_quality)
            
            # Load the saved image
            saved_img = Image.open(temp_io)
            
            # Create a new image for ELA results
            ela_img = Image.new('RGB', img.size, (0, 0, 0))
            
            # Compare original with resaved image pixel by pixel
            for x in range(img.width):
                for y in range(img.height):
                    orig_pixel = img.getpixel((x, y))
                    saved_pixel = saved_img.getpixel((x, y))
                    
                    # Calculate differences for each color channel
                    diff_r = abs(orig_pixel[0] - saved_pixel[0]) * 10
                    diff_g = abs(orig_pixel[1] - saved_pixel[1]) * 10
                    diff_b = abs(orig_pixel[2] - saved_pixel[2]) * 10
                    
                    # Scale differences for visibility
                    ela_pixel = (min(diff_r, 255), min(diff_g, 255), min(diff_b, 255))
                    ela_img.putpixel((x, y), ela_pixel)
            
            logger.info("Completed pixel-wise difference calculation for ELA")
            
            # Convert the ELA image to numpy array for further analysis
            ela_np = np.array(ela_img)
            gray_ela = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray_ela, 50, 255, cv2.THRESH_BINARY)
            logger.debug("Thresholding applied on ELA grayscale image")
            
            # Find contours of suspicious regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info("Found %d contours in ELA analysis", len(contours))
            
            # Filter small contours based on minimum area
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
            logger.info("Detected %d suspicious regions from ELA", len(suspicious_regions))
            
            # Calculate manipulation score based on suspicious regions
            if suspicious_regions:
                total_suspicious_area = sum(region["area"] for region in suspicious_regions)
                image_area = img.width * img.height
                area_percentage = (total_suspicious_area / image_area) * 100
                manipulation_score = min(90, area_percentage * 3)  # Scale for sensitivity, cap at 90%
                logger.info("ELA manipulation score calculated: %f", manipulation_score)
            else:
                manipulation_score = 0
                logger.info("No suspicious regions detected in ELA")
            
            return {
                "manipulation_score": manipulation_score,
                "suspicious_regions": suspicious_regions,
                "analysis_method": "Error Level Analysis"
            }
            
        except Exception as e:
            logger.error("ELA error: %s", str(e))
            return {
                "manipulation_score": 0,
                "suspicious_regions": [],
                "error": str(e)
            }
    
    def _detect_noise_inconsistency(self, img_cv):
        """
        Detect inconsistent noise patterns, which can indicate manipulation.
        """
        logger.info("Starting noise inconsistency detection")
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            logger.debug("Converted image to grayscale for noise analysis")
            
            # Apply median blur to reduce noise
            median = cv2.medianBlur(gray, 5)
            logger.debug("Applied median blur")
            
            # Calculate residual noise by comparing original and blurred image
            residual = cv2.absdiff(gray, median)
            
            # Apply adaptive thresholding to highlight noise inconsistencies
            thresh = cv2.adaptiveThreshold(
                residual, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            logger.debug("Adaptive thresholding applied on residual noise")
            
            # Find contours of suspicious regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logger.info("Found %d contours in noise analysis", len(contours))
            
            # Filter small contours based on image area
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
            logger.info("Detected %d suspicious regions from noise analysis", len(suspicious_regions))
            
            # Calculate manipulation score based on noise analysis
            if suspicious_regions:
                total_suspicious_area = sum(region["area"] for region in suspicious_regions)
                image_area = img_cv.shape[0] * img_cv.shape[1]
                area_percentage = (total_suspicious_area / image_area) * 100
                manipulation_score = min(80, area_percentage * 2)  # Scale for sensitivity, cap at 80%
                logger.info("Noise inconsistency manipulation score: %f", manipulation_score)
            else:
                manipulation_score = 0
                logger.info("No suspicious noise regions detected")
            
            return {
                "manipulation_score": manipulation_score,
                "suspicious_regions": suspicious_regions,
                "analysis_method": "Noise Inconsistency"
            }
            
        except Exception as e:
            logger.error("Noise analysis error: %s", str(e))
            return {
                "manipulation_score": 0,
                "suspicious_regions": [],
                "error": str(e)
            }
    
    def _detect_cloned_regions(self, img_cv):
        """
        Detect copy-pasted (cloned) regions in an image.
        """
        logger.info("Starting clone detection")
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            logger.debug("Converted image to grayscale for clone detection")
            
            # Use SIFT for feature detection
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            logger.info("Detected %d keypoints using SIFT", len(keypoints))
            
            # If not enough keypoints, return early
            if descriptors is None or len(keypoints) < 10:
                logger.warning("Not enough keypoints for clone detection")
                return {
                    "manipulation_score": 0,
                    "cloned_regions": [],
                    "analysis_method": "Clone Detection"
                }
            
            # Match features to themselves using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors, descriptors, k=2)
            logger.info("Performed feature matching using FLANN")
            
            # Filter good matches based on distance and distinct locations
            clone_matches = []
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance and m.queryIdx != m.trainIdx:
                    query_idx = m.queryIdx
                    train_idx = m.trainIdx
                    p1 = keypoints[query_idx].pt
                    p2 = keypoints[train_idx].pt
                    distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    if distance > 50:  # Minimum distance threshold
                        clone_matches.append((keypoints[query_idx], keypoints[train_idx]))
            logger.info("Found %d clone matches", len(clone_matches))
            
            # Group nearby matches to find cloned regions
            cloned_regions = []
            if len(clone_matches) > 5:  # At least 5 matches to consider cloning
                points = []
                for kp1, kp2 in clone_matches:
                    points.append((int(kp1.pt[0]), int(kp1.pt[1])))
                    points.append((int(kp2.pt[0]), int(kp2.pt[1])))
                points_array = np.array(points, dtype=np.int32)
                logger.debug("Extracted %d keypoint coordinates for clustering", len(points_array))
                
                # Scale points to improve clustering performance
                scaled_points = points_array.astype(np.float32)
                scaled_points[:, 0] /= img_cv.shape[1]
                scaled_points[:, 1] /= img_cv.shape[0]
                
                # Use DBSCAN clustering to group similar points
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(eps=0.05, min_samples=3).fit(scaled_points)
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                logger.info("DBSCAN clustering found %d clusters", n_clusters)
                
                for i in range(n_clusters):
                    cluster_points = points_array[labels == i]
                    if len(cluster_points) >= 3:  # Minimum of 3 points to form a region
                        x, y, w, h = cv2.boundingRect(cluster_points)
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
                logger.info("Detected %d cloned regions", len(cloned_regions))
            
            # Calculate manipulation score for clone detection
            manipulation_score = 0
            if cloned_regions:
                match_ratio = len(clone_matches) / len(keypoints)
                region_ratio = len(cloned_regions)
                manipulation_score = min(85, (match_ratio * 50) + (region_ratio * 10))
                logger.info("Clone detection manipulation score: %f", manipulation_score)
            else:
                logger.info("No cloned regions detected")
            
            return {
                "manipulation_score": manipulation_score,
                "cloned_regions": cloned_regions,
                "match_count": len(clone_matches),
                "analysis_method": "Clone Detection"
            }
            
        except Exception as e:
            logger.error("Clone detection error: %s", str(e))
            return {
                "manipulation_score": 0,
                "cloned_regions": [],
                "error": str(e)
            }