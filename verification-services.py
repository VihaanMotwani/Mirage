# services/reverse_image_search.py
import aiohttp
import logging
import json
import os
from datetime import datetime
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ReverseImageSearch:
    """Service for reverse image searching to find the earliest published version."""
    
    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY", "your_serp_api_key_here")
        self.api_url = "https://serpapi.com/search.json"
    
    async def search(self, img):
        """
        Perform reverse image search.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Results including earliest source, similar images, and reliability score
        """
        try:
            # Convert image to bytes for upload
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create form data for the API request
            data = aiohttp.FormData()
            data.add_field('api_key', self.api_key)
            data.add_field('engine', 'google_reverse_image')
            data.add_field('image_file', img_byte_arr)
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=data) as response:
                    if response.status != 200:
                        return {
                            "score": 0,
                            "error": f"API error: {response.status}",
                            "message": await response.text()
                        }
                    
                    result = await response.json()
            
            # Process results
            image_results = result.get("image_results", [])
            
            # Extract sources with dates
            sources_with_dates = []
            for img_result in image_results:
                source_name = img_result.get("source")
                link = img_result.get("link")
                snippet = img_result.get("snippet")
                
                # Try to extract date from snippet or link
                date = self._extract_date(snippet)
                
                if date:
                    domain = self._extract_domain(link)
                    sources_with_dates.append({
                        "date": date,
                        "timestamp": self._date_to_timestamp(date),
                        "source": source_name,
                        "site": domain,
                        "link": link,
                        "snippet": snippet
                    })
            
            # Sort by date (oldest first)
            sources_with_dates.sort(key=lambda x: x.get("timestamp", 0))
            
            # Extract keywords from related text
            related_text = [
                result.get("title", ""),
                result.get("snippet", "")
            ]
            related_text.extend([r.get("snippet", "") for r in image_results[:5]])
            keywords = self._extract_keywords(" ".join(related_text))
            
            # Calculate score based on:
            # 1. Do we have any dated sources?
            # 2. Is the earliest source from a reliable domain?
            # 3. How many different sources found the image?
            
            source_count = len(sources_with_dates)
            score = 0
            
            if source_count > 0:
                # Base score for having sources
                score = 50
                
                # Bonus for multiple sources
                if source_count > 1:
                    score += min(source_count * 5, 20)  # Up to 20 points for 4+ sources
                
                # Bonus for reliable domains
                earliest_source = sources_with_dates[0]
                reliable_domains = ["nytimes.com", "reuters.com", "apnews.com", "bbc.com", 
                                  "washingtonpost.com", "theguardian.com"]
                
                for domain in reliable_domains:
                    if domain in earliest_source.get("site", ""):
                        score += 15
                        break
                
                # Cap at 100
                score = min(score, 100)
            
            return {
                "score": score,
                "earliest_source": sources_with_dates[0] if sources_with_dates else None,
                "all_sources": sources_with_dates,
                "source_count": source_count,
                "keywords": keywords,
                "result_count": len(image_results)
            }
            
        except Exception as e:
            logger.error(f"Reverse image search error: {str(e)}")
            return {
                "score": 0,
                "error": str(e)
            }
    
    def _extract_date(self, text):
        """Extract date from text using regex patterns."""
        if not text:
            return None
        
        # Various date formats to try
        patterns = [
            r'(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4})',  # 15 Jan 2023
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4})',  # Jan 15, 2023
            r'(\d{4}-\d{2}-\d{2})',  # 2023-01-15
            r'(\d{1,2}/\d{1,2}/\d{4})',  # 1/15/2023 or 15/1/2023
            r'(\d{1,2}\.\d{1,2}\.\d{4})'  # 15.01.2023
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        
        return None
    
    def _date_to_timestamp(self, date_str):
        """Convert date string to timestamp for comparison."""
        try:
            # Try different date formats
            formats = [
                "%d %b %Y",  # 15 Jan 2023
                "%b %d, %Y",  # Jan 15, 2023
                "%b %d %Y",   # Jan 15 2023
                "%Y-%m-%d",   # 2023-01-15
                "%m/%d/%Y",   # 1/15/2023
                "%d/%m/%Y",   # 15/1/2023
                "%d.%m.%Y"    # 15.01.2023
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
                
            return 0  # Default if parsing fails
        except Exception:
            return 0
    
    def _extract_domain(self, url):
        """Extract domain from URL."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url
    
    def _extract_keywords(self, text):
        """Extract relevant keywords from text."""
        # Simple keyword extraction based on common words
        if not text:
            return []
        
        # Remove special chars and lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and count
        words = text.split()
        word_count = {}
        
        for word in words:
            if len(word) > 3:  # Skip short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Skip common stopwords
        stopwords = ["the", "and", "that", "this", "with", "from", "have", "for", "not", "are", "were"]
        for word in stopwords:
            if word in word_count:
                del word_count[word]
        
        # Sort by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 10 keywords
        return [word for word, count in sorted_words[:10]]


# services/deepfake_detector.py
import aiohttp
import logging
import io
import os
import base64
import json

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Detects AI-generated or deepfake images."""
    
    def __init__(self):
        self.deepware_api_key = os.getenv("DEEPWARE_API_KEY", "your_deepware_api_key")
        self.deepware_api_url = "https://api.deepware.ai/detect"
        
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
        self.openai_api_url = "https://api.openai.com/v1/detections"
    
    async def detect(self, img):
        """
        Detect if an image is AI-generated using multiple services.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Detection results
        """
        try:
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
            img_bytes = img_byte_arr.getvalue()
            
            # Use both services in parallel for cross-verification
            deepware_result = await self._detect_with_deepware(img_bytes)
            dalle_result = await self._detect_with_dalle(img_bytes)
            
            # Calculate combined score
            is_deepfake = deepware_result.get("is_deepfake", False) or dalle_result.get("is_ai_generated", False)
            
            # Calculate confidence as average of both services
            deepware_confidence = deepware_result.get("confidence", 0)
            dalle_confidence = dalle_result.get("confidence", 0)
            
            # Weighted average (giving more weight to higher confidence)
            if deepware_confidence > dalle_confidence:
                combined_confidence = (deepware_confidence * 0.7) + (dalle_confidence * 0.3)
            else:
                combined_confidence = (dalle_confidence * 0.7) + (deepware_confidence * 0.3)
            
            # Invert the score for consistency (100 = real, 0 = fake)
            authenticity_score = 100 - combined_confidence if is_deepfake else 100
            
            return {
                "score": authenticity_score,
                "is_deepfake": is_deepfake,
                "confidence": combined_confidence,
                "deepware_result": deepware_result,
                "dalle_result": dalle_result
            }
        
        except Exception as e:
            logger.error(f"Deepfake detection error: {str(e)}")
            return {
                "score": 50,  # Neutral score on error
                "error": str(e),
                "is_deepfake": False,
                "confidence": 0
            }
    
    async def _detect_with_deepware(self, img_bytes):
        """Use Deepware API for detection."""
        try:
            # Encode image as base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.deepware_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "image": img_base64
            }
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.deepware_api_url, 
                                       headers=headers, 
                                       json=data) as response:
                    if response.status != 200:
                        return {
                            "is_deepfake": False,
                            "confidence": 0,
                            "error": f"API error: {response.status}"
                        }
                    
                    result = await response.json()
            
            # Parse result
            is_deepfake = result.get("prediction", 0) > 0.5
            confidence = result.get("prediction", 0) * 100  # Convert to percentage
            
            return {
                "is_deepfake": is_deepfake,
                "confidence": confidence,
                "details": result
            }
            
        except Exception as e:
            logger.error(f"Deepware API error: {str(e)}")
            return {
                "is_deepfake": False,
                "confidence": 0,
                "error": str(e)
            }
    
    async def _detect_with_dalle(self, img_bytes):
        """Use DALL-E API for detection."""
        try:
            # Encode image as base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "image": img_base64
            }
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.openai_api_url, 
                                       headers=headers, 
                                       json=data) as response:
                    if response.status != 200:
                        return {
                            "is_ai_generated": False,
                            "confidence": 0,
                            "error": f"API error: {response.status}"
                        }
                    
                    result = await response.json()
            
            # Parse result
            is_ai_generated = result.get("ai_generated", False)
            confidence = result.get("confidence", 0) * 100  # Convert to percentage
            
            return {
                "is_ai_generated": is_ai_generated,
                "confidence": confidence,
                "details": result
            }
            
        except Exception as e:
            logger.error(f"DALL-E API error: {str(e)}")
            return {
                "is_ai_generated": False,
                "confidence": 0,
                "error": str(e)
            }


# services/photoshop_detector.py
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
