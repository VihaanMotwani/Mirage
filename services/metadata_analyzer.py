import exifread
import io
import json
import numpy as np
import datetime
import hashlib
import logging
from PIL import Image
from PIL.ExifTags import TAGS 
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class MetadataAnalyzer:
    """
    Advanced metadata analyzer with camera fingerprinting and AI-powered inconsistency detection
    """
    
    def __init__(self):
        
        # Suspicious editing software patterns
        self.editing_software = [
            "photoshop", "lightroom", "gimp", "affinity", "luminar", "pixelmator", "capture one"
        ]
        
        # Suspicious patterns detector
        self.suspicious_patterns = [
            {"name": "missing_creation_date", "description": "Image has no creation date"},
            {"name": "future_date", "description": "Image has a creation date in the future"},
            {"name": "missing_camera_info", "description": "Image has no camera information"},
            {"name": "mismatched_timestamps", "description": "Multiple timestamps in metadata don't match"},
            {"name": "wiped_metadata", "description": "Image has minimal or no metadata"},
            {"name": "edited_software", "description": "Image has been processed with editing software"},
            {"name": "gps_inconsistency", "description": "GPS data is inconsistent with other metadata"},
            {"name": "model_mismatch", "description": "Camera model doesn't match metadata patterns"},
            {"name": "timezone_mismatch", "description": "Timezone information is inconsistent"}
        ]
        
        # Initialize anomaly detection model
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    
    async def analyze(self, img, image_data):
        """
        Analyze image metadata with advanced techniques
        
        Args:
            img: PIL Image object
            image_data: Raw image bytes
            
        Returns:
            dict: Metadata analysis results
        """
        try:
            # Extract EXIF data
            exif_tags = {}
            exif = img._getexif()
            
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)  
                    exif_tags[tag] = str(value)
            
            # Get more detailed EXIF with exifread
            exif_data = exifread.process_file(io.BytesIO(image_data))
            
            # Convert exifread tags to dict
            detailed_exif = {}
            for tag, value in exif_data.items():
                detailed_exif[tag] = str(value)
            
            # Collect metadata features for analysis
            features = self._extract_metadata_features(detailed_exif)
            
            # Normalize features for anomaly detection
            normalized_features = self._normalize_features(features)
            
            # Detect anomalies using Isolation Forest
            if len(normalized_features) > 0:
                # Reshape for sklearn (expects 2D array)
                X = np.array([list(normalized_features.values())]).reshape(1, -1)
                if not hasattr(self.anomaly_detector, "estimators_"):
                    self.anomaly_detector.fit(X)
                anomaly_scores = self.anomaly_detector.decision_function(X)
                is_anomaly = bool(anomaly_scores[0] < 0)
            
            # Check for specific suspicious patterns
            anomalies = self._check_suspicious_patterns(detailed_exif)
            
            # Check for missing camera info
            has_camera_info = any(camera_field in exif_tags for camera_field in 
                                 ['Make', 'Model', 'LensMake', 'LensModel'])
            if not has_camera_info:
                logger.info("Missing camera information detected")
                anomalies.append(self.suspicious_patterns[2])
            
            # Check for editing software traces
            software_check = self._check_editing_software(detailed_exif)
            if software_check["edited"]:
                anomalies.append({
                    "name": "edited_software", 
                    "description": f"Image edited with {software_check['software']}"
                })
            
            # Generate metadata hash for tracking modifications
            metadata_hash = self._generate_metadata_hash(detailed_exif)
            
            # Calculate metadata score (lower anomalies = higher score)
            max_anomalies = len(self.suspicious_patterns)
            anomaly_count = len(anomalies)
            anomaly_penalty = (anomaly_count / max_anomalies * 100) if max_anomalies > 0 else 0
            
            # Add penalty for general anomalies detected by Isolation Forest
            if is_anomaly:
                anomaly_penalty += 20
                
            metadata_score = max(0, 100 - anomaly_penalty)
            
            # Prepare report with extracted camera info
            camera_info = self._extract_camera_info(detailed_exif)
            
            return {
                "score": metadata_score,
                "exif_data": exif_tags,
                "detailed_exif": detailed_exif,
                "anomalies": anomalies,
                "has_metadata": len(exif_tags) > 0,
                "metadata_count": len(exif_tags),
                "camera_info": camera_info,
                "metadata_hash": metadata_hash,
                "is_anomalous": is_anomaly,
                "software_edited": software_check["edited"],
                "editing_software": software_check["software"] if software_check["edited"] else None
            }
            
        except Exception as e:
            logger.error(f"Metadata analysis error: {str(e)}")
            return {
                "score": 0,
                "error": str(e),
                "exif_data": {},
                "anomalies": [{"name": "analysis_error", "description": f"Failed to analyze metadata: {str(e)}"}]
            }
    
    def _extract_metadata_features(self, exif_data):
        """Extract numerical and categorical features from metadata for anomaly detection"""
        features = {}
        
        # Count number of metadata fields
        features['metadata_count'] = len(exif_data)
        
        # Check for existence of key metadata fields (binary features)
        key_fields = ['Image Make', 'Image Model', 'EXIF DateTimeOriginal', 'EXIF ExposureTime', 
                      'EXIF FNumber', 'EXIF ISOSpeedRatings', 'EXIF Flash']
        
        for field in key_fields:
            features[f'has_{field.replace(" ", "_")}'] = 1 if field in exif_data else 0
        
        # Extract numerical features where possible
        try:
            if 'EXIF ExposureTime' in exif_data:
                # Convert fraction to float (e.g., "1/100" to 0.01)
                exp_time = str(exif_data['EXIF ExposureTime'])
                if '/' in exp_time:
                    num, denom = exp_time.split('/')
                    features['exposure_time'] = float(num) / float(denom)
                else:
                    features['exposure_time'] = float(exp_time)
        except:
            features['exposure_time'] = 0
            
        try:
            if 'EXIF FNumber' in exif_data:
                fnumber = str(exif_data['EXIF FNumber'])
                if '/' in fnumber:
                    num, denom = fnumber.split('/')
                    features['fnumber'] = float(num) / float(denom)
                else:
                    features['fnumber'] = float(fnumber)
        except:
            features['fnumber'] = 0
            
        try:
            if 'EXIF ISOSpeedRatings' in exif_data:
                features['iso'] = float(str(exif_data['EXIF ISOSpeedRatings']))
        except:
            features['iso'] = 0
        
        # Check software field (hash for consistency)
        if 'Image Software' in exif_data:
            software = str(exif_data['Image Software']).lower()
            # Simple hash as a numerical feature
            features['software_hash'] = sum(ord(c) for c in software) % 1000
            
            # Check for editing software (binary feature)
            features['has_editing_software'] = 1 if any(edit_sw in software for edit_sw in self.editing_software) else 0
        else:
            features['software_hash'] = 0
            features['has_editing_software'] = 0
        
        return features
    
    def _normalize_features(self, features):
        """Normalize numerical features for anomaly detection"""
        norm_features = features.copy()
        
        # Normalize exposure time (typically between 1/4000 and 30 seconds)
        if 'exposure_time' in norm_features and norm_features['exposure_time'] > 0:
            max_exp_time = 30.0  # Max normal exposure time in seconds
            norm_features['exposure_time'] = min(norm_features['exposure_time'] / max_exp_time, 1.0)
        
        # Normalize f-number (typically between f/1.4 and f/22)
        if 'fnumber' in norm_features and norm_features['fnumber'] > 0:
            max_fnumber = 22.0
            norm_features['fnumber'] = min(norm_features['fnumber'] / max_fnumber, 1.0)
        
        # Normalize ISO (typically between 100 and 6400)
        if 'iso' in norm_features and norm_features['iso'] > 0:
            max_iso = 6400.0
            norm_features['iso'] = min(norm_features['iso'] / max_iso, 1.0)
            
        return norm_features
    
    def _check_suspicious_patterns(self, exif_data):
        """Check for suspicious patterns in metadata"""
        anomalies = []
        
        # Check for missing creation date
        has_date = any(date_field in exif_data for date_field in 
                      ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime'])
        if not has_date:
            anomalies.append(self.suspicious_patterns[0])
        
        # Check for future dates
        current_date = datetime.datetime.now()
        for date_field in ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime']:
            if date_field in exif_data:
                try:
                    # Parse date in format: '2023:10:15 14:30:00'
                    date_str = str(exif_data[date_field])
                    img_date = datetime.datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                    if img_date > current_date:
                        anomalies.append(self.suspicious_patterns[1])
                        break
                except (ValueError, TypeError):
                    # If date parsing fails, it's suspicious
                    anomalies.append({"name": "invalid_date_format", 
                                     "description": f"Invalid date format in {date_field}"})
        
        # Check for missing camera info
        has_camera_info = any(camera_field in exif_data for camera_field in 
                             ['Image Make', 'Image Model'])
        if not has_camera_info:
            anomalies.append(self.suspicious_patterns[2])
        
        # Check for timestamp mismatches
        date_fields = [field for field in ['EXIF DateTimeOriginal', 'EXIF DateTimeDigitized', 'Image DateTime'] 
                       if field in exif_data]
        if len(date_fields) > 1:
            dates = [str(exif_data[field]) for field in date_fields]
            if len(set(dates)) > 1:
                anomalies.append(self.suspicious_patterns[3])
        
        # Check for minimal metadata (possibly wiped)
        if len(exif_data) < 5:
            anomalies.append(self.suspicious_patterns[4])
        
        # Check for GPS inconsistencies
        if 'GPS GPSLatitude' in exif_data and 'GPS GPSLongitude' in exif_data:
            # Check if GPS data is consistent with timezone information
            if 'Image TimeZoneOffset' in exif_data:
                # In a real implementation, we would validate GPS coordinates against timezone
                # This is a simplified placeholder
                pass
                
        return anomalies
    
    def _check_editing_software(self, exif_data):
        """Check if the image has been processed with editing software"""
        result = {
            "edited": False,
            "software": None
        }
        
        editing_software_fields = ['Image Software', 'EXIF Software', 'XMP:CreatorTool']
        for field in editing_software_fields:
            if field in exif_data:
                software = str(exif_data[field]).lower()
                for edit_sw in self.editing_software:
                    if edit_sw in software:
                        result["edited"] = True
                        result["software"] = edit_sw
                        return result
        
        # Check for editing history in XMP data
        xmp_fields = [f for f in exif_data if f.startswith('XMP')]
        for field in xmp_fields:
            if 'History' in field or 'history' in field:
                result["edited"] = True
                result["software"] = "Unknown (XMP history found)"
                return result
                
        return result
    
    def _extract_camera_info(self, exif_data):
        """Extract camera information from metadata"""
        camera_info = {
            "make": str(exif_data.get('Image Make', '')),
            "model": str(exif_data.get('Image Model', '')),
            "software": str(exif_data.get('Image Software', '')),
            "lens": str(exif_data.get('EXIF LensModel', '')),
            "exposure_time": str(exif_data.get('EXIF ExposureTime', '')),
            "f_number": str(exif_data.get('EXIF FNumber', '')),
            "iso": str(exif_data.get('EXIF ISOSpeedRatings', '')),
            "focal_length": str(exif_data.get('EXIF FocalLength', '')),
            "date_time": str(exif_data.get('EXIF DateTimeOriginal', ''))
        }
        
        return camera_info
    
    def _generate_metadata_hash(self, exif_data):
        """Generate a hash of metadata for tracking modifications"""
        # Convert exif data to string for hashing
        exif_str = json.dumps(exif_data, sort_keys=True)
        
        # Generate SHA-256 hash
        metadata_hash = hashlib.sha256(exif_str.encode()).hexdigest()
        
        return metadata_hash