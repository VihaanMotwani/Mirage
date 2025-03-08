import exifread
import io
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetadataAnalyzer:
    """Analyzes image metadata and EXIF data for inconsistencies."""
    
    def __init__(self):
        self.suspicious_patterns = [
            {"name": "missing_creation_date", "description": "Image has no creation date"},
            {"name": "future_date", "description": "Image has a creation date in the future"},
            {"name": "missing_camera_info", "description": "Image has no camera information"},
            {"name": "mismatched_timestamps", "description": "Multiple timestamps in metadata don't match"},
            {"name": "wiped_metadata", "description": "Image has minimal or no metadata"},
            {"name": "edited_software", "description": "Image has been processed with editing software"}
        ]
        logger.info("MetadataAnalyzer initialized with %d suspicious patterns", len(self.suspicious_patterns))
    
    async def analyze(self, img, image_data):
        """
        Analyze image metadata for suspicious patterns.
        
        Args:
            img: PIL Image object
            image_data: Raw image bytes
            
        Returns:
            dict: Metadata analysis results
        """
        logger.info("Starting metadata analysis")
        try:
            # Extract EXIF data using PIL
            exif_tags = {}
            exif = img._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_tags[tag] = str(value)
                logger.info("Extracted %d EXIF tags from PIL", len(exif_tags))
            else:
                logger.warning("No EXIF data found using PIL")
            
            # Get more detailed EXIF with exifread
            exif_data = exifread.process_file(io.BytesIO(image_data))
            detailed_exif = {}
            for tag, value in exif_data.items():
                detailed_exif[tag] = str(value)
            logger.info("Extracted detailed EXIF data with %d tags", len(detailed_exif))
            
            # Check for suspicious patterns
            anomalies = []
            
            # Check for missing creation date
            has_date = any(date_field in exif_tags for date_field in 
                          ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized'])
            if not has_date:
                logger.info("Missing creation date detected")
                anomalies.append(self.suspicious_patterns[0])
            
            # Check for future dates
            current_date = datetime.now()
            for date_field in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized']:
                if date_field in exif_tags:
                    try:
                        # Parse date in format: '2023:10:15 14:30:00'
                        date_str = exif_tags[date_field]
                        img_date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                        if img_date > current_date:
                            logger.info("Future date detected in field %s: %s", date_field, date_str)
                            anomalies.append(self.suspicious_patterns[1])
                            break
                    except (ValueError, TypeError):
                        logger.warning("Invalid date format in field %s: %s", date_field, exif_tags[date_field])
                        anomalies.append({"name": "invalid_date_format", 
                                          "description": f"Invalid date format in {date_field}"})
            
            # Check for missing camera info
            has_camera_info = any(camera_field in exif_tags for camera_field in 
                                 ['Make', 'Model', 'LensMake', 'LensModel'])
            if not has_camera_info:
                logger.info("Missing camera information detected")
                anomalies.append(self.suspicious_patterns[2])
            
            # Check for timestamp mismatches
            date_fields = [field for field in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized'] 
                           if field in exif_tags]
            if len(date_fields) > 1:
                dates = [exif_tags[field] for field in date_fields]
                if len(set(dates)) > 1:
                    logger.info("Timestamp mismatches detected: %s", dates)
                    anomalies.append(self.suspicious_patterns[3])
            
            # Check for minimal metadata (possibly wiped)
            if len(exif_tags) < 5:
                logger.info("Minimal metadata detected: only %d EXIF tags found", len(exif_tags))
                anomalies.append(self.suspicious_patterns[4])
            
            # Check for editing software
            editing_software_fields = ['Software', 'ProcessingSoftware']
            editing_software_keywords = ['photoshop', 'lightroom', 'gimp', 'affinity', 'luminar']
            for field in editing_software_fields:
                if field in exif_tags:
                    software = exif_tags[field].lower()
                    if any(keyword in software for keyword in editing_software_keywords):
                        logger.info("Editing software detected in field %s: %s", field, exif_tags[field])
                        anomalies.append(self.suspicious_patterns[5])
                        break
            
            # Calculate metadata score (lower anomalies = higher score)
            max_anomalies = len(self.suspicious_patterns)
            anomaly_count = len(anomalies)
            metadata_score = 100 - (anomaly_count / max_anomalies * 100) if max_anomalies > 0 else 100
            logger.info("Metadata analysis completed with %d anomalies; score: %f", anomaly_count, metadata_score)
            
            return {
                "score": metadata_score,
                "exif_data": exif_tags,
                "detailed_exif": detailed_exif,
                "anomalies": anomalies,
                "has_metadata": len(exif_tags) > 0,
                "metadata_count": len(exif_tags)
            }
            
        except Exception as e:
            logger.error("Metadata analysis error: %s", str(e))
            return {
                "score": 0,
                "error": str(e),
                "exif_data": {},
                "anomalies": [{"name": "analysis_error", "description": f"Failed to analyze metadata: {str(e)}"}]
            }