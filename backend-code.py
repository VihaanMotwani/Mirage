# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any, Union
import aiohttp
import io
import json
import uvicorn
import logging
from datetime import datetime

# Import service modules
from services.image_processor import ImageProcessor
from services.metadata_analyzer import MetadataAnalyzer
from services.reverse_image_search import ReverseImageSearch
from services.deepfake_detector import DeepfakeDetector
from services.photoshop_detector import PhotoshopDetector
from services.fact_checker import FactChecker
from services.trust_calculator import TrustScoreCalculator
from db.database import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Verification API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
metadata_analyzer = MetadataAnalyzer()
reverse_image_search = ReverseImageSearch()
deepfake_detector = DeepfakeDetector()
photoshop_detector = PhotoshopDetector()
fact_checker = FactChecker()
trust_calculator = TrustScoreCalculator()


class VerificationResponse(BaseModel):
    trust_score: float
    metadata_score: float
    reverse_image_score: float
    deepfake_score: float
    photoshop_score: float
    fact_check_score: float
    summary: str
    key_findings: List[str]
    metadata_results: Dict[str, Any]
    reverse_image_results: Dict[str, Any]
    deepfake_results: Dict[str, Any]
    photoshop_results: Dict[str, Any]
    fact_check_results: Dict[str, Any]


@app.get("/")
async def root():
    return {"message": "Image Verification API"}


@app.post("/api/verify", response_model=VerificationResponse)
async def verify_image(
    source_type: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    try:
        # Validate input
        if source_type == "upload" and not image:
            raise HTTPException(status_code=400, detail="Image file is required")
        if source_type == "url" and not image_url:
            raise HTTPException(status_code=400, detail="Image URL is required")

        # Process the image based on source type
        image_processor = ImageProcessor()
        if source_type == "upload":
            image_data = await image.read()
            img = image_processor.process_image_bytes(image_data)
        else:  # source_type == "url"
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
                    image_data = await response.read()
                    img = image_processor.process_image_bytes(image_data)

        # Run all verification services in parallel
        metadata_results = await metadata_analyzer.analyze(img, image_data)
        reverse_image_results = await reverse_image_search.search(img)
        deepfake_results = await deepfake_detector.detect(img)
        photoshop_results = await photoshop_detector.detect(img)
        
        # Use reverse image search keywords for fact checking
        keywords = reverse_image_results.get("keywords", [])
        fact_check_results = await fact_checker.check(img, keywords)
        
        # Calculate trust score
        trust_score, component_scores, summary, key_findings = trust_calculator.calculate(
            metadata_results,
            reverse_image_results,
            deepfake_results,
            photoshop_results,
            fact_check_results
        )
        
        # Create response
        response = {
            "trust_score": trust_score,
            "metadata_score": component_scores["metadata"],
            "reverse_image_score": component_scores["reverse_image"],
            "deepfake_score": component_scores["deepfake"],
            "photoshop_score": component_scores["photoshop"],
            "fact_check_score": component_scores["fact_check"],
            "summary": summary,
            "key_findings": key_findings,
            "metadata_results": metadata_results,
            "reverse_image_results": reverse_image_results,
            "deepfake_results": deepfake_results,
            "photoshop_results": photoshop_results,
            "fact_check_results": fact_check_results
        }
        
        # Log verification in database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO verification_logs (
                source_type, trust_score, metadata_score, reverse_image_score,
                deepfake_score, photoshop_score, fact_check_score, summary,
                timestamp, results_json
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                source_type, trust_score, component_scores["metadata"],
                component_scores["reverse_image"], component_scores["deepfake"],
                component_scores["photoshop"], component_scores["fact_check"],
                summary, datetime.now(), json.dumps(response)
            )
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return response
        
    except Exception as e:
        logger.error(f"Error in verification process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# services/image_processor.py
from PIL import Image
import io
import numpy as np
import cv2

class ImageProcessor:
    def process_image_bytes(self, image_bytes):
        """Process image bytes into a PIL Image object."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def to_cv2_image(self, pil_img):
        """Convert PIL Image to OpenCV format."""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# services/metadata_analyzer.py
import exifread
import io
from PIL import Image
from PIL.ExifTags import TAGS
import json
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
    
    async def analyze(self, img, image_data):
        """
        Analyze image metadata for suspicious patterns.
        
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
            
            # Check for suspicious patterns
            anomalies = []
            
            # Check for missing creation date
            has_date = any(date_field in exif_tags for date_field in 
                          ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized'])
            if not has_date:
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
                            anomalies.append(self.suspicious_patterns[1])
                            break
                    except (ValueError, TypeError):
                        # If date parsing fails, it's suspicious
                        anomalies.append({"name": "invalid_date_format", 
                                         "description": f"Invalid date format in {date_field}"})
            
            # Check for missing camera info
            has_camera_info = any(camera_field in exif_tags for camera_field in 
                                 ['Make', 'Model', 'LensMake', 'LensModel'])
            if not has_camera_info:
                anomalies.append(self.suspicious_patterns[2])
            
            # Check for timestamp mismatches
            date_fields = [field for field in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized'] 
                           if field in exif_tags]
            if len(date_fields) > 1:
                dates = [exif_tags[field] for field in date_fields]
                if len(set(dates)) > 1:
                    anomalies.append(self.suspicious_patterns[3])
            
            # Check for minimal metadata (possibly wiped)
            if len(exif_tags) < 5:
                anomalies.append(self.suspicious_patterns[4])
            
            # Check for editing software
            editing_software_fields = ['Software', 'ProcessingSoftware']
            editing_software_keywords = ['photoshop', 'lightroom', 'gimp', 'affinity', 'luminar']
            
            for field in editing_software_fields:
                if field in exif_tags:
                    software = exif_tags[field].lower()
                    if any(keyword in software for keyword in editing_software_keywords):
                        anomalies.append(self.suspicious_patterns[5])
                        break
            
            # Calculate metadata score (lower anomalies = higher score)
            max_anomalies = len(self.suspicious_patterns)
            anomaly_count = len(anomalies)
            metadata_score = 100 - (anomaly_count / max_anomalies * 100) if max_anomalies > 0 else 100
            
            return {
                "score": metadata_score,
                "exif_data": exif_tags,
                "detailed_exif": detailed_exif,
                "anomalies": anomalies,
                "has_metadata": len(exif_tags) > 0,
                "metadata_count": len(exif_tags)
            }
            
        except Exception as e:
            logger.error(f"Metadata analysis error: {str(e)}")
            return {
                "score": 0,
                "error": str(e),
                "exif_data": {},
                "anomalies": [{"name": "analysis_error", "description": f"Failed to analyze metadata: {str(e)}"}]
            }


# services/trust_calculator.py
class TrustScoreCalculator:
    """Calculates overall trust score based on all verification components."""
    
    def __init__(self):
        # Weights for each component in the final score
        self.weights = {
            "metadata": 0.15,
            "reverse_image": 0.25,
            "deepfake": 0.25,
            "photoshop": 0.25,
            "fact_check": 0.10
        }
    
    def calculate(self, metadata_results, reverse_image_results, 
                 deepfake_results, photoshop_results, fact_check_results):
        """
        Calculate the overall trust score and component scores.
        
        Args:
            metadata_results: Results from metadata analysis
            reverse_image_results: Results from reverse image search
            deepfake_results: Results from deepfake detection
            photoshop_results: Results from photoshop detection
            fact_check_results: Results from fact checking
        
        Returns:
            tuple: (trust_score, component_scores, summary, key_findings)
        """
        # Extract scores from each component
        metadata_score = metadata_results.get("score", 0)
        reverse_image_score = reverse_image_results.get("score", 0)
        deepfake_score = deepfake_results.get("score", 0)
        photoshop_score = photoshop_results.get("score", 0)
        fact_check_score = fact_check_results.get("score", 0)
        
        # Calculate weighted overall score
        trust_score = (
            self.weights["metadata"] * metadata_score +
            self.weights["reverse_image"] * reverse_image_score +
            self.weights["deepfake"] * deepfake_score +
            self.weights["photoshop"] * photoshop_score +
            self.weights["fact_check"] * fact_check_score
        )
        
        # Round to 1 decimal place
        trust_score = round(trust_score, 1)
        
        # Component scores dictionary
        component_scores = {
            "metadata": metadata_score,
            "reverse_image": reverse_image_score,
            "deepfake": deepfake_score,
            "photoshop": photoshop_score,
            "fact_check": fact_check_score
        }
        
        # Generate summary and key findings
        summary = self._generate_summary(trust_score, component_scores)
        key_findings = self._generate_key_findings(
            metadata_results, 
            reverse_image_results,
            deepfake_results,
            photoshop_results,
            fact_check_results
        )
        
        return trust_score, component_scores, summary, key_findings
    
    def _generate_summary(self, trust_score, component_scores):
        """Generate a summary based on the trust score."""
        if trust_score >= 80:
            return "This image appears to be authentic with high confidence. Most verification checks passed successfully."
        elif trust_score >= 60:
            return "This image shows some signs of potential manipulation or inconsistencies, but many verification checks passed."
        elif trust_score >= 40:
            return "This image has several suspicious characteristics that suggest it may be manipulated or misrepresented."
        else:
            return "This image shows strong evidence of manipulation, forgery, or misrepresentation. It should not be trusted."
    
    def _generate_key_findings(self, metadata_results, reverse_image_results,
                              deepfake_results, photoshop_results, fact_check_results):
        """Generate key findings based on component results."""
        findings = []
        
        # Add metadata findings
        if metadata_results.get("anomalies"):
            for anomaly in metadata_results["anomalies"][:3]:  # Limit to top 3
                findings.append(f"Metadata issue: {anomaly['description']}")
        
        # Add reverse image search findings
        if reverse_image_results.get("earliest_source"):
            findings.append(f"Earliest source: {reverse_image_results['earliest_source']['date']} from {reverse_image_results['earliest_source']['site']}")
        
        # Add deepfake detection findings
        if deepfake_results.get("is_deepfake", False):
            findings.append(f"Deepfake detection: {deepfake_results.get('confidence', 0)}% confidence this is AI-generated")
        
        # Add photoshop detection findings
        if photoshop_results.get("manipulated_regions"):
            regions = len(photoshop_results["manipulated_regions"])
            findings.append(f"Found {regions} potentially edited region(s) in the image")
        
        # Add fact check findings
        if fact_check_results.get("related_fact_checks"):
            for check in fact_check_results["related_fact_checks"][:2]:  # Limit to top 2
                findings.append(f"Fact check: {check['title']} - {check['rating']}")
        
        return findings
