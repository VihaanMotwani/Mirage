from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import aiohttp
import io
import json
import uvicorn
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import service modules
from services.image_processor import ImageProcessor, upload_and_get_url
from services.metadata_analyzer import MetadataAnalyzer
from services.reverse_image_search import ReverseImageSearch
from services.deepfake_detector import DeepfakeDetector
from services.photoshop_detector import PhotoshopDetector
from services.fact_checker import FactChecker
from services.trust_calculator import TrustScoreCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Verification API")

# Get CORS settings from environment variables
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000", "https://mirage-image.app")
cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS configured with origins: {cors_origins}")

# Initialize services
metadata_analyzer = MetadataAnalyzer()
reverse_image_search = ReverseImageSearch()
deepfake_detector = DeepfakeDetector()
photoshop_detector = PhotoshopDetector()
fact_checker = FactChecker()
trust_calculator = TrustScoreCalculator()

# In-memory storage for verification results (temporary replacement for database)
verification_history = []

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
    logger.info("Root endpoint accessed")
    return {"message": "Image Verification API"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "metadata_analyzer": "available",
            "reverse_image_search": "available",
            "deepfake_detector": "available",
            "photoshop_detector": "available",
            "fact_checker": "available"
        }
    }


@app.post("/api/verify", response_model=VerificationResponse)
async def verify_image(
    source_type: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    logger.info(f"Verification started with source_type: {source_type}")
    try:
        # Validate input
        if source_type == "upload" and not image:
            logger.error("Image file not provided for upload source")
            raise HTTPException(status_code=400, detail="Image file is required")
        if source_type == "url" and not image_url:
            logger.error("Image URL not provided for URL source")
            raise HTTPException(status_code=400, detail="Image URL is required")

        # Process the image based on source type using ImageProcessor
        image_processor = ImageProcessor()
        if source_type == "upload":
            image_data = await image.read()
            img = image_processor.process_image_bytes(image_data)
            logger.info("Image processed from upload")

            # Try to upload to Cloudinary, but don't fail if it doesn't work
            try:
                img_url = await upload_and_get_url(img)
                if not img_url:
                    logger.warning("Failed to get upload to Cloudinary, will use fallback methods.")
            except Exception as e:
                logger.warning(f"Cloudinary upload failed: {str(e)}, will use fallback methods.")
                img_url = None

        else:  # source_type == "url"
            img_url = image_url
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(image_url) as response:
                        if response.status != 200:
                            logger.error(f"Failed to fetch image from URL: {image_url}")
                            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
                        image_data = await response.read()
                        img = image_processor.process_image_bytes(image_data)
                        logger.info("Image processed from URL")
                except Exception as e:
                    logger.error(f"Error processing URL: {str(e)}")
                    raise HTTPException(status_code=400, detail=f"Error processing image URL: {str(e)}")
    
        # Run verification services
        logger.info("Starting metadata analysis")
        try:
            metadata_results = await metadata_analyzer.analyze(img, image_data)
            logger.info("Metadata analysis completed")

        except Exception as e:
            logger.error(f"Metadata analysis failed: {str(e)}")
            metadata_results = {"score": 50, "error": str(e), "exif_data": {}, "anomalies": []}

        logger.info("Starting reverse image search")
        if img_url:
            try:
                reverse_image_results = await reverse_image_search.search(img_url)
                logger.info("Reverse image search completed")
            except Exception as e:
                logger.error(f"Reverse image search failed: {str(e)}")
                reverse_image_results = {"score": 50, "error": str(e), "content_context": []}
        else:
            logger.warning("Skipping reverse image search due to missing image URL")
            reverse_image_results = {"score": 50, "error": "No image URL available", "content_context": []}


        logger.info("Starting deepfake detection")
        if img_url:
            try:
                # detector = DeepfakeDetector(model_path="path/to/downloaded/model/final_999_DeepFakeClassifier_EfficientNetB7_face_2.pt")
                # deepfake_results = await detector.detect(img)
                deepfake_results = await deepfake_detector.detect(img_url)
                logger.info("Deepfake detection completed")
            except Exception as e:
                logger.error(f"Deepfake detection failed: {str(e)}")
                deepfake_results = {"score": 50, "error": str(e), "is_deepfake": False}
        else:
            logger.warning("Skipping deepfake detection due to missing image URL")
            deepfake_results = {"score": 50, "error": "No image URL available", "is_deepfake": False}
       

        logger.info("Starting Photoshop detection")
        try:
            photoshop_results = await photoshop_detector.detect(img)
            logger.info("Photoshop detection completed")
        except Exception as e:
            logger.error(f"Photoshop detection failed: {str(e)}")
            photoshop_results = {"score": 50, "error": str(e), "manipulation_probability": 0}

        # Use reverse image search's content_context for fact checking
        content_context = reverse_image_results.get("content_context", [])

        # If reverse image search didn't provide context, create a basic one
        if not content_context:
            logger.info(f"Starting fact checking with content context: {content_context}")
            # Extract EXIF data if available
            exif_context = []
            exif_data = metadata_results.get("exif_data", {})
            if exif_data:
                for key, value in exif_data.items():
                    if key in ["ImageDescription", "UserComment", "XPComment", "XPSubject", "XPTitle", "Comment"]:
                        if value and len(str(value)) > 5:  # Only use non-empty meaningful fields
                            exif_context.append(str(value))
            
            # Create a basic context
            basic_context = {
                "title": "Image verification analysis",
                "description": "Image submitted for verification"
            }
            
            # Add file metadata if available
            if source_type == "upload" and image:
                basic_context["description"] += f" - filename: {image.filename}"
            
            # Add any EXIF context
            if exif_context:
                basic_context["description"] += " - " + " ".join(exif_context)
                
            content_context = [basic_context]

        # Fact checking
        logger.info(f"Starting fact checking with {len(content_context)} context items")
        try:
            fact_check_results = await fact_checker.check(content_context)
            logger.info("Fact checking completed")
        except Exception as e:
            logger.error(f"Fact checking failed: {str(e)}")
            fact_check_results = {"score": 50, "error": str(e), "related_fact_checks": []}
        
        # Calculate trust score
        logger.info("Calculating trust score")
        try:
            trust_score, component_scores, summary, key_findings = trust_calculator.calculate(
                metadata_results,
                reverse_image_results,
                deepfake_results,
                photoshop_results,
                fact_check_results
            )
            logger.info(f"Trust score calculated: {trust_score}")
        except Exception as e:
            logger.error(f"Trust score calculation failed: {str(e)}")
            trust_score = 50
            component_scores = {
                "metadata": metadata_results.get("score", 50),
                "reverse_image": reverse_image_results.get("score", 50),
                "deepfake": deepfake_results.get("score", 50),
                "photoshop": photoshop_results.get("score", 50),
                "fact_check": fact_check_results.get("score", 50)
            }
            summary = "Verification results are inconclusive due to processing errors."
            key_findings = [f"Error: {str(e)}"]
            
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
        
        # Store verification result in memory (instead of database)
        verification_history.append({
            "source_type": source_type,
            "timestamp": datetime.now().isoformat(),
            "trust_score": trust_score,
            "results": response
        })
        logger.info("Verification result stored in history")
        
        # Keep only the last 50 verifications to prevent memory issues
        if len(verification_history) > 50:
            verification_history.pop(0)
            logger.info("Oldest verification record removed to maintain history limit")
        
        return response
        
    except Exception as e:
        logger.error(f"Error in verification process: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.get("/api/history")
async def get_verification_history(limit: int = 10):
    """Get recent verification history (from in-memory storage)."""
    logger.info(f"Fetching verification history with limit: {limit}")
    return verification_history[-limit:] if verification_history else []


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)