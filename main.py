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
from services.image_processor import ImageProcessor
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
cors_origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000")
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

        # Process the image based on source type
        image_processor = ImageProcessor()
        if source_type == "upload":
            image_data = await image.read()
            img = image_processor.process_image_bytes(image_data)
            logger.info("Image processed from upload")
        else:  # source_type == "url"
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch image from URL: {image_url}")
                        raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
                    image_data = await response.read()
                    img = image_processor.process_image_bytes(image_data)
                    logger.info("Image processed from URL")

        # Run all verification services in parallel
        logger.info("Starting metadata analysis")
        metadata_results = await metadata_analyzer.analyze(img, image_data)
        logger.info("Metadata analysis completed")

        logger.info("Starting reverse image search")
        reverse_image_results = await reverse_image_search.search(img)
        logger.info("Reverse image search completed")

        logger.info("Starting deepfake detection")
        deepfake_results = await deepfake_detector.detect(img)
        logger.info("Deepfake detection completed")

        logger.info("Starting Photoshop detection")
        photoshop_results = await photoshop_detector.detect(img)
        logger.info("Photoshop detection completed")
        
        # Use reverse image search keywords for fact checking
        keywords = reverse_image_results.get("keywords", [])
        logger.info(f"Starting fact checking with keywords: {keywords}")
        fact_check_results = await fact_checker.check(img, keywords)
        logger.info("Fact checking completed")
        
        # Calculate trust score
        logger.info("Calculating trust score")
        trust_score, component_scores, summary, key_findings = trust_calculator.calculate(
            metadata_results,
            reverse_image_results,
            deepfake_results,
            photoshop_results,
            fact_check_results
        )
        logger.info(f"Trust score calculated: {trust_score}")
        
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