from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any, Union
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
from db.database import get_db_connection

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
    allow_origins=cors_origins,
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