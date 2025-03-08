import aiohttp
import logging
import io
import os
import base64

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Detects AI-generated or deepfake images."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
        self.openai_api_url = "https://api.openai.com/v1/detections"
        logger.info("DeepfakeDetector initialized")
    
    async def detect(self, img):
        """
        Detect if an image is AI-generated using multiple services.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Detection results
        """
        try:
            logger.info("Starting deepfake detection process")
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
            img_bytes = img_byte_arr.getvalue()
            logger.debug(f"Image converted to bytes, size: {len(img_bytes)} bytes")
            
            # Use the DALL-E service for detection
            logger.info("Calling _detect_with_dalle for deepfake analysis")
            dalle_result = await self._detect_with_dalle(img_bytes)
            logger.info("Received result from DALL-E detection")
            
            # Calculate combined score
            is_deepfake = dalle_result.get("is_ai_generated", False)
            dalle_confidence = dalle_result.get("confidence", 0)
            
            # Invert the score for consistency (100 = real, 0 = fake)
            authenticity_score = 100 - dalle_confidence if is_deepfake else 100
            logger.info(f"Deepfake detection completed: is_deepfake={is_deepfake}, "
                        f"confidence={dalle_confidence}, authenticity_score={authenticity_score}")
            
            return {
                "score": authenticity_score,
                "is_deepfake": is_deepfake,
                "confidence": dalle_confidence,
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
    
    async def _detect_with_dalle(self, img_bytes):
        """Use DALL-E API for detection."""
        try:
            logger.info("Preparing image for DALL-E API detection")
            # Encode image as base64
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            logger.debug("Image successfully encoded to base64")
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "image": img_base64
            }
            logger.debug("Prepared headers and data for DALL-E API request")
            
            # Make API request
            logger.info("Sending request to DALL-E API")
            async with aiohttp.ClientSession() as session:
                async with session.post(self.openai_api_url, headers=headers, json=data) as response:
                    if response.status != 200:
                        logger.error(f"DALL-E API returned non-200 status: {response.status}")
                        return {
                            "is_ai_generated": False,
                            "confidence": 0,
                            "error": f"API error: {response.status}"
                        }
                    
                    result = await response.json()
                    logger.info("Received successful response from DALL-E API")
            
            # Parse result
            is_ai_generated = result.get("ai_generated", False)
            confidence = result.get("confidence", 0) * 100  # Convert to percentage
            logger.debug(f"DALL-E API result parsed: is_ai_generated={is_ai_generated}, confidence={confidence}")
            
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