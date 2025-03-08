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