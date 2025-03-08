import logging
import json
import os

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Detects AI-generated or deepfake images using a public image URL."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "your_openai_api_key")
        logger.info("DeepfakeDetector initialized")
    
    async def detect(self, image_url):
        """
        Expects a publicly accessible image URL.
        """
        try:
            logger.info("Starting deepfake detection process")
            if not isinstance(image_url, str):
                logger.error("Expected image URL as a string")
                return {
                    "score": 50,
                    "error": "Input must be a URL string",
                    "is_deepfake": False,
                    "confidence": 0
                }
            
            logger.debug(f"Using image URL: {image_url}")
            
            # Call the OpenAI API for detection with the provided URL
            logger.info("Calling _detect_with_dalle for deepfake analysis")
            dalle_result = await self._detect_with_dalle(image_url)
            logger.info("Received result from DALL-E detection")
            
            is_deepfake = dalle_result.get("is_ai_generated", False)
            dalle_confidence = dalle_result.get("confidence", 0)
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
                "score": 50,
                "error": str(e),
                "is_deepfake": False,
                "confidence": 0
            }
    
    async def _detect_with_dalle(self, image_url):
        try:
            logger.info("Preparing image for OpenAI API detection using public URL")
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": (
                                    "Analyze this image and determine if it's AI-generated or a deepfake. "
                                    "Respond with a JSON object that has 'is_ai_generated' as true or false "
                                    "and 'confidence' as a number between 0 and 1."
                                )
                            },
                            {
                                "type": "image_url", 
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                "response_format": {"type": "json_object"}
            }
            
            logger.debug("Prepared headers and data for OpenAI API request")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers, json=data
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            f"OpenAI API returned non-200 status: {response.status}, details: {error_text}"
                        )
                        return {
                            "is_ai_generated": False,
                            "confidence": 0,
                            "error": f"API error: {response.status}: {error_text}"
                        }
                    
                    result = await response.json()
                    logger.info("Received successful response from OpenAI API")
            
            # Parse the result from the GPT-4 response
            if not result or "choices" not in result or not result["choices"]:
                logger.error(f"Unexpected API response format: {result}")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": "Invalid API response format"
                }
                    
            content = result["choices"][0]["message"]["content"]
            if not content:
                logger.error("Empty content in API response")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": "Empty API response content"
                }
            
            analysis = json.loads(content)
            is_ai_generated = analysis.get("is_ai_generated", False)
            confidence = analysis.get("confidence", 0) * 100  # Convert to percentage
            
            logger.debug(f"OpenAI API result parsed: is_ai_generated={is_ai_generated}, confidence={confidence}")
            
            return {
                "is_ai_generated": is_ai_generated,
                "confidence": confidence,
                "details": analysis
            }
                
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                "is_ai_generated": False,
                "confidence": 0,
                "error": str(e)
            }