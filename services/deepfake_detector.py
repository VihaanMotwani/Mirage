import logging
import json
import os
import aiohttp

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Detects AI-generated or deepfake images using a public image URL."""
    
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable not set")
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
            logger.error(f"Deepfake detection error: {str(e)}", exc_info=True)
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
                        "role": "system",
                        "content": (
                            "You are a deepfake detection system. Analyze the provided image and determine if it is AI-generated or a deepfake. "
                            "Respond ONLY with a JSON object containing 'is_ai_generated' (boolean) and 'confidence' (float between 0 and 1). "
                            "Do not include any other text or formatting."
                        )
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text", 
                                "text": "Analyze this image for AI generation or deepfake indicators."
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
                    response_text = await response.text()
                    logger.debug(f"OpenAI API response: {response_text}")
                    
                    if response.status != 200:
                        logger.error(
                            f"OpenAI API returned non-200 status: {response.status}, details: {response_text}"
                        )
                        return {
                            "is_ai_generated": False,
                            "confidence": 0,
                            "error": f"API error: {response.status}: {response_text}"
                        }
                    
                    try:
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse OpenAI API response as JSON")
                        return {
                            "is_ai_generated": False,
                            "confidence": 0,
                            "error": "Invalid JSON response from API"
                        }
            
            logger.info("Received successful response from OpenAI API")
            
            # Validate API response structure
            if not isinstance(result, dict):
                logger.error(f"Unexpected API response format: {result}")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": "Invalid API response format"
                }
            
            if "choices" not in result or not isinstance(result["choices"], list) or len(result["choices"]) == 0:
                logger.error(f"Unexpected API response format: missing choices - {result}")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": "Invalid API response format: missing choices"
                }
            
            first_choice = result["choices"][0]
            if "message" not in first_choice or not isinstance(first_choice["message"], dict):
                logger.error(f"Unexpected API response format: missing message - {first_choice}")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": "Invalid API response format: missing message"
                }
            
            message = first_choice["message"]
            if "content" not in message or not message["content"]:
                logger.error("Empty content in API response")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": "Empty API response content"
                }
            
            content = message["content"]
            
            # Parse the JSON content
            try:
                analysis = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse content JSON: {e} - Content: {content}")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": f"Invalid JSON content: {str(e)}"
                }
            
            if not isinstance(analysis, dict):
                logger.error(f"Response content is not a JSON object: {content}")
                return {
                    "is_ai_generated": False,
                    "confidence": 0,
                    "error": "Response content is not a JSON object"
                }
            
            is_ai_generated = analysis.get("is_ai_generated", False)
            confidence = analysis.get("confidence", 0)
            
            # Validate confidence value
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence)) * 100  # Convert to percentage
            except (TypeError, ValueError):
                logger.warning(f"Invalid confidence value: {confidence}")
                confidence = 0.0
            
            logger.debug(f"OpenAI API result parsed: is_ai_generated={is_ai_generated}, confidence={confidence}")
            
            return {
                "is_ai_generated": is_ai_generated,
                "confidence": confidence,
                "details": analysis
            }
                
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}", exc_info=True)
            return {
                "is_ai_generated": False,
                "confidence": 0,
                "error": str(e)
            }