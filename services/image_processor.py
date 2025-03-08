from PIL import Image
import io
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    def process_image_bytes(self, image_bytes):
        """Process image bytes into a PIL Image object."""
        try:
            logger.info("Starting processing image bytes into PIL Image")
            img = Image.open(io.BytesIO(image_bytes))
            logger.info("Successfully processed image bytes into PIL Image")
            return img
        except Exception as e:
            logger.error("Failed to process image bytes: %s", str(e))
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def to_cv2_image(self, pil_img):
        """Convert PIL Image to OpenCV format."""
        try:
            logger.info("Converting PIL image to OpenCV format")
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            logger.info("Successfully converted image to OpenCV format")
            return cv2_img
        except Exception as e:
            logger.error("Error converting PIL image to OpenCV format: %s", str(e))
            raise ValueError(f"Failed to convert image to cv2 format: {str(e)}")