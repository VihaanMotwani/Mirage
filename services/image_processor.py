# CLOUDINARY_URL=f"cloudinary://{api_key}:{api_secret}@{cloud_name}"
# upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET", "your_unsigned_upload_preset")

from PIL import Image
import io
import numpy as np
import cv2
import logging
import aiohttp
import os
import asyncio
import cloudinary

# Get Cloudinary credentials from environment variables
cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
api_key = os.getenv("CLOUDINARY_API_KEY")
api_secret = os.getenv("CLOUDINARY_API_SECRET")

# Configure Cloudinary with credentials
cloudinary.config(
    cloud_name=cloud_name,
    api_key=api_key,
    api_secret=api_secret
)

import cloudinary.uploader
import cloudinary.api
import time
from cloudinary import CloudinaryImage

logger = logging.getLogger(__name__)

async def upload_and_get_url(img):
    """
    Uploads a PIL image to Cloudinary using the official SDK and returns the image URL.
    
    Args:
        img: PIL Image object
    
    Returns:
        str: Secure URL of the uploaded image, or None if upload fails
    """
    try:
        # Convert PIL Image to bytes
        img_format = getattr(img, 'format', 'JPEG')
        if img_format.upper() not in ['PNG', 'JPG', 'JPEG', 'WEBP']:
            img_format = 'JPEG'
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        img_bytes = img_byte_arr.getvalue()
        
        # Generate a unique public ID based on timestamp
        timestamp = int(time.time())
        public_id = f"verification_{timestamp}"
        
        logger.info(f"Uploading image to Cloudinary with public_id: {public_id}")
        
        # Use a thread executor to run the synchronous Cloudinary upload in an async context
        loop = asyncio.get_event_loop()
        upload_result = await loop.run_in_executor(
            None,
            lambda: cloudinary.uploader.upload(
                img_bytes,
                folder="verification_images",       # Folder to store images
                public_id=public_id,                # Custom ID for the image
                overwrite=True,                     # Overwrite if exists
                notification_url=None,              # Optional notification URL
                resource_type="image"               # Specify resource type as image
            )
        )
        
        # Get the secure URL from the response
        secure_url = upload_result.get('secure_url')
        
        if secure_url:
            logger.info(f"Image uploaded successfully to Cloudinary: {secure_url}")
            return secure_url
        else:
            logger.error("No secure URL in Cloudinary response")
            return None
            
    except Exception as e:
        logger.error(f"Error uploading image to Cloudinary: {str(e)}")
        return None

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
            # Ensure the image is in RGB mode for proper conversion
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            logger.info("Successfully converted image to OpenCV format")
            return cv2_img
        except Exception as e:
            logger.error("Error converting PIL image to OpenCV format: %s", str(e))
            raise ValueError(f"Failed to convert image to cv2 format: {str(e)}")