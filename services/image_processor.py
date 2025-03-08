from PIL import Image
import io
import numpy as np
import cv2
import logging
import aiohttp
import os

logger = logging.getLogger(__name__)

cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "your_cloud_name")
upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET", "your_unsigned_upload_preset")

async def upload_and_get_url(img):
    """
    Uploads a PIL image to Cloudinary using an unsigned upload preset and returns the image URL.
    """
    try:
        img_format = getattr(img, 'format', 'JPEG')
        if img_format.upper() not in ['PNG', 'JPG', 'JPEG', 'WEBP']:
            img_format = 'JPEG'
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        img_bytes = img_byte_arr.getvalue()
        
        upload_url = f"https://api.cloudinary.com/v1_1/{cloud_name}/upload"
        
        data = aiohttp.FormData()
        data.add_field("file",
                    img_bytes,
                    filename="image." + img_format.lower(),
                    content_type="image/" + img_format.lower())
        data.add_field("upload_preset", upload_preset)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(upload_url, data=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    link = result.get("secure_url")
                    logger.info("Image uploaded successfully to Cloudinary: %s", link)
                    return link
                else:
                    error_message = await resp.text()
                    logger.error("Cloudinary upload failed with status %d: %s", resp.status, error_message)
                    return None
    except Exception as e:
        logger.error("Error uploading image to Cloudinary: %s", str(e))
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