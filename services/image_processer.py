from PIL import Image
import io
import numpy as np
import cv2

class ImageProcessor:
    def process_image_bytes(self, image_bytes):
        """Process image bytes into a PIL Image object."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")
    
    def to_cv2_image(self, pil_img):
        """Convert PIL Image to OpenCV format."""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)