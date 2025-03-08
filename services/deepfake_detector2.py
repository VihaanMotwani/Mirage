import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
import sys

# Add the dfdc repository to the path
sys.path.append('dfdc_deepfake_challenge')

# Import the model and utilities from the cloned repo
from training.zoo import efficient_net
from training.tools.config import load_config

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Deepfake detection using pre-trained DFDC models."""
    
    def __init__(self, model_path="models/final_999_DeepFakeClassifier_EfficientNetB7_face_2.pt"):
        """
        Initialize the deepfake detector with a pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model checkpoint
        """
        logger.info("Initializing DeepfakeDetector")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model configuration
        self.model_config = {
            "encoder": "tf_efficientnet_b7_ns",
            "img_size": 380
        }
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.model_config["img_size"], self.model_config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("DeepfakeDetector initialization complete")
    
    def _load_model(self, model_path):
        """
        Load the pre-trained model.
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            torch.nn.Module: Loaded model
        """
        try:
            logger.info(f"Loading model from {model_path}")
            model = efficient_net(self.model_config["encoder"], pretrained=False)
            model.to(self.device)
            
            # Load state dict
            checkpoint = torch.load(model_path, map_location=self.device)
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    async def detect(self, image):
        """
        Detect if an image is a deepfake.
        
        Args:
            image: PIL Image object or path to image
            
        Returns:
            dict: Detection results with confidence score and analysis
        """
        logger.info("Starting deepfake detection")
        try:
            # Ensure we have a PIL image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                logger.error("Invalid image input format")
                return {"fake_score": 0.5, "error": "Invalid image format"}
            
            # Process the image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Model prediction
                predictions = self.model(input_tensor)
                fake_score = torch.sigmoid(predictions).item()
            
            # Process results
            result = {
                "fake_score": fake_score,
                "is_fake": fake_score > 0.5,
                "confidence": abs(fake_score - 0.5) * 2,  # Convert to 0-1 confidence level
                "analysis": self._get_analysis(fake_score)
            }
            
            logger.info(f"Detection complete. Fake score: {fake_score:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return {"fake_score": 0.5, "error": str(e)}
    
    def _get_analysis(self, fake_score):
        """
        Provide a detailed analysis based on the fake score.
        
        Args:
            fake_score: Model prediction score
            
        Returns:
            str: Analysis text
        """
        if fake_score > 0.9:
            return "High confidence this is a deepfake"
        elif fake_score > 0.7:
            return "Likely a deepfake but with moderate confidence"
        elif fake_score > 0.5:
            return "Possibly a deepfake but with low confidence"
        elif fake_score > 0.3:
            return "Likely authentic but with low confidence"
        elif fake_score > 0.1:
            return "Likely authentic with moderate confidence"
        else:
            return "High confidence this is an authentic image"