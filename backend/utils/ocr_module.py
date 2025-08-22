"""
OCR Module Enhanced with Gemini Embeddings for maritime documents
Supports Azure Cognitive Services and Pytesseract with semantic understanding
"""

import os
import logging
import google.generativeai as genai
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OCRProcessor:
    """OCR processing enhanced with Gemini embeddings for maritime documents"""
    
    def __init__(self):
        """Initialize OCR processor with configuration and Gemini integration"""
        self.use_azure = bool(os.getenv("AZURE_COGNITIVE_KEY"))
        
        # Initialize Gemini for text enhancement
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key and gemini_api_key != "your-gemini-api-key-here":
            genai.configure(api_key=gemini_api_key)
            self.use_gemini = True
            logger.info("Gemini integration enabled for OCR enhancement")
        else:
            self.use_gemini = False
            logger.warning("Gemini API key not configured for OCR enhancement")
        
        if self.use_azure:
            try:
                from azure.cognitiveservices.vision.computervision import ComputerVisionClient
                from msrest.authentication import CognitiveServicesCredentials
                
                self.cv_client = ComputerVisionClient(
                    os.getenv("AZURE_COGNITIVE_ENDPOINT"),
                    CognitiveServicesCredentials(os.getenv("AZURE_COGNITIVE_KEY"))
                )
                logger.info("Azure Cognitive Services OCR initialized")
            except ImportError:
                logger.warning("Azure Cognitive Services not available, falling back to Pytesseract")
                self.use_azure = False
        
        # Configure Pytesseract (fallback)
        if not self.use_azure:
            # Set Tesseract path if needed (uncomment and modify for your system)
            # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
            logger.info("Pytesseract OCR initialized")
            
        # Maritime context for Gemini enhancement
        if self.use_gemini:
            self.maritime_context = "maritime shipping vessel port cargo operations statement of facts"
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=self.maritime_context,
                    task_type="semantic_similarity"
                )
                self.maritime_embedding = np.array(result['embedding'])
            except Exception as e:
                logger.warning(f"Failed to create maritime context embedding: {e}")
                self.use_gemini = False
    
    async def extract_text_from_image(self, image_path: Path) -> Optional[str]:
        """
        Extract text from image file using OCR with Gemini enhancement
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted and enhanced text content
        """
        try:
            # Extract raw text using OCR
            if self.use_azure:
                raw_text = await self._extract_with_azure(image_path)
            else:
                raw_text = self._extract_with_pytesseract(image_path)
            
            if not raw_text:
                return None
            
            # Enhance text with Gemini if available
            if self.use_gemini:
                enhanced_text = await self._enhance_text_with_gemini(raw_text)
                return enhanced_text if enhanced_text else raw_text
            
            return raw_text
                
        except Exception as e:
            logger.error(f"OCR text extraction failed: {str(e)}")
            return None
    
    async def _enhance_text_with_gemini(self, raw_text: str) -> Optional[str]:
        """
        Enhance OCR text using Gemini embeddings and context understanding
        
        Args:
            raw_text: Raw OCR text with potential errors
            
        Returns:
            Enhanced and corrected text
        """
        try:
            # Split text into chunks for processing
            chunks = [chunk.strip() for chunk in raw_text.split('\n') if chunk.strip()]
            maritime_chunks = []
            
            for chunk in chunks:
                if len(chunk) < 5:  # Skip very short chunks
                    continue
                
                # Get embedding for the chunk
                try:
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=chunk,
                        task_type="semantic_similarity"
                    )
                    chunk_embedding = np.array(result['embedding'])
                    
                    # Check similarity with maritime context
                    similarity = cosine_similarity([chunk_embedding], [self.maritime_embedding])[0][0]
                    
                    # Keep chunks that are relevant to maritime context
                    if similarity > 0.2:  # Lower threshold for OCR text
                        maritime_chunks.append(chunk)
                        
                except Exception as e:
                    logger.warning(f"Failed to process chunk: {e}")
                    # Include chunk anyway if embedding fails
                    maritime_chunks.append(chunk)
            
            # Combine relevant chunks
            enhanced_text = '\n'.join(maritime_chunks)
            
            # Use Gemini to correct common OCR errors in maritime context
            corrected_text = await self._correct_maritime_ocr_errors(enhanced_text)
            
            logger.info(f"Gemini OCR enhancement: {len(chunks)} -> {len(maritime_chunks)} relevant chunks")
            return corrected_text if corrected_text else enhanced_text
            
        except Exception as e:
            logger.warning(f"Gemini text enhancement failed: {e}")
            return raw_text
    
    async def _correct_maritime_ocr_errors(self, text: str) -> Optional[str]:
        """
        Use Gemini to correct common OCR errors in maritime documents
        
        Args:
            text: Text with potential OCR errors
            
        Returns:
            Corrected text
        """
        try:
            # Create a prompt for maritime OCR correction
            correction_prompt = f"""
            Please correct common OCR errors in this maritime document text while preserving the original meaning and structure. 
            Focus on correcting:
            - Ship/vessel names
            - Port names and locations
            - Maritime terminology
            - Dates and times
            - Technical terms
            
            Original text:
            {text[:2000]}  # Limit text length
            
            Return only the corrected text without additional commentary.
            """
            
            # Use Gemini for text correction
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(correction_prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Gemini OCR correction failed: {e}")
            return None
    
    async def _extract_with_azure(self, image_path: Path) -> Optional[str]:
        """
        Extract text using Azure Cognitive Services
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text content
        """
        try:
            # Read image file
            with open(image_path, 'rb') as image_stream:
                # Call Azure OCR API
                read_response = self.cv_client.read_in_stream(
                    image_stream, 
                    raw=True
                )
            
            # Get operation ID from response headers
            operation_id = read_response.headers["Operation-Location"].split("/")[-1]
            
            # Wait for operation to complete
            import time
            while True:
                read_result = self.cv_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)
            
            # Extract text from result
            text_content = []
            if read_result.status == 'succeeded':
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        text_content.append(line.text)
            
            full_text = '\n'.join(text_content)
            
            if not full_text.strip():
                logger.warning("No text found in image using Azure OCR")
                return None
            
            logger.info(f"Azure OCR extracted {len(full_text)} characters from image")
            return full_text
            
        except Exception as e:
            logger.error(f"Azure OCR extraction failed: {str(e)}")
            # Fallback to Pytesseract
            return self._extract_with_pytesseract(image_path)
    
    def _extract_with_pytesseract(self, image_path: Path) -> Optional[str]:
        """
        Extract text using Pytesseract
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text content
        """
        try:
            # Open image with PIL
            image = Image.open(image_path)
            
            # Preprocess image for better OCR results
            image = self._preprocess_image(image)
            
            # Extract text with Pytesseract
            # Use custom config for better maritime document recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:-/\()[]{}"\'' + "' "
            
            text = pytesseract.image_to_string(
                image, 
                config=custom_config,
                lang='eng'
            )
            
            if not text.strip():
                logger.warning("No text found in image using Pytesseract")
                return None
            
            logger.info(f"Pytesseract extracted {len(text)} characters from image")
            return text
            
        except Exception as e:
            logger.error(f"Pytesseract extraction failed: {str(e)}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Resize if image is too small
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Enhance contrast (simple threshold)
            # This helps with scanned documents
            import numpy as np
            img_array = np.array(image)
            threshold = np.mean(img_array)
            img_array = np.where(img_array > threshold, 255, 0)
            image = Image.fromarray(img_array.astype('uint8'))
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}")
            return image
    
    def get_supported_formats(self) -> list:
        """
        Get list of supported image formats
        
        Returns:
            List of supported file extensions
        """
        return ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']
    
    def is_image_readable(self, image_path: Path) -> bool:
        """
        Check if image file can be processed
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if image is readable, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                # Try to load the image
                img.load()
                return True
        except Exception:
            return False
