"""
PDF Parser Module
Handles extraction of text content from PDF documents
"""

import PyPDF2
from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class PDFParser:
    """PDF document text extraction utility"""
    
    def __init__(self):
        """Initialize PDF parser"""
        pass
    
    def extract_text(self, file_path: Path) -> Optional[str]:
        """
        Extract text content from PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text)
                            logger.info(f"Extracted text from page {page_num + 1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                        continue
                
                # Combine all text
                full_text = '\n'.join(text_content)
                
                if not full_text.strip():
                    logger.warning("No text content found in PDF")
                    return None
                
                logger.info(f"Successfully extracted {len(full_text)} characters from PDF")
                return full_text
                
        except Exception as e:
            logger.error(f"PDF text extraction failed: {str(e)}")
            return None
    
    def get_pdf_info(self, file_path: Path) -> dict:
        """
        Get PDF document information
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF metadata
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                info = {
                    "num_pages": len(pdf_reader.pages),
                    "encrypted": pdf_reader.is_encrypted,
                    "metadata": {}
                }
                
                # Extract metadata if available
                if pdf_reader.metadata:
                    for key, value in pdf_reader.metadata.items():
                        info["metadata"][key] = str(value) if value else None
                
                return info
                
        except Exception as e:
            logger.error(f"Failed to get PDF info: {str(e)}")
            return {"error": str(e)}
    
    def is_pdf_searchable(self, file_path: Path) -> bool:
        """
        Check if PDF contains searchable text
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            True if PDF has extractable text, False otherwise
        """
        try:
            text = self.extract_text(file_path)
            return text is not None and len(text.strip()) > 10
        except Exception:
            return False
