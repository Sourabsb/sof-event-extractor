"""
GPT Event Extractor Module Enhanced with Gemini Embeddings
Uses OpenAI GPT for intelligent event extraction and Gemini embeddings for semantic understanding
"""

import openai
import google.generativeai as genai
import json
import logging
import os
import re
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import spacy
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GPTExtractor:
    """GPT-powered event extraction enhanced with Gemini embeddings"""
    
    def __init__(self):
        """Initialize GPT extractor with API configuration and Gemini embeddings"""
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"  # Use GPT-4 if available
        
        # Configure Gemini for embeddings
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key and gemini_api_key != "your-gemini-api-key-here" and gemini_api_key != "gpt-key-here":
            genai.configure(api_key=gemini_api_key)
            self.use_gemini = True
        else:
            self.use_gemini = False
            logger.warning("Gemini API key not configured, using basic extraction")
        
        # Load spaCy model for preprocessing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Maritime event embeddings for semantic filtering
        if self.use_gemini:
            self._initialize_gemini_embeddings()
    
    def _initialize_gemini_embeddings(self):
        """Initialize maritime-specific embeddings using Gemini."""
        try:
            # Check if API key is available and valid
            if not os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_API_KEY') == 'your-google-api-key-here':
                raise ValueError("Google API key not properly configured")
            
            maritime_terms = [
                "grounding", "collision", "fire", "explosion", "machinery failure",
                "cargo shift", "structural damage", "oil spill", "man overboard",
                "navigation error", "weather damage", "port state control",
                "vessel coordinates", "latitude longitude", "nautical miles",
                "knots speed", "heading bearing", "maritime safety",
                "international waters", "coastal waters", "harbor entry",
                "anchorage", "berth", "pilot boarding", "tugs assistance"
            ]
            
            # Create embeddings for maritime terms with timeout
            result = genai.embed_content(
                model="models/embedding-001",
                content=maritime_terms
            )
            
            self.maritime_embeddings = result['embedding']
            print("✅ Maritime embeddings initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Gemini embeddings: {e}")
            print("📝 Note: Gemini embeddings will be disabled for this session")
            print("🔧 System will use GPT-only extraction method")
            self.gemini_available = False
            self.maritime_embeddings = None
    
    def _get_gemini_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get Gemini embedding for text"""
        if not self.use_gemini:
            return None
            
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="semantic_similarity"
            )
            return np.array(result['embedding'])
        except Exception as e:
            logger.warning(f"Gemini embedding failed: {e}")
            return None
            
        # Maritime event embeddings for semantic filtering
        if self.use_gemini:
            self._initialize_gemini_embeddings()
    
    async def extract_events(self, text: str) -> List[Dict]:
        """
        Extract port events from document text using GPT and Gemini embeddings
        
        Args:
            text: Raw text content from document
            
        Returns:
            List of extracted events with timestamps
        """
        try:
            # Preprocess text with Gemini-enhanced filtering
            if self.use_gemini and self.nlp:
                text = self._preprocess_text_with_gemini(text)
            elif self.nlp:
                text = self._preprocess_text(text)
            
            # Try regex-based extraction first (fast fallback)
            regex_events = self._extract_events_regex(text)
            
            # Use GPT for intelligent extraction
            gpt_events = await self._extract_events_gpt(text)
            
            # Merge and deduplicate events using Gemini embeddings if available
            if self.use_gemini:
                all_events = self._merge_events_with_gemini(regex_events, gpt_events)
            else:
                all_events = self._merge_events(regex_events, gpt_events)
            
            # Validate and format events
            formatted_events = self._format_events(all_events)
            
            logger.info(f"Extracted {len(formatted_events)} events from document")
            return formatted_events
            
        except Exception as e:
            logger.error(f"Event extraction failed: {str(e)}")
            # Return empty list instead of failing completely
            return []
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text using spaCy for better extraction
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text[:1000000])  # Limit text length for spaCy
            
            # Extract sentences related to port activities
            relevant_sentences = []
            port_keywords = ["port", "berth", "dock", "anchor", "arrival", "departure", 
                           "loading", "discharge", "pilot", "tug", "customs", "clearance"]
            
            for sent in doc.sents:
                sent_lower = sent.text.lower()
                if any(keyword in sent_lower for keyword in port_keywords):
                    relevant_sentences.append(sent.text.strip())
            
            return '\n'.join(relevant_sentences) if relevant_sentences else text
            
        except Exception as e:
            logger.warning(f"Text preprocessing failed: {str(e)}")
            return text
    
    def _preprocess_text_with_gemini(self, text: str) -> str:
        """
        Enhanced text preprocessing using Gemini embeddings for maritime relevance
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text with enhanced maritime focus
        """
        try:
            # Process text with spaCy
            doc = self.nlp(text[:1000000])  # Limit text length for spaCy
            
            # Extract sentences and get their embeddings
            relevant_sentences = []
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                if len(sent_text) < 10:
                    continue
                
                # Get Gemini embedding for sentence
                sent_embedding = self._get_gemini_embedding(sent_text)
                if sent_embedding is None:
                    # Fallback to keyword matching
                    port_keywords = ["port", "berth", "dock", "anchor", "arrival", "departure", 
                                   "loading", "discharge", "pilot", "tug", "customs", "clearance"]
                    sent_lower = sent_text.lower()
                    if any(keyword in sent_lower for keyword in port_keywords):
                        relevant_sentences.append(sent_text)
                    continue
                
                # Check similarity with maritime contexts
                max_similarity = 0.0
                for maritime_embedding in self.maritime_embeddings:
                    similarity = cosine_similarity([sent_embedding], [maritime_embedding])[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                # Include sentences with high maritime relevance
                if max_similarity > 0.3:  # Threshold for maritime relevance
                    relevant_sentences.append(sent_text)
            
            enhanced_text = '\n'.join(relevant_sentences) if relevant_sentences else text
            logger.info(f"Gemini preprocessing: {len(relevant_sentences)} relevant sentences from {len(list(doc.sents))} total")
            return enhanced_text
            
        except Exception as e:
            logger.warning(f"Gemini preprocessing failed: {str(e)}")
            return self._preprocess_text(text)  # Fallback to basic preprocessing
    
    def _extract_events_regex(self, text: str) -> List[Dict]:
        """
        Extract events using regex patterns (fast fallback)
        
        Args:
            text: Document text content
            
        Returns:
            List of events found using regex
        """
        events = []
        
        # Enhanced maritime event patterns
        patterns = [
            # Arrival/Departure patterns with more variations
            (r'(?:arrived?|berthed?|docked?)\s+(?:at\s+)?([^,\n]*?)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'arrival'),
            (r'(?:departed?|left|sailed|unberthed?)\s+(?:from\s+)?([^,\n]*?)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'departure'),
            
            # Time and event patterns (more flexible)
            (r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})\s*-?\s*(.*?(?:arrived?|berthed?|pilot|cargo|loading|discharge|departed?|left|mooring|anchor)[^.\n]*)', 'timed_event'),
            
            # Loading/Discharging patterns
            (r'(?:commenced|started|began)\s+(?:loading|discharg\w*|cargo)\s+(?:operations?\s+)?(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'cargo_start'),
            (r'(?:completed|finished|ended)\s+(?:loading|discharg\w*|cargo)\s+(?:operations?\s+)?(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'cargo_end'),
            
            # Pilot operations
            (r'pilot\s+(?:boarded?|embarked?|aboard)\s+(?:vessel\s+)?(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'pilot_boarding'),
            (r'pilot\s+(?:disembarked?|left)\s+(?:vessel\s+)?(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'pilot_departure'),
            
            # Mooring and anchoring
            (r'(?:mooring|all fast)\s+(?:operations?\s+)?(?:completed?)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'mooring_complete'),
            (r'(?:anchor|anchored?)\s+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\s+(?:at\s+)?(\d{1,2}:\d{2})', 'anchoring'),
        ]
        
        for pattern, event_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    if event_type == 'timed_event':
                        # For timed events, extract the description
                        date_str = groups[0]
                        time_str = groups[1]
                        description = groups[2].strip() if len(groups) > 2 else ""
                        
                        event = {
                            "event": self._extract_event_type_from_description(description),
                            "start": f"{date_str} {time_str}",
                            "end": None,
                            "location": self._extract_location_from_description(description),
                            "description": description,
                            "source": "regex"
                        }
                    else:
                        # For other patterns
                        if len(groups) >= 3:
                            location = groups[0] if groups[0] and len(groups[0].strip()) > 0 else None
                            date_str = groups[1]
                            time_str = groups[2]
                        else:
                            location = None
                            date_str = groups[0]
                            time_str = groups[1]
                        
                        event = {
                            "event": event_type.replace('_', ' ').title(),
                            "start": f"{date_str} {time_str}",
                            "end": None,
                            "location": location,
                            "source": "regex"
                        }
                    
                    events.append(event)
        
        logger.info(f"Regex extraction found {len(events)} events")
        return events
    
    def _extract_event_type_from_description(self, description: str) -> str:
        """Extract event type from description text"""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['arrived', 'berthed', 'docked', 'moored']):
            return 'Arrival'
        elif any(word in description_lower for word in ['departed', 'left', 'sailed', 'unberthed']):
            return 'Departure'
        elif any(word in description_lower for word in ['loading', 'cargo', 'discharge']):
            return 'Cargo Operations'
        elif any(word in description_lower for word in ['pilot', 'boarded', 'embarked']):
            return 'Pilot Operations'
        elif any(word in description_lower for word in ['anchor', 'anchored']):
            return 'Anchoring'
        elif any(word in description_lower for word in ['mooring', 'all fast']):
            return 'Mooring'
        else:
            return 'Port Operation'
    
    def _extract_location_from_description(self, description: str) -> str:
        """Extract location information from description"""
        # Look for location patterns in the description
        location_patterns = [
            r'(?:at|in|from|to)\s+([A-Z][a-zA-Z\s]+(?:Port|Terminal|Berth|Bay|Harbor|Harbour))',
            r'berth\s+([A-Z0-9]+)',
            r'terminal\s+([A-Z0-9\s]+)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    async def _extract_events_gpt(self, text: str) -> List[Dict]:
        """
        Extract events using GPT API
        
        Args:
            text: Document text content
            
        Returns:
            List of events extracted by GPT
        """
        try:
            # Check if using demo key - still try to process with fallback patterns
            api_key = os.getenv("OPENAI_API_KEY", "")
            if api_key == "sk-demo-key-for-testing" or not api_key:
                logger.info("Demo mode detected - using pattern-based extraction")
                return self._extract_events_regex(text)  # Use regex instead of empty list
            
            # Prepare the prompt for GPT
            prompt = self._create_extraction_prompt(text)
            
            # Call GPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an AI maritime document parser specialized in extracting port events from Statement of Facts documents."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=2000
            )
            
            # Parse GPT response
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                events = json.loads(response_text)
                if isinstance(events, list):
                    return events
                elif isinstance(events, dict) and "events" in events:
                    return events["events"]
                else:
                    logger.warning("Unexpected GPT response format")
                    return []
            except json.JSONDecodeError:
                logger.warning("GPT response is not valid JSON")
                return self._parse_gpt_text_response(response_text)
                
        except Exception as e:
            logger.error(f"GPT extraction failed: {str(e)}")
            # Fallback to regex extraction instead of empty list
            return self._extract_events_regex(text)
    
    def _create_extraction_prompt(self, text: str) -> str:
        """
        Create optimized prompt for GPT event extraction
        
        Args:
            text: Document text content
            
        Returns:
            Formatted prompt string
        """
        # Truncate text if too long (GPT token limits)
        if len(text) > 8000:
            text = text[:8000] + "... [truncated]"
        
        prompt = f"""
You are an AI maritime document parser. Input is a raw text of a port Statement of Facts. 

Extract each port event in this exact JSON format:
{{
  "event": "event_description",
  "start": "YYYY-MM-DD HH:MM" or "DD/MM/YYYY HH:MM",
  "end": "YYYY-MM-DD HH:MM" or null if not provided,
  "location": "port/berth name" or null,
  "description": "additional details" or null
}}

Focus on these types of events:
- Vessel arrival/departure
- Berthing/unberthing
- Loading/discharging operations
- Pilot boarding/disembarking
- Port clearances
- Anchor dropping/weighing
- Tug assistance
- Customs/immigration

Ignore irrelevant headers, signatures, and administrative text.
Output as a valid JSON array only, no additional text.

Document text:
{text}
"""
        return prompt
    
    def _parse_gpt_text_response(self, response_text: str) -> List[Dict]:
        """
        Parse GPT text response when JSON parsing fails
        
        Args:
            response_text: Raw GPT response
            
        Returns:
            List of parsed events
        """
        events = []
        lines = response_text.strip().split('\n')
        
        current_event = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for event patterns in text response
            if line.startswith('Event:') or line.startswith('event:'):
                if current_event:
                    events.append(current_event)
                current_event = {"event": line.split(':', 1)[1].strip()}
            elif line.startswith('Start:') or line.startswith('start:'):
                current_event["start"] = line.split(':', 1)[1].strip()
            elif line.startswith('End:') or line.startswith('end:'):
                end_time = line.split(':', 1)[1].strip()
                current_event["end"] = end_time if end_time.lower() != 'null' else None
        
        if current_event:
            events.append(current_event)
        
        return events
    
    def _merge_events(self, regex_events: List[Dict], gpt_events: List[Dict]) -> List[Dict]:
        """
        Merge events from different extraction methods
        
        Args:
            regex_events: Events from regex extraction
            gpt_events: Events from GPT extraction
            
        Returns:
            Merged and deduplicated events
        """
        # Prioritize GPT events, add regex events that don't conflict
        all_events = gpt_events.copy()
        
        for regex_event in regex_events:
            # Simple deduplication based on event type and start time
            is_duplicate = any(
                event.get("event", "").lower() == regex_event.get("event", "").lower() and
                event.get("start", "") == regex_event.get("start", "")
                for event in all_events
            )
            
            if not is_duplicate:
                all_events.append(regex_event)
        
        return all_events
    
    def _merge_events_with_gemini(self, regex_events: List[Dict], gpt_events: List[Dict]) -> List[Dict]:
        """
        Merge events from different extraction methods using Gemini embeddings for deduplication
        
        Args:
            regex_events: Events from regex extraction
            gpt_events: Events from GPT extraction
            
        Returns:
            Merged and deduplicated events using semantic similarity
        """
        try:
            all_events = gpt_events.copy()
            
            # For each regex event, check semantic similarity with existing events
            for regex_event in regex_events:
                regex_text = f"{regex_event.get('event', '')} {regex_event.get('start', '')}"
                regex_embedding = self._get_gemini_embedding(regex_text)
                
                if regex_embedding is None:
                    # Fallback to simple deduplication
                    is_duplicate = any(
                        event.get("event", "").lower() == regex_event.get("event", "").lower() and
                        event.get("start", "") == regex_event.get("start", "")
                        for event in all_events
                    )
                    if not is_duplicate:
                        all_events.append(regex_event)
                    continue
                
                # Check semantic similarity with existing events
                is_duplicate = False
                for existing_event in all_events:
                    existing_text = f"{existing_event.get('event', '')} {existing_event.get('start', '')}"
                    existing_embedding = self._get_gemini_embedding(existing_text)
                    
                    if existing_embedding is not None:
                        similarity = cosine_similarity([regex_embedding], [existing_embedding])[0][0]
                        if similarity > 0.7:  # High similarity threshold for duplicates
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    all_events.append(regex_event)
            
            logger.info(f"Gemini merging: {len(all_events)} events after deduplication")
            return all_events
            
        except Exception as e:
            logger.warning(f"Gemini event merging failed: {e}")
            return self._merge_events(regex_events, gpt_events)  # Fallback
    
    def _format_events(self, events: List[Dict]) -> List[Dict]:
        """
        Format and validate extracted events
        
        Args:
            events: Raw extracted events
            
        Returns:
            Formatted and validated events
        """
        formatted_events = []
        
        for event in events:
            try:
                # Ensure required fields
                formatted_event = {
                    "event": event.get("event", "Unknown Event").strip(),
                    "start": self._normalize_datetime(event.get("start")),
                    "end": self._normalize_datetime(event.get("end")) if event.get("end") else None,
                    "location": event.get("location", "").strip() if event.get("location") else None,
                    "description": event.get("description", "").strip() if event.get("description") else None
                }
                
                # Only include events with valid start time
                if formatted_event["start"]:
                    formatted_events.append(formatted_event)
                    
            except Exception as e:
                logger.warning(f"Failed to format event: {str(e)}")
                continue
        
        # Sort events by start time
        formatted_events.sort(key=lambda x: x["start"] or "")
        
        return formatted_events
    
    def _normalize_datetime(self, dt_str: Optional[str]) -> Optional[str]:
        """
        Normalize datetime strings to consistent format
        
        Args:
            dt_str: Raw datetime string
            
        Returns:
            Normalized datetime string or None
        """
        if not dt_str or dt_str.lower() == 'null':
            return None
        
        # Common datetime patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})',
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s+(\d{1,2}:\d{2})',
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
        ]
        
        dt_str = dt_str.strip()
        
        for pattern in patterns:
            match = re.search(pattern, dt_str)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 2:  # ISO format
                        return f"{groups[0]} {groups[1]}"
                    elif len(groups) == 4:  # DD/MM/YYYY HH:MM
                        day, month, year, time = groups
                        if len(year) == 2:
                            year = f"20{year}"
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)} {time}"
                    elif len(groups) == 3:  # DD/MM/YYYY
                        day, month, year = groups
                        if len(year) == 2:
                            year = f"20{year}"
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except Exception:
                    continue
        
        return dt_str  # Return original if no pattern matches
