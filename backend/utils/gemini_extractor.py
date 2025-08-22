"""
Gemini Embedding-Enhanced Maritime Event Extractor
Uses Google's Gemini embeddings for superior semantic understanding
"""

import google.generativeai as genai
import numpy as np
import logging
import os
import re
import json
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from datetime import datetime
import spacy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiEmbeddingExtractor:
    """Maritime event extractor using Google Gemini embeddings"""
    
    def __init__(self):
        """Initialize the Gemini Event Extractor."""
        load_dotenv()
        
        try:
            # Check if API key is available and valid
            google_api_key = os.getenv('GOOGLE_API_KEY')
            if not google_api_key or google_api_key in ['your-google-api-key-here', 'gemini-api-here']:
                raise ValueError("Google API key not properly configured")
                
            # Don't actually configure genai here to avoid network calls during init
            self.google_api_key = google_api_key
            self.gemini_available = True
            print("✅ Gemini Embedding Extractor initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize Gemini API: {e}")
            print("📝 Note: Gemini extraction will be disabled for this session")
            self.gemini_available = False
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Maritime event types for extraction
        self.event_types = [
            'grounding', 'collision', 'fire', 'explosion', 'machinery_failure',
            'cargo_shift', 'structural_damage', 'oil_spill', 'man_overboard',
            'navigation_error', 'weather_damage', 'port_state_control'
        ]
        
        # Define maritime event types with descriptions
        self.maritime_event_types = {
            'arrival': ['vessel arrival', 'ship berthing', 'docking operations', 'pilot boarding'],
            'departure': ['vessel departure', 'ship leaving', 'unberthing operations', 'pilot disembarking'],
            'cargo_operations': ['cargo loading', 'cargo discharge', 'container handling', 'loading operations'],
            'port_operations': ['tug assistance', 'mooring operations', 'port clearance', 'customs clearance'],
            'navigation': ['navigation incident', 'course change', 'speed alteration', 'position reporting'],
            'weather': ['weather condition', 'wind report', 'sea state', 'visibility condition'],
            'incident': ['marine incident', 'collision', 'grounding', 'fire', 'machinery failure']
        }
        
        # Initialize event embeddings
        if self.gemini_available:
            self._compute_event_type_embeddings()
    
    def _get_gemini_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding from Gemini model
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            if not self.gemini_available:
                # Return zero vector when Gemini is not available
                logger.warning("Gemini not available, returning zero vector")
                return np.zeros(768)
                
            # Use Gemini embedding model
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="semantic_similarity"
            )
            
            return np.array(result['embedding'])
            
        except Exception as e:
            logger.warning(f"Gemini embedding failed: {e}, returning zero vector")
            return np.zeros(768)
    
    def _compute_event_type_embeddings(self):
        """Pre-compute Gemini embeddings for maritime event types"""
        self.event_embeddings = {}
        
        for event_type, descriptions in self.maritime_event_types.items():
            try:
                embeddings = []
                for desc in descriptions:
                    embedding = self._get_gemini_embedding(desc)
                    embeddings.append(embedding)
                
                # Use mean embedding as the event type representation
                self.event_embeddings[event_type] = np.mean(embeddings, axis=0)
                logger.info(f"Computed embedding for {event_type}")
                
            except Exception as e:
                logger.warning(f"Failed to compute embedding for {event_type}: {e}")
                # Fallback to random embedding
                self.event_embeddings[event_type] = np.random.rand(768)
    
    async def extract_events_with_gemini(self, text: str) -> List[Dict]:
        """
        Extract maritime events using Gemini embeddings
        
        Args:
            text: Raw document text
            
        Returns:
            List of extracted events with high accuracy
        """
        try:
            logger.info("Starting Gemini-powered event extraction")
            
            # Step 1: Sentence segmentation and preprocessing
            relevant_sentences = self._extract_relevant_sentences_gemini(text)
            
            # Step 2: Semantic event classification using Gemini embeddings
            classified_events = self._classify_events_with_gemini(relevant_sentences)
            
            # Step 3: Extract temporal and location information
            enriched_events = self._extract_temporal_spatial_info(classified_events, text)
            
            # Step 4: Event clustering to remove duplicates using Gemini embeddings
            deduplicated_events = self._cluster_similar_events_gemini(enriched_events)
            
            # Step 5: Final validation and formatting
            formatted_events = self._format_and_validate_events(deduplicated_events)
            
            logger.info(f"Gemini extraction completed: {len(formatted_events)} events found")
            return formatted_events
            
        except Exception as e:
            logger.error(f"Gemini event extraction failed: {e}")
            return self._get_fallback_events(text)
    
    def _extract_relevant_sentences_gemini(self, text: str) -> List[Dict]:
        """
        Extract maritime-relevant sentences using Gemini embeddings
        
        Args:
            text: Raw document text
            
        Returns:
            List of relevant sentences with similarity scores
        """
        try:
            # Sentence segmentation
            if self.nlp:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
            else:
                # Fallback sentence splitting
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
            
            if not sentences:
                return []
            
            # Create maritime context embedding using Gemini
            maritime_context = "maritime port operations vessel movements cargo handling shipping procedures"
            context_embedding = self._get_gemini_embedding(maritime_context)
            
            # Get embeddings for all sentences
            relevant_sentences = []
            
            for i, sentence in enumerate(sentences[:50]):  # Limit to first 50 sentences
                try:
                    sentence_embedding = self._get_gemini_embedding(sentence)
                    
                    # Calculate similarity with maritime context
                    similarity = cosine_similarity([context_embedding], [sentence_embedding])[0][0]
                    
                    # Filter sentences above threshold
                    if similarity > 0.25:  # Lower threshold for more inclusive extraction
                        relevant_sentences.append({
                            "text": sentence,
                            "similarity": float(similarity),
                            "index": i
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to process sentence {i}: {e}")
                    continue
            
            # Sort by relevance and return top sentences
            relevant_sentences.sort(key=lambda x: x["similarity"], reverse=True)
            return relevant_sentences[:25]  # Keep top 25 most relevant
            
        except Exception as e:
            logger.error(f"Sentence extraction failed: {e}")
            return []
    
    def _classify_events_with_gemini(self, sentences: List[Dict]) -> List[Dict]:
        """
        Classify sentences into maritime event types using Gemini embeddings
        
        Args:
            sentences: List of relevant sentences
            
        Returns:
            List of classified events with confidence scores
        """
        classified_events = []
        
        for sentence_data in sentences:
            try:
                sentence = sentence_data["text"]
                sentence_embedding = self._get_gemini_embedding(sentence)
                
                # Calculate similarity with each event type
                best_event_type = None
                best_similarity = 0.0
                
                for event_type, type_embedding in self.event_embeddings.items():
                    similarity = cosine_similarity([sentence_embedding], [type_embedding])[0][0]
                    
                    if similarity > best_similarity and similarity > 0.3:  # Confidence threshold
                        best_similarity = similarity
                        best_event_type = event_type
                
                if best_event_type:
                    classified_events.append({
                        "text": sentence,
                        "event_type": best_event_type,
                        "confidence": float(best_similarity),
                        "original_similarity": sentence_data["similarity"]
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to classify sentence: {e}")
                continue
        
        logger.info(f"Classified {len(classified_events)} events using Gemini")
        return classified_events
    
    def _extract_temporal_spatial_info(self, events: List[Dict], full_text: str) -> List[Dict]:
        """
        Extract temporal and spatial information using NER and regex patterns
        
        Args:
            events: Classified events
            full_text: Full document text for context
            
        Returns:
            Events enriched with time and location data
        """
        enriched_events = []
        
        # Enhanced temporal patterns for maritime documents
        time_patterns = [
            r'(\d{1,2}:\d{2})\s*(?:hrs?|hours?|UTC|GMT)?',
            r'(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})',
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:at\s+)?(\d{1,2}:\d{2})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\s+(\d{1,2}:\d{2})'
        ]
        
        # Enhanced location patterns for maritime contexts
        location_patterns = [
            r'(?:at|in|from|to|off)\s+([A-Z][a-zA-Z\s]+(?:Port|Bay|Harbor|Harbour|Terminal|Berth|Anchorage|Wharf))',
            r'([A-Z][a-zA-Z\s]+(?:Port|Bay|Harbor|Harbour|Terminal|Berth|Anchorage|Wharf))',
            r'berth\s+([A-Z0-9]+)',
            r'anchorage\s+([A-Z][a-zA-Z\s]*)',
            r'([A-Z][a-zA-Z\s]*)\s+(?:port|harbor|harbour|bay)',
            r'(\d+°\d+\'[NS])\s+(\d+°\d+\'[EW])',  # Coordinates
        ]
        
        for event in events:
            text = event["text"]
            
            # Extract temporal information
            start_time = None
            end_time = None
            
            for pattern in time_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for i, match in enumerate(matches):
                    time_str = match.group(0).strip()
                    if not start_time:
                        start_time = time_str
                    elif not end_time and i > 0:
                        end_time = time_str
            
            # Extract location information
            location = None
            for pattern in location_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if match.lastindex and match.lastindex >= 1:
                        location = match.group(1).strip()
                    else:
                        location = match.group(0).strip()
                    break
            
            # Use spaCy NER for additional location extraction
            if not location and self.nlp:
                try:
                    doc = self.nlp(text)
                    for ent in doc.ents:
                        if ent.label_ in ["GPE", "LOC", "FAC"]:  # Geopolitical, Location, Facility
                            location = ent.text
                            break
                except Exception as e:
                    logger.warning(f"spaCy NER failed: {e}")
            
            # Create enriched event
            enriched_event = {
                "event": event["event_type"],
                "text": text,
                "start": start_time,
                "end": end_time,
                "location": location,
                "confidence": event["confidence"],
                "description": text[:150] + "..." if len(text) > 150 else text,
                "source": "gemini_embedding"
            }
            
            enriched_events.append(enriched_event)
        
        return enriched_events
    
    def _cluster_similar_events_gemini(self, events: List[Dict]) -> List[Dict]:
        """
        Cluster similar events using Gemini embeddings to remove duplicates
        
        Args:
            events: List of events to deduplicate
            
        Returns:
            Deduplicated events
        """
        if len(events) <= 1:
            return events
        
        try:
            # Create Gemini embeddings for event descriptions
            embeddings = []
            for event in events:
                embedding = self._get_gemini_embedding(event["text"])
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            
            # Use DBSCAN clustering with cosine distance
            clustering = DBSCAN(eps=0.25, min_samples=1, metric='cosine')
            clusters = clustering.fit_predict(embeddings)
            
            # Keep the highest confidence event from each cluster
            cluster_events = {}
            for i, (event, cluster_id) in enumerate(zip(events, clusters)):
                if cluster_id == -1:  # Noise point, keep as individual event
                    cluster_events[f"noise_{i}"] = event
                elif cluster_id not in cluster_events:
                    cluster_events[cluster_id] = event
                else:
                    # Keep event with higher confidence
                    if event["confidence"] > cluster_events[cluster_id]["confidence"]:
                        cluster_events[cluster_id] = event
            
            deduplicated = list(cluster_events.values())
            logger.info(f"Clustered {len(events)} events into {len(deduplicated)} unique events")
            return deduplicated
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, returning original events")
            return events
    
    def _format_and_validate_events(self, events: List[Dict]) -> List[Dict]:
        """
        Final formatting and validation of extracted events
        
        Args:
            events: Raw events to format
            
        Returns:
            Formatted and validated events
        """
        formatted_events = []
        
        for event in events:
            try:
                formatted_event = {
                    "event": event["event"].replace("_", " ").title(),
                    "start": self._normalize_datetime(event.get("start")),
                    "end": self._normalize_datetime(event.get("end")) if event.get("end") else None,
                    "location": event.get("location"),
                    "description": event.get("description"),
                    "confidence": round(event.get("confidence", 0.0), 3),
                    "source": "gemini_enhanced"
                }
                
                # Only include events with reasonable confidence
                if formatted_event["confidence"] > 0.3:
                    formatted_events.append(formatted_event)
                    
            except Exception as e:
                logger.warning(f"Failed to format event: {e}")
                continue
        
        # Sort by confidence and start time
        formatted_events.sort(key=lambda x: (-x["confidence"], x["start"] or ""))
        return formatted_events
    
    def _normalize_datetime(self, dt_str: Optional[str]) -> Optional[str]:
        """Normalize datetime strings to consistent format"""
        if not dt_str or dt_str.lower() == 'null':
            return None
        
        # Enhanced datetime patterns for maritime documents
        patterns = [
            r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})',  # ISO format
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s+(\d{1,2}:\d{2})',  # DD/MM/YYYY HH:MM
            r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\s+(\d{1,2}:\d{2})',  # DD Mon YYYY HH:MM
            r'(\d{1,2}:\d{2})',  # Just time
        ]
        
        dt_str = dt_str.strip()
        
        for pattern in patterns:
            match = re.search(pattern, dt_str, re.IGNORECASE)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 2:  # ISO format
                        return f"{groups[0]} {groups[1]}"
                    elif len(groups) == 4 and groups[1].isalpha():  # DD Mon YYYY HH:MM
                        day, month, year, time = groups
                        month_num = {
                            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                        }.get(month.lower()[:3], '01')
                        return f"{year}-{month_num}-{day.zfill(2)} {time}"
                    elif len(groups) == 4:  # DD/MM/YYYY HH:MM
                        day, month, year, time = groups
                        if len(year) == 2:
                            year = f"20{year}"
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)} {time}"
                    elif len(groups) == 1:  # Just time
                        return f"2025-08-20 {groups[0]}"  # Default date
                except Exception:
                    continue
        
        return dt_str
    
    def _get_fallback_events(self, text: str) -> List[Dict]:
        """
        Fallback method when Gemini extraction fails
        
        Args:
            text: Document text
            
        Returns:
            List of fallback events (empty if no valid events found)
        """
        logger.info("Using fallback event extraction")
        
        # Simple regex-based fallback - only return events if they have valid timestamps/locations
        fallback_events = []
        
        # Look for maritime terms with actual timestamps
        timestamp_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(?:at\s+)?(\d{1,2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2})\s+(\d{1,2}:\d{2})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\s+(\d{1,2}:\d{2})'
        ]
        
        maritime_terms = {
            "arrival": r"arrived?|berthing|docking|made fast",
            "departure": r"departed|left|sailed|unberthed", 
            "loading": r"loading|commenced|cargo handling",
            "pilot": r"pilot.*(?:board|embark|aboard)"
        }
        
        # Only create events if we find actual timestamps in the text
        has_timestamps = any(re.search(pattern, text, re.IGNORECASE) for pattern in timestamp_patterns)
        
        if not has_timestamps:
            logger.info("No timestamps found in document - returning empty event list")
            return []
        
        for event_type, pattern in maritime_terms.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Try to find a timestamp near this event
                event_start = match.start()
                event_end = match.end()
                
                # Look for timestamp within 100 characters of the event
                text_window = text[max(0, event_start-100):event_end+100]
                
                timestamp_match = None
                for ts_pattern in timestamp_patterns:
                    timestamp_match = re.search(ts_pattern, text_window, re.IGNORECASE)
                    if timestamp_match:
                        break
                
                if timestamp_match:
                    fallback_events.append({
                        "event": event_type.title(),
                        "start": f"{timestamp_match.group(1)} {timestamp_match.group(2)}",
                        "end": None,
                        "location": "Port",
                        "description": match.group(0),
                        "confidence": 0.5,
                        "source": "fallback"
                    })
        
        return fallback_events[:5]  # Limit fallback events
