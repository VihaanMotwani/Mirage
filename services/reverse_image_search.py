import aiohttp
import logging
import json
import os
from datetime import datetime
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ReverseImageSearch:
    """Service for reverse image searching to find the earliest published version."""
    
    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY", "your_serp_api_key_here")
        self.api_url = "https://serpapi.com/search.json"
        logger.info("ReverseImageSearch initialized with API URL: %s", self.api_url)
    
    async def search(self, img):
        """
        Perform reverse image search.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Results including earliest source, similar images, and reliability score
        """
        logger.info("Starting reverse image search")
        try:
            # Convert image to bytes for upload
            import io
            logger.debug("Converting image to bytes")
            img_byte_arr = io.BytesIO()
            img_format = img.format if img.format else 'JPEG'
            img.save(img_byte_arr, format=img_format)
            img_byte_arr = img_byte_arr.getvalue()
            logger.debug("Image converted to bytes (format: %s, size: %d bytes)", img_format, len(img_byte_arr))
            
            # Create form data for the API request
            data = aiohttp.FormData()
            data.add_field('api_key', self.api_key)
            data.add_field('engine', 'google_reverse_image')
            data.add_field('image_file', img_byte_arr)
            logger.debug("Form data created for API request")
            
            # Make API request
            logger.info("Sending API request to %s", self.api_url)
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=data) as response:
                    if response.status != 200:
                        error_message = await response.text()
                        logger.error("API error: %d, message: %s", response.status, error_message)
                        return {
                            "score": 0,
                            "error": f"API error: {response.status}",
                            "message": error_message
                        }
                    result = await response.json()
                    logger.info("API request successful")
            
            # Process results
            image_results = result.get("image_results", [])
            logger.info("Found %d image results", len(image_results))
            
            # Extract sources with dates
            sources_with_dates = []
            for img_result in image_results:
                source_name = img_result.get("source")
                link = img_result.get("link")
                snippet = img_result.get("snippet")
                
                # Try to extract date from snippet or link
                date = self._extract_date(snippet)
                if date:
                    domain = self._extract_domain(link)
                    timestamp = self._date_to_timestamp(date)
                    sources_with_dates.append({
                        "date": date,
                        "timestamp": timestamp,
                        "source": source_name,
                        "site": domain,
                        "link": link,
                        "snippet": snippet
                    })
                    logger.debug("Extracted date %s from snippet; domain: %s", date, domain)
            
            logger.info("Extracted %d sources with dates", len(sources_with_dates))
            # Sort by date (oldest first)
            sources_with_dates.sort(key=lambda x: x.get("timestamp", 0))
            if sources_with_dates:
                logger.info("Earliest source date: %s", sources_with_dates[0].get("date"))
            
            # Extract keywords from related text
            related_text = [
                result.get("title", ""),
                result.get("snippet", "")
            ]
            related_text.extend([r.get("snippet", "") for r in image_results[:5]])
            logger.debug("Aggregated related text for keyword extraction")
            keywords = self._extract_keywords(" ".join(related_text))
            logger.info("Extracted keywords: %s", keywords)
            
            # Calculate score based on:
            # 1. Do we have any dated sources?
            # 2. Is the earliest source from a reliable domain?
            # 3. How many different sources found the image?
            source_count = len(sources_with_dates)
            score = 0
            if source_count > 0:
                # Base score for having sources
                score = 50
                logger.debug("Base score set to 50 due to available sources")
                
                # Bonus for multiple sources
                if source_count > 1:
                    bonus = min(source_count * 5, 20)
                    score += bonus
                    logger.debug("Added bonus for multiple sources: %d", bonus)
                
                # Bonus for reliable domains
                earliest_source = sources_with_dates[0]
                reliable_domains = ["nytimes.com", "reuters.com", "apnews.com", "bbc.com", 
                                    "washingtonpost.com", "theguardian.com"]
                for domain in reliable_domains:
                    if domain in earliest_source.get("site", ""):
                        score += 15
                        logger.debug("Bonus added for reliable domain: %s", domain)
                        break
                
                # Cap score at 100
                score = min(score, 100)
                logger.info("Final score calculated: %d", score)
            
            return {
                "score": score,
                "earliest_source": sources_with_dates[0] if sources_with_dates else None,
                "all_sources": sources_with_dates,
                "source_count": source_count,
                "keywords": keywords,
                "result_count": len(image_results)
            }
            
        except Exception as e:
            logger.error("Reverse image search error: %s", str(e))
            return {
                "score": 0,
                "error": str(e)
            }
    
    def _extract_date(self, text):
        """Extract date from text using regex patterns."""
        if not text:
            return None
        logger.debug("Extracting date from text: %s", text)
        # Various date formats to try
        patterns = [
            r'(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4})',  # 15 Jan 2023
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4})',  # Jan 15, 2023
            r'(\d{4}-\d{2}-\d{2})',  # 2023-01-15
            r'(\d{1,2}/\d{1,2}/\d{4})',  # 1/15/2023 or 15/1/2023
            r'(\d{1,2}\.\d{1,2}\.\d{4})'  # 15.01.2023
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                extracted_date = match.group(1)
                logger.debug("Date extracted using pattern '%s': %s", pattern, extracted_date)
                return extracted_date
        
        logger.debug("No date found in text")
        return None
    
    def _date_to_timestamp(self, date_str):
        """Convert date string to timestamp for comparison."""
        logger.debug("Converting date string to timestamp: %s", date_str)
        try:
            # Try different date formats
            formats = [
                "%d %b %Y",  # 15 Jan 2023
                "%b %d, %Y",  # Jan 15, 2023
                "%b %d %Y",   # Jan 15 2023
                "%Y-%m-%d",   # 2023-01-15
                "%m/%d/%Y",   # 1/15/2023
                "%d/%m/%Y",   # 15/1/2023
                "%d.%m.%Y"    # 15.01.2023
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    timestamp = dt.timestamp()
                    logger.debug("Date string %s converted to timestamp %f using format %s", date_str, timestamp, fmt)
                    return timestamp
                except ValueError:
                    continue
            logger.warning("Failed to parse date string: %s", date_str)
            return 0  # Default if parsing fails
        except Exception as e:
            logger.error("Error converting date to timestamp: %s", str(e))
            return 0
    
    def _extract_domain(self, url):
        """Extract domain from URL."""
        logger.debug("Extracting domain from URL: %s", url)
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            logger.debug("Extracted domain: %s", domain)
            return domain
        except Exception as e:
            logger.error("Error extracting domain: %s", str(e))
            return url
    
    def _extract_keywords(self, text):
        """Extract relevant keywords from text."""
        logger.debug("Extracting keywords from text")
        if not text:
            return []
        
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        word_count = {}
        
        for word in words:
            if len(word) > 3:  # Skip short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Remove common stopwords
        stopwords = ["the", "and", "that", "this", "with", "from", "have", "for", "not", "are", "were"]
        for word in stopwords:
            if word in word_count:
                del word_count[word]
        
        # Sort words by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:10]]
        logger.debug("Keywords extracted: %s", keywords)
        return keywords