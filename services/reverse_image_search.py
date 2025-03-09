import logging
import os
import re
import io
from datetime import datetime
from urllib.parse import urlparse
from PIL import Image
import aiohttp

logger = logging.getLogger(__name__)

class ReverseImageSearch:
    """Service for reverse image searching to find the earliest published version."""
    
    def __init__(self):
        self.api_key = os.getenv("RAPID_API_KEY", "your_rapidapi_key_here")
        self.api_url = "https://reverse-image-search1.p.rapidapi.com/reverse-image-search"
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': "reverse-image-search1.p.rapidapi.com"
        }
        logger.info("ReverseImageSearch initialized with RapidAPI URL: %s", self.api_url)
    
    async def search(self, img_url):
        """
        Perform reverse image search.
        
        Args:
            img_url: URL string of image to search
            
        Returns:
            dict: Results including earliest source, similar images, and reliability score
        """
        logger.info("Starting reverse image search")
        try:
            params = {
                'url': img_url,
                'limit': 100,  # Limit results to process
                'safe_search': 'off'
            }
            
            logger.info("Sending API request to %s", self.api_url)
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params, headers=self.headers) as response:
                    return await self._process_response(response)
        except Exception as e:
            logger.error("Reverse image search error: %s", str(e))
            return {
                "score": 0,
                "error": str(e)
            }
    
    async def _process_response(self, response):
        """Process API response."""
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
        
        # Check for API error response
        if result.get("status") == "ERROR":
            error = result.get("error", {})
            error_message = error.get("message", "Unknown error")
            logger.error("RapidAPI error: %s", error_message)
            return {
                "score": 0,
                "error": error_message
            }
        
        # Process results from the API response
        data = result.get("data", [])
        image_results = data
        logger.info("Found %d image results", len(image_results))

        # Extract sources with dates
        sources_with_dates = []
        for img_result in image_results:
            source_name = img_result.get("title")
            link = img_result.get("link")  # Key changed from 'url' to 'link'
            snippet = img_result.get("description", "")
            
            # Use domain extraction from URL instead of API-provided field
            domain = self._extract_domain(link) if link else ""

            # Check if API provides a direct date field first
            api_date = img_result.get("date")
            if api_date:
                date = api_date  # Use directly if available
            else:
                date = self._extract_date(snippet)  # Fallback to snippet parsing

            if date:
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
        
        # Build a list of (title, description) pairs instead of extracting keywords
        content_context = []
        for r in image_results:
            content_context.append({
                "title": r.get("title", ""),
                "description": r.get("description", "")
            })
        
        # Calculate a score based on the available sources
        source_count = len(sources_with_dates)
        score = 0
        if source_count > 0:
            score = 15
            logger.debug("Base score set to 15 due to available sources")
            
            if source_count > 1:
                bonus = min(source_count * 5, 20)
                score += bonus
                logger.debug("Added bonus for multiple sources: %d", bonus)
            
            if sources_with_dates:
                reliable_domains = [
                    "nytimes.com", "reuters.com", "apnews.com", 
                    "bbc.com", "washingtonpost.com", "theguardian.com"
                ]
                for domain in reliable_domains:
                    if domain in sources_with_dates[0].get("site", ""):
                        score += 15
                        logger.debug("Bonus added for reliable domain: %s", domain)
                        break
            
            score = min(score, 100)
            logger.info("Final score calculated: %d", score)
        
        # Return our final response dictionary
        return {
            "score": score,
            "earliest_source": sources_with_dates[0] if sources_with_dates else None,
            "all_sources": sources_with_dates,
            "source_count": source_count,
            "content_context": content_context,  # Entire content/context returned here
            "result_count": len(image_results)
        }
    
    def _extract_date(self, text):
        """Extract date from text using regex patterns."""
        if not text:
            return None
        logger.debug("Extracting date from text: %s", text)
        patterns = [
            r'(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{1,2}/\d{1,2}/\d{4})',
            r'(\d{1,2}\.\d{1,2}\.\d{4})'
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
        """Convert date string to a timestamp for comparison."""
        logger.debug("Converting date string to timestamp: %s", date_str)
        formats = [
            "%d %b %Y", "%b %d, %Y", "%b %d %Y", "%Y-%m-%d",
            "%m/%d/%Y", "%d/%m/%Y", "%d.%m.%Y"
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                timestamp = dt.timestamp()
                logger.debug("Date string %s converted to timestamp %f using format %s", 
                             date_str, timestamp, fmt)
                return timestamp
            except ValueError:
                continue
        logger.warning("Failed to parse date string: %s", date_str)
        return 0
    
    def _extract_domain(self, url):
        """Extract domain from a URL."""
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