import aiohttp
import logging
import os
import re
import io
from datetime import datetime
from urllib.parse import urlparse
from PIL import Image

logger = logging.getLogger(__name__)

class ReverseImageSearch:
    """Service for reverse image searching to find the earliest published version."""
    
    def __init__(self):
        self.api_key = os.getenv("SERP_API_KEY", "your_serp_api_key_here")
        self.api_url = "https://serpapi.com/search"
        # Cloudinary settings – update these with your Cloudinary account details.
        self.cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "your_cloud_name")
        self.upload_preset = os.getenv("CLOUDINARY_UPLOAD_PRESET", "your_unsigned_upload_preset")
        logger.info("ReverseImageSearch initialized with SERP API URL: %s", self.api_url)
    
    async def search(self, img):
        """
        Perform reverse image search.
        
        Args:
            img: PIL Image object or URL string
            
        Returns:
            dict: Results including earliest source, similar images, and reliability score
        """
        logger.info("Starting reverse image search")
        try:
            params = {
                'api_key': self.api_key,
                'engine': 'google_reverse_image'
            }
            if isinstance(img, str):
                # Use the provided URL directly.
                params['image_url'] = img
                logger.debug("Using URL-based search with image: %s", img)
            else:
                # Upload the PIL image to Cloudinary to obtain a public URL.
                hosted_url = await self.upload_and_get_url(img)
                if not hosted_url:
                    raise ValueError("Failed to upload image to Cloudinary.")
                params['image_url'] = hosted_url
                logger.debug("Using externally hosted image URL: %s", hosted_url)
            
            logger.info("Sending API request to %s", self.api_url)
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params) as response:
                    return await self._process_response(response)
        except Exception as e:
            logger.error("Reverse image search error: %s", str(e))
            return {
                "score": 0,
                "error": str(e)
            }
    
    async def upload_and_get_url(self, img):
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
            
            upload_url = f"https://api.cloudinary.com/v1_1/{self.cloud_name}/upload"
            
            data = aiohttp.FormData()
            data.add_field("file",
                           img_bytes,
                           filename="image." + img_format.lower(),
                           content_type="image/" + img_format.lower())
            data.add_field("upload_preset", self.upload_preset)
            
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
        
        # Process results
        image_results = result.get("image_results", [])
        logger.info("Found %d image results", len(image_results))
        
        # Extract sources with dates
        sources_with_dates = []
        for img_result in image_results:
            source_name = img_result.get("source")
            link = img_result.get("link")
            snippet = img_result.get("snippet")
            
            # Try to extract a date from snippet or link.
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
        
        # Calculate a score based on the available sources.
        source_count = len(sources_with_dates)
        score = 0
        if source_count > 0:
            score = 50
            logger.debug("Base score set to 50 due to available sources")
            
            if source_count > 1:
                bonus = min(source_count * 5, 20)
                score += bonus
                logger.debug("Added bonus for multiple sources: %d", bonus)
            
            if sources_with_dates:
                reliable_domains = ["nytimes.com", "reuters.com", "apnews.com", "bbc.com", 
                                    "washingtonpost.com", "theguardian.com"]
                for domain in reliable_domains:
                    if domain in sources_with_dates[0].get("site", ""):
                        score += 15
                        logger.debug("Bonus added for reliable domain: %s", domain)
                        break
            
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
                logger.debug("Date string %s converted to timestamp %f using format %s", date_str, timestamp, fmt)
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
    
    def _extract_keywords(self, text):
        """Extract relevant keywords from text."""
        logger.debug("Extracting keywords from text")
        if not text:
            return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        word_count = {}
        for word in words:
            if len(word) > 3:
                word_count[word] = word_count.get(word, 0) + 1
        stopwords = ["the", "and", "that", "this", "with", "from", "have", "for", "not", "are", "were"]
        for word in stopwords:
            word_count.pop(word, None)
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:10]]
        logger.debug("Keywords extracted: %s", keywords)
        return keywords
