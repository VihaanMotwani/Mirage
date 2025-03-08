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
    
    async def search(self, img):
        """
        Perform reverse image search.
        
        Args:
            img: PIL Image object
            
        Returns:
            dict: Results including earliest source, similar images, and reliability score
        """
        try:
            # Convert image to bytes for upload
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format if img.format else 'JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create form data for the API request
            data = aiohttp.FormData()
            data.add_field('api_key', self.api_key)
            data.add_field('engine', 'google_reverse_image')
            data.add_field('image_file', img_byte_arr)
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, data=data) as response:
                    if response.status != 200:
                        return {
                            "score": 0,
                            "error": f"API error: {response.status}",
                            "message": await response.text()
                        }
                    
                    result = await response.json()
            
            # Process results
            image_results = result.get("image_results", [])
            
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
                    sources_with_dates.append({
                        "date": date,
                        "timestamp": self._date_to_timestamp(date),
                        "source": source_name,
                        "site": domain,
                        "link": link,
                        "snippet": snippet
                    })
            
            # Sort by date (oldest first)
            sources_with_dates.sort(key=lambda x: x.get("timestamp", 0))
            
            # Extract keywords from related text
            related_text = [
                result.get("title", ""),
                result.get("snippet", "")
            ]
            related_text.extend([r.get("snippet", "") for r in image_results[:5]])
            keywords = self._extract_keywords(" ".join(related_text))
            
            # Calculate score based on:
            # 1. Do we have any dated sources?
            # 2. Is the earliest source from a reliable domain?
            # 3. How many different sources found the image?
            
            source_count = len(sources_with_dates)
            score = 0
            
            if source_count > 0:
                # Base score for having sources
                score = 50
                
                # Bonus for multiple sources
                if source_count > 1:
                    score += min(source_count * 5, 20)  # Up to 20 points for 4+ sources
                
                # Bonus for reliable domains
                earliest_source = sources_with_dates[0]
                reliable_domains = ["nytimes.com", "reuters.com", "apnews.com", "bbc.com", 
                                  "washingtonpost.com", "theguardian.com"]
                
                for domain in reliable_domains:
                    if domain in earliest_source.get("site", ""):
                        score += 15
                        break
                
                # Cap at 100
                score = min(score, 100)
            
            return {
                "score": score,
                "earliest_source": sources_with_dates[0] if sources_with_dates else None,
                "all_sources": sources_with_dates,
                "source_count": source_count,
                "keywords": keywords,
                "result_count": len(image_results)
            }
            
        except Exception as e:
            logger.error(f"Reverse image search error: {str(e)}")
            return {
                "score": 0,
                "error": str(e)
            }
    
    def _extract_date(self, text):
        """Extract date from text using regex patterns."""
        if not text:
            return None
        
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
                return match.group(1)
        
        return None
    
    def _date_to_timestamp(self, date_str):
        """Convert date string to timestamp for comparison."""
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
                    return dt.timestamp()
                except ValueError:
                    continue
                
            return 0  # Default if parsing fails
        except Exception:
            return 0
    
    def _extract_domain(self, url):
        """Extract domain from URL."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return url
    
    def _extract_keywords(self, text):
        """Extract relevant keywords from text."""
        # Simple keyword extraction based on common words
        if not text:
            return []
        
        # Remove special chars and lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words and count
        words = text.split()
        word_count = {}
        
        for word in words:
            if len(word) > 3:  # Skip short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Skip common stopwords
        stopwords = ["the", "and", "that", "this", "with", "from", "have", "for", "not", "are", "were"]
        for word in stopwords:
            if word in word_count:
                del word_count[word]
        
        # Sort by frequency
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 10 keywords
        return [word for word, count in sorted_words[:10]]