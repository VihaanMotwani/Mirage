# services/fact_checker.py
import aiohttp
import logging
import os
import json
from typing import List, Dict, Any, Optional
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class FactChecker:
    """Queries Perplexity Sonar API to find fact-checks relevant to the image."""
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY", "your_perplexity_api_key_here")
        self.api_url = "https://api.perplexity.ai/search"
        
        # List of known fact-checking sites
        self.fact_check_sites = [
            "factcheck.org",
            "politifact.com",
            "snopes.com",
            "apnews.com/hub/ap-fact-check",
            "reuters.com/fact-check",
            "checkyourfact.com",
            "factcheck.afp.com",
            "fullfact.org",
            "leadstories.com",
            "usatoday.com/fact-check"
        ]
        
        # Reliability tiers for domains
        self.reliability_tiers = {
            "high": [
                "reuters.com",
                "apnews.com",
                "bbc.com",
                "npr.org",
                "politifact.com",
                "factcheck.org",
                "snopes.com",
            ],
            "medium": [
                "nytimes.com",
                "washingtonpost.com",
                "cnn.com",
                "nbcnews.com",
                "abcnews.go.com",
                "theguardian.com",
                "usatoday.com",
            ],
            "low": []  # Will be determined dynamically for less known sources
        }
    
    async def check(self, img, keywords: List[str]) -> Dict[str, Any]:
        """
        Check for fact-checking articles related to the image.
        
        Args:
            img: PIL Image object
            keywords: List of keywords extracted from the image analysis
            
        Returns:
            dict: Results including related fact-checks and reliability score
        """
        try:
            if not keywords or len(keywords) < 2:
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "Insufficient keywords for fact checking"
                }
            
            # Construct search query from keywords
            # Take the top 5 keywords to keep the query focused
            search_keywords = keywords[:5]
            
            # Add "fact check" to the query to bias toward fact-checking results
            query = " ".join(search_keywords) + " fact check"
            
            # Search both for general fact-checks and specific site fact-checks
            general_results = await self._search_sonar(query)
            
            # Additional search specifically for fact-checking sites
            fact_check_query = query + " site:" + " OR site:".join(self.fact_check_sites[:3])
            specific_results = await self._search_sonar(fact_check_query)
            
            # Combine and process results
            combined_results = self._merge_results(general_results, specific_results)
            fact_checks = self._extract_fact_checks(combined_results)
            
            # Calculate score based on the fact-checks found
            score = self._calculate_reliability_score(fact_checks)
            
            return {
                "score": score,
                "related_fact_checks": fact_checks,
                "query_used": query,
                "raw_result_count": len(combined_results)
            }
            
        except Exception as e:
            logger.error(f"Fact checking error: {str(e)}")
            return {
                "score": 50,  # Neutral score on error
                "error": str(e),
                "related_fact_checks": []
            }
    
    async def _search_sonar(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Perplexity Sonar API.
        
        Args:
            query: Search query string
            
        Returns:
            list: Search results
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "query": query,
                "max_results": 10,  # Limit to top 10 results
                "search_mode": "internet_search"  # Use internet search mode
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        logger.error(f"Perplexity API error: {response.status}")
                        return []
                    
                    result = await response.json()
                    
                    # Extract and return the search results
                    return result.get("results", [])
                    
        except Exception as e:
            logger.error(f"Perplexity API search error: {str(e)}")
            return []
    
    def _merge_results(self, results1: List[Dict[str, Any]], results2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge results from multiple searches, removing duplicates.
        
        Args:
            results1: First set of search results
            results2: Second set of search results
            
        Returns:
            list: Merged search results
        """
        seen_urls = set()
        merged_results = []
        
        for result_set in [results1, results2]:
            for result in result_set:
                url = result.get("url", "")
                
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged_results.append(result)
        
        return merged_results
    
    def _extract_fact_checks(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract fact-check information from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            list: Extracted fact-checks with standardized format
        """
        fact_checks = []
        
        for result in search_results:
            try:
                url = result.get("url", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                # Skip if missing essential information
                if not (url and title and snippet):
                    continue
                
                # Parse domain for reliability assessment
                domain = self._extract_domain(url)
                
                # Determine if this is likely a fact-check
                is_fact_check = (
                    any(site in domain for site in self.fact_check_sites) or
                    "fact check" in title.lower() or
                    "fact check" in snippet.lower() or
                    "fact-check" in url.lower()
                )
                
                if not is_fact_check:
                    continue
                
                # Try to determine the rating
                rating = self._extract_rating(title, snippet)
                
                # Determine source reliability
                reliability = self._determine_source_reliability(domain)
                
                fact_checks.append({
                    "title": title,
                    "url": url,
                    "source": domain,
                    "description": snippet,
                    "rating": rating,
                    "reliability": reliability
                })
                
            except Exception as e:
                logger.error(f"Error extracting fact-check: {str(e)}")
                continue
        
        # Sort by reliability (high to low)
        fact_checks.sort(key=lambda x: 
            0 if x["reliability"] == "high" else 
            1 if x["reliability"] == "medium" else 2
        )
        
        return fact_checks
    
    def _extract_rating(self, title: str, snippet: str) -> str:
        """
        Extract fact-check rating from title or snippet.
        
        Args:
            title: Article title
            snippet: Article snippet
            
        Returns:
            str: Extracted rating or "Unrated"
        """
        # Combine text for analysis
        text = f"{title.lower()} {snippet.lower()}"
        
        # Look for common rating patterns
        if re.search(r'\bfalse\b|\bfake\b|\bmisinformation\b|\bhoax\b', text):
            return "False"
        elif re.search(r'\btrue\b|\baccurate\b|\bcorrect\b|\blegitimate\b', text):
            return "True"
        elif re.search(r'\bmisleading\b|\bpartly false\b|\bpartially\s+true\b|\bmixed\b', text):
            return "Partly false"
        elif re.search(r'\bunsubstantiated\b|\bunverified\b|\bunproven\b', text):
            return "Unverified"
        elif re.search(r'\boutdated\b|\bold\b|\bnot current\b', text):
            return "Outdated"
        
        return "Unrated"
    
    def _determine_source_reliability(self, domain: str) -> str:
        """
        Determine the reliability of a source based on domain.
        
        Args:
            domain: Website domain
            
        Returns:
            str: Reliability tier ('high', 'medium', or 'low')
        """
        for tier, domains in self.reliability_tiers.items():
            if any(trusted_domain in domain for trusted_domain in domains):
                return tier
        
        # Default to low for unknown sources
        return "low"
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: Full URL
            
        Returns:
            str: Domain name
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Remove 'www.' if present
            if domain.startswith('www.'):
                domain = domain[4:]
                
            return domain
        except Exception:
            return url
    
    def _calculate_reliability_score(self, fact_checks: List[Dict[str, Any]]) -> float:
        """
        Calculate overall reliability score based on found fact-checks.
        
        Args:
            fact_checks: List of extracted fact-checks
            
        Returns:
            float: Reliability score (0-100)
        """
        if not fact_checks:
            return 50.0  # Neutral score if no fact-checks found
        
        # Initialize starting score
        score = 50.0
        
        # Points for having multiple fact-checks
        fact_check_count = len(fact_checks)
        if fact_check_count > 1:
            score += min(fact_check_count * 5, 15)  # Up to 15 points for 3+ fact-checks
        
        # Points for high-reliability sources
        high_reliability_count = sum(1 for fc in fact_checks if fc["reliability"] == "high")
        if high_reliability_count > 0:
            score += min(high_reliability_count * 10, 20)  # Up to 20 points
        
        # Adjust based on ratings
        true_count = sum(1 for fc in fact_checks if fc["rating"] == "True")
        false_count = sum(1 for fc in fact_checks if fc["rating"] == "False")
        mixed_count = sum(1 for fc in fact_checks if fc["rating"] == "Partly false")
        
        # If clear consensus on true or false, adjust score accordingly
        if true_count > false_count + mixed_count:
            score += 15
        elif false_count > true_count + mixed_count:
            score -= 15
        elif mixed_count > true_count + false_count:
            score -= 5  # Small penalty for mixed results
        
        # Cap the score
        return max(0, min(100, score))