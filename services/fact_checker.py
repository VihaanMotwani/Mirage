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
        logger.info("FactChecker initialized with API URL: %s", self.api_url)
    
    async def check(self, img, keywords: List[str]) -> Dict[str, Any]:
        """
        Check for fact-checking articles related to the image.
        
        Args:
            img: PIL Image object
            keywords: List of keywords extracted from the image analysis
            
        Returns:
            dict: Results including related fact-checks and reliability score
        """
        logger.info("Starting fact check process")
        try:
            if not keywords or len(keywords) < 2:
                logger.warning("Insufficient keywords for fact checking. Provided keywords: %s", keywords)
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "Insufficient keywords for fact checking"
                }
            
            # Construct search query from keywords
            search_keywords = keywords[:5]
            query = " ".join(search_keywords) + " fact check"
            logger.debug("Constructed query for general search: %s", query)
            
            # Search both for general fact-checks and specific site fact-checks
            general_results = await self._search_sonar(query)
            logger.info("General search returned %d results", len(general_results))
            
            fact_check_query = query + " site:" + " OR site:".join(self.fact_check_sites[:3])
            logger.debug("Constructed query for specific site search: %s", fact_check_query)
            specific_results = await self._search_sonar(fact_check_query)
            logger.info("Specific site search returned %d results", len(specific_results))
            
            # Combine and process results
            combined_results = self._merge_results(general_results, specific_results)
            logger.info("Combined results count: %d", len(combined_results))
            fact_checks = self._extract_fact_checks(combined_results)
            logger.info("Extracted %d fact-checks", len(fact_checks))
            
            # Calculate score based on the fact-checks found
            score = self._calculate_reliability_score(fact_checks)
            logger.info("Calculated reliability score: %f", score)
            
            return {
                "score": score,
                "related_fact_checks": fact_checks,
                "query_used": query,
                "raw_result_count": len(combined_results)
            }
            
        except Exception as e:
            logger.error("Fact checking error: %s", str(e))
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
        logger.info("Performing search with query: %s", query)
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
            logger.debug("Search payload: %s", payload)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        logger.error("Perplexity API error: %d", response.status)
                        return []
                    
                    result = await response.json()
                    logger.debug("Received search results: %s", result)
                    
                    return result.get("results", [])
                    
        except Exception as e:
            logger.error("Perplexity API search error: %s", str(e))
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
        logger.debug("Merging results from two sources")
        seen_urls = set()
        merged_results = []
        
        for result_set in [results1, results2]:
            for result in result_set:
                url = result.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged_results.append(result)
        
        logger.info("Merged results count: %d", len(merged_results))
        return merged_results
    
    def _extract_fact_checks(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract fact-check information from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            list: Extracted fact-checks with standardized format
        """
        logger.info("Extracting fact-checks from search results")
        fact_checks = []
        
        for result in search_results:
            try:
                url = result.get("url", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                
                # Skip if missing essential information
                if not (url and title and snippet):
                    logger.debug("Skipping result due to missing information: %s", result)
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
                    logger.debug("Result is not a fact-check: %s", url)
                    continue
                
                # Try to determine the rating
                rating = self._extract_rating(title, snippet)
                
                # Determine source reliability
                reliability = self._determine_source_reliability(domain)
                
                fact_check = {
                    "title": title,
                    "url": url,
                    "source": domain,
                    "description": snippet,
                    "rating": rating,
                    "reliability": reliability
                }
                fact_checks.append(fact_check)
                logger.debug("Extracted fact-check: %s", fact_check)
                
            except Exception as e:
                logger.error("Error extracting fact-check: %s", str(e))
                continue
        
        fact_checks.sort(key=lambda x: 
            0 if x["reliability"] == "high" else 
            1 if x["reliability"] == "medium" else 2
        )
        
        logger.info("Total fact-checks after extraction: %d", len(fact_checks))
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
        logger.debug("Extracting rating from title and snippet")
        text = f"{title.lower()} {snippet.lower()}"
        
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
        logger.debug("Determining reliability for domain: %s", domain)
        for tier, domains in self.reliability_tiers.items():
            if any(trusted_domain in domain for trusted_domain in domains):
                logger.debug("Domain %s determined as %s reliability", domain, tier)
                return tier
        
        logger.debug("Domain %s defaulted to low reliability", domain)
        return "low"
    
    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        
        Args:
            url: Full URL
            
        Returns:
            str: Domain name
        """
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
    
    def _calculate_reliability_score(self, fact_checks: List[Dict[str, Any]]) -> float:
        """
        Calculate overall reliability score based on found fact-checks.
        
        Args:
            fact_checks: List of extracted fact-checks
            
        Returns:
            float: Reliability score (0-100)
        """
        logger.info("Calculating reliability score based on fact-checks")
        if not fact_checks:
            logger.warning("No fact-checks found, returning neutral score of 50.0")
            return 50.0
        
        score = 50.0
        fact_check_count = len(fact_checks)
        logger.debug("Fact-check count: %d", fact_check_count)
        if fact_check_count > 1:
            additional_points = min(fact_check_count * 5, 15)
            score += additional_points
            logger.debug("Added %d points for multiple fact-checks", additional_points)
        
        high_reliability_count = sum(1 for fc in fact_checks if fc["reliability"] == "high")
        logger.debug("High reliability count: %d", high_reliability_count)
        if high_reliability_count > 0:
            additional_points = min(high_reliability_count * 10, 20)
            score += additional_points
            logger.debug("Added %d points for high reliability sources", additional_points)
        
        true_count = sum(1 for fc in fact_checks if fc["rating"] == "True")
        false_count = sum(1 for fc in fact_checks if fc["rating"] == "False")
        mixed_count = sum(1 for fc in fact_checks if fc["rating"] == "Partly false")
        logger.debug("Rating counts - True: %d, False: %d, Mixed: %d", true_count, false_count, mixed_count)
        
        if true_count > false_count + mixed_count:
            score += 15
            logger.debug("Consensus true detected, added 15 points")
        elif false_count > true_count + mixed_count:
            score -= 15
            logger.debug("Consensus false detected, subtracted 15 points")
        elif mixed_count > true_count + false_count:
            score -= 5
            logger.debug("Mixed results detected, subtracted 5 points")
        
        final_score = max(0, min(100, score))
        logger.info("Final reliability score: %f", final_score)
        return final_score