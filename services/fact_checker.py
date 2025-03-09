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
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key or self.api_key == "your_perplexity_api_key_here":
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")
        
        self.api_url = "https://api.perplexity.ai/chat/completions"
        
        # Reliability tiers for domains (you can keep or modify these as needed)
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
            "low": []
        }
        logger.info("FactChecker initialized with API URL: %s", self.api_url)

    async def check(self, content_context: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Check for fact-checking articles related to the provided content (title/description pairs).

        Args:
            content_context: List of dictionaries, each containing "title" and "description"
        
        Returns:
            dict: Results including related fact-checks and a reliability score
        """
        logger.info("Starting fact check process")
        try:
            # Combine the text from all title/description pairs
            if not content_context:
                logger.warning("No content context provided.")
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "No content context provided"
                }
            
            combined_text = []
            for item in content_context:
                t = item.get("title", "").strip()
                d = item.get("description", "").strip()
                combined_text.append(t)
                combined_text.append(d)
            
            full_text = " ".join(combined_text)
            if len(full_text) < 10:
                logger.warning("Insufficient text for fact checking. Combined text < 10 characters.")
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "Insufficient text for fact checking"
                }
            
            query = f"{full_text} fact check"
            
            logger.debug("Constructed query: %s", query)
            search_results = await self._search_sonar(query)
            logger.info("Search returned %d results", len(search_results))
            
            fact_checks = self._extract_fact_checks(search_results)
            logger.info("Extracted %d fact-checks", len(fact_checks))
            
            score = self._calculate_reliability_score(fact_checks)
            logger.info("Calculated reliability score: %f", score)
            
            return {
                "score": score,
                "related_fact_checks": fact_checks,
                "query_used": query,
                "raw_result_count": len(search_results)
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
        Search using Perplexity Sonar API, requesting reputable and reliable sources.

        Args:
            query: Search query string
            
        Returns:
            list: Search results (citations) from Perplexity
        """
        logger.info("Performing search with query: %s", query)
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "accept": "application/json"
            }

            # The first message is the system prompt. The second is from the user.
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant who can use web retrieval to provide fact-checks, references, and citations. "
                            "Your goals:"
                            "1. Provide relevant sources (e.g., articles, fact-checking pages, news outlets, research papers) whenever possible."
                            "2. Strive for reliability and accuracy, but do not artificially exclude sources unless they are overtly unreliable. "
                            "3. If unsure about the credibility of a source, include a short disclaimer explaining your uncertainty (e.g., 'This source has limited track record')."
                            "4. If you find recognized fact-checking organizations (PolitiFact, Snopes, FactCheck.org, etc.), highlight them. "
                            "5. If you cannot find well-known fact-checks, you may provide other relevant sources but explain that they might not be official or fully verified."
                            "6. Offer any context that helps evaluate the credibility of a source or article. "
                            "7. If no relevant fact-check is found, say so explicitly and encourage further research."

                            "Please note:"
                            "- Return all relevant citations in your response, so users can follow them if they want more information."
                            "- Maintain clarity and transparency about what is known and unknown."
                            "- Avoid speculation; focus on the best-available evidence or disclaim if evidence is incomplete."
                            "- When possible, summarize the key points from the cited sources in a concise way."
                        )
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status != 200:
                        error_body = await response.text()
                        logger.error("Perplexity API error: %d - %s", response.status, error_body)
                        return []
                    
                    result = await response.json()
                    logger.debug("Received API response: %s", json.dumps(result, indent=2))
                    
                    # Extract citations from the first choice's message
                    citations = []
                    if result.get("choices"):
                        for choice in result["choices"]:
                            citations.extend(choice.get("message", {}).get("citations", []))
                    
                    logger.info("Extracted %d citations", len(citations))
                    return citations
                    
        except Exception as e:
            logger.error("Perplexity API search error: %s", str(e))
            return []

    def _extract_fact_checks(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract fact-check information from search results.

        Args:
            search_results: List of search results (citations)
            
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
                
                # Skip if missing essential info
                if not (url and title and snippet):
                    logger.debug("Skipping result due to missing information: %s", result)
                    continue
                
                domain = self._extract_domain(url)
                
                # Determine if this is likely a fact-check.
                # We no longer check a restricted list of domains,
                # only the presence of "fact check" or "fact-check" in text or URL.
                text_lower = f"{title.lower()} {snippet.lower()} {url.lower()}"
                is_fact_check = (
                    "fact check" in text_lower or
                    "fact-check" in text_lower or
                    "debunk" in text_lower
                )
                
                if not is_fact_check:
                    logger.debug("Result is not a clear fact-check: %s", url)
                    continue
                
                rating = self._extract_rating(title, snippet)
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
        
        # Sort so higher reliability appears first
        fact_checks.sort(key=lambda x: 
            0 if x["reliability"] == "high" else 
            1 if x["reliability"] == "medium" else 2
        )
        
        logger.info("Total fact-checks after extraction: %d", len(fact_checks))
        return fact_checks

    def _extract_rating(self, title: str, snippet: str) -> str:
        """
        Try to detect the rating from the text.
        """
        text = f"{title.lower()} {snippet.lower()}"
        rating_patterns = {
            "False": r'\b(false|fake|hoax|misinformation|debunked)\b',
            "True": r'\b(true|accurate|correct|legitimate|verified)\b',
            "Partly false": r'\b(partially false|half-truth|misleading|mixed|exaggerat|out of context)\b',
            "Unverified": r'\b(unverified|unsubstantiated|unproven|disputed|questioned)\b',
            "Outdated": r'\b(outdated|old news|no longer true|superseded)\b'
        }
        
        for rating, pattern in rating_patterns.items():
            if re.search(pattern, text):
                return rating
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
        
        # Add points if multiple fact-checks
        if fact_check_count > 1:
            additional_points = min(fact_check_count * 5, 15)
            score += additional_points
            logger.debug("Added %d points for multiple fact-checks", additional_points)
        
        # Add points if we have high reliability sources
        high_reliability_count = sum(1 for fc in fact_checks if fc["reliability"] == "high")
        logger.debug("High reliability count: %d", high_reliability_count)
        if high_reliability_count > 0:
            additional_points = min(high_reliability_count * 10, 20)
            score += additional_points
            logger.debug("Added %d points for high reliability sources", additional_points)
        
        # Adjust score based on rating prevalence
        true_count = sum(1 for fc in fact_checks if fc["rating"] == "True")
        false_count = sum(1 for fc in fact_checks if fc["rating"] == "False")
        mixed_count = sum(1 for fc in fact_checks if fc["rating"] == "Partly false")
        
        logger.debug("Rating counts - True: %d, False: %d, Mixed: %d", true_count, false_count, mixed_count)
        
        if true_count > false_count + mixed_count:
            score += 15
            logger.debug("Consensus 'True' detected, added 15 points")
        elif false_count > true_count + mixed_count:
            score -= 15
            logger.debug("Consensus 'False' detected, subtracted 15 points")
        elif mixed_count > true_count + false_count:
            score -= 5
            logger.debug("Mixed results detected, subtracted 5 points")
        
        final_score = max(0, min(100, score))
        logger.info("Final reliability score: %f", final_score)
        return final_score