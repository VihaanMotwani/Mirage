import aiohttp
import logging
import os
import json
from typing import List, Dict, Any, Optional
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

try:
    import spacy
except ImportError:
    spacy = None

try:
    from nltk.corpus import wordnet as wn
    import nltk
except ImportError:
    wn = None
    nltk = None

logger = logging.getLogger(__name__)

class FactChecker:
    """Enhanced fact-checking with NLP validation, query expansion, and confidence scoring"""
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key or self.api_key == "your_perplexity_api_key_here":
            raise ValueError("PERPLEXITY_API_KEY environment variable not set")
        
        self.api_url = "https://api.perplexity.ai/chat/completions"
        
        # Initialize NLP components
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"spaCy model load failed: {str(e)}")
        
        # Initialize synonym capabilities
        self.synonym_cache = {}
        if nltk and wn:
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                logger.warning("NLTK WordNet corpus not found. Synonym expansion disabled.")
                wn = None
        
        # Fact-checking site configuration
        self.fact_check_sites = [
            "factcheck.org", "politifact.com", "snopes.com",
            "apnews.com/hub/ap-fact-check", "reuters.com/fact-check",
            "checkyourfact.com", "factcheck.afp.com", "fullfact.org",
            "leadstories.com", "usatoday.com/fact-check", "boomlive.in",
            "dpa-factchecking.com", "factcrescendo.com", "logically.ai"
        ]
        
        self.reliability_tiers = {
            "high": [
                "reuters.com", "apnews.com", "bbc.com", "npr.org",
                "politifact.com", "factcheck.org", "snopes.com",
            ],
            "medium": [
                "nytimes.com", "washingtonpost.com", "cnn.com",
                "nbcnews.com", "abcnews.go.com", "theguardian.com",
                "usatoday.com",
            ],
            "low": []
        }
        logger.info("FactChecker initialized with NLP and synonym capabilities")

    async def check(self, img, keywords: List[str]) -> Dict[str, Any]:
        """Enhanced check with validation, expansion, and confidence scoring"""
        logger.info("Starting enhanced fact check process")
        try:
            # Query validation
            validation_result = self._validate_query(keywords)
            if not validation_result["valid"]:
                logger.warning(validation_result["message"])
                return {
                    "score": max(0, 50 - int((1 - validation_result["confidence"]) * 30)),
                    "related_fact_checks": [],
                    "message": validation_result["message"],
                    "confidence": validation_result["confidence"],
                    "validation_passed": False
                }
            
            # Query expansion
            expanded_keywords = self._expand_query_with_synonyms(keywords)
            search_keywords = (expanded_keywords + keywords)[:7]  # Prioritize original keywords
            query = " ".join(search_keywords) + " fact check"
            logger.debug(f"Expanded query: {query}")
            
            # Perform searches
            general_results = await self._search_sonar(query)
            fact_check_query = query + " site:" + " OR site:".join(self.fact_check_sites[:3])
            specific_results = await self._search_sonar(fact_check_query)
            
            # Process results
            combined_results = self._merge_results(general_results, specific_results)
            fact_checks = self._extract_fact_checks(combined_results, keywords)
            
            # Calculate enhanced score
            score = self._calculate_enhanced_score(fact_checks)
            
            return {
                "score": score,
                "related_fact_checks": fact_checks,
                "query_used": query,
                "raw_result_count": len(combined_results),
                "validation_confidence": validation_result["confidence"],
                "expanded_keywords": expanded_keywords
            }
            
        except Exception as e:
            logger.error(f"Fact checking error: {str(e)}")
            return {
                "score": 50,
                "error": str(e),
                "related_fact_checks": []
            }

    def _validate_query(self, keywords: List[str]) -> Dict[str, Any]:
        """Validate query using NLP techniques"""
        validation_result = {
            "valid": False,
            "message": "Invalid query",
            "confidence": 0.0
        }
        
        if not keywords:
            return validation_result
        
        query_text = " ".join(keywords)
        
        # Rule-based validation
        if len(keywords) < 2:
            validation_result["message"] = "Insufficient keywords for validation"
            return validation_result
        
        # SpaCy-based validation if available
        if self.nlp:
            doc = self.nlp(query_text)
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            nouns = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]
            
            if len(verbs) > 0 and len(nouns) > 0:
                validation_result.update({
                    "valid": True,
                    "message": "Valid query structure (contains verbs and nouns)",
                    "confidence": min(0.9, 0.3 + len(verbs)*0.1 + len(nouns)*0.1)
                })
            else:
                validation_result["message"] = "Query lacks meaningful structure"
        else:
            # Fallback regex validation
            question_pattern = r'\b(who|what|when|where|why|how|is|are|does|did)\b'
            claim_pattern = r'\b(claim|says?|alleged?|reported|stated)\b'
            
            if re.search(question_pattern, query_text, re.I) or re.search(claim_pattern, query_text, re.I):
                validation_result.update({
                    "valid": True,
                    "message": "Contains question or claim indicators",
                    "confidence": 0.7
                })
            else:
                validation_result["message"] = "No clear question or claim detected"
        
        return validation_result

    def _expand_query_with_synonyms(self, keywords: List[str]) -> List[str]:
        """Expand query using WordNet synonyms"""
        expanded = []
        if not wn:
            return expanded
        
        for keyword in keywords:
            if keyword.lower() in self.synonym_cache:
                expanded.extend(self.synonym_cache[keyword.lower()])
                continue
            
            synonyms = set()
            for syn in wn.synsets(keyword):
                for lemma in syn.lemmas():
                    name = lemma.name().replace('_', ' ')
                    if name.lower() != keyword.lower():
                        synonyms.add(name)
                        if len(synonyms) >= 2:
                            break
                if len(synonyms) >= 2:
                    break
            
            # Cache and add to results
            self.synonym_cache[keyword.lower()] = list(synonyms)[:2]
            expanded.extend(self.synonym_cache[keyword.lower()])
        
        return expanded

    def _extract_fact_checks(self, search_results: List[Dict[str, Any]], 
                           original_keywords: List[str]) -> List[Dict[str, Any]]:
        """Extract fact-checks with match confidence scoring"""
        fact_checks = []
        original_lower = [k.lower() for k in original_keywords]
        
        for result in search_results:
            try:
                url = result.get("url", "")
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()
                domain = self._extract_domain(url)
                
                # Calculate keyword matches
                title_matches = sum(1 for kw in original_lower if kw in title)
                snippet_matches = sum(1 for kw in original_lower if kw in snippet)
                total_matches = title_matches + snippet_matches
                match_confidence = min(1.0, total_matches / len(original_lower)) if original_lower else 0.0
                
                fact_check = {
                    "title": result.get("title", ""),
                    "url": url,
                    "source": domain,
                    "description": result.get("snippet", ""),
                    "rating": self._extract_rating(title, snippet),
                    "reliability": self._determine_source_reliability(domain),
                    "match_confidence": round(match_confidence, 2),
                    "keywords_matched": total_matches
                }
                
                if any(site in domain for site in self.fact_check_sites):
                    fact_checks.append(fact_check)
                
            except Exception as e:
                logger.error(f"Error extracting fact-check: {str(e)}")
        
        # Sort by confidence and reliability
        fact_checks.sort(key=lambda x: (
            -x["match_confidence"],
            0 if x["reliability"] == "high" else 1 if x["reliability"] == "medium" else 2
        ))
        return fact_checks

    def _calculate_enhanced_score(self, fact_checks: List[Dict[str, Any]]) -> float:
        """Calculate score with confidence weighting"""
        if not fact_checks:
            return 50.0
        
        base_score = 50.0
        weights = {
            "high": 1.2,
            "medium": 1.0,
            "low": 0.7
        }
        
        # Weighted reliability scoring
        reliability_points = sum(
            (10 * weights[fc["reliability"]] * fc["match_confidence"])
            for fc in fact_checks
        )
        
        # Rating analysis with confidence
        rating_scores = {
            "True": 0.0,
            "False": 0.0,
            "Partly false": 0.0
        }
        
        for fc in fact_checks:
            rating = fc["rating"]
            if rating in rating_scores:
                rating_scores[rating] += 5 * fc["match_confidence"]
        
        # Apply rating adjustments
        if rating_scores["True"] > sum(rating_scores.values()) * 0.6:
            base_score += 15
        elif rating_scores["False"] > sum(rating_scores.values()) * 0.6:
            base_score -= 15
        elif rating_scores["Partly false"] > sum(rating_scores.values()) * 0.4:
            base_score -= 5
        
        # Combine components
        final_score = base_score + min(reliability_points, 30)
        return max(0, min(100, round(final_score, 1)))
    
    async def _search_sonar(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using Perplexity Sonar API.
        
        Args:
            query: Search query string
            
        Returns:
            list: Search results from citations
        """
        logger.info("Performing search with query: %s", query)
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "accept": "application/json"
            }
            
            payload = {
                "model": "sonar",
                "messages": [{"role": "user", "content": query}],
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
                            if "message" in choice:
                                citations.extend(choice["message"].get("citations", []))
                                # Add content analysis
                                if "content" in choice["message"]:
                                    analyze_content(choice["message"]["content"])
                    
                    logger.info("Extracted %d citations", len(citations))
                    return citations
                    
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
        """Improved rating extraction with more patterns"""
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