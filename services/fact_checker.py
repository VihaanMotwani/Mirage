import aiohttp
import logging
import os
import json
from typing import List, Dict, Any, Optional
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class FactChecker:
    """
    Enhanced fact checker that uses Perplexity to gather information and OpenAI to analyze it.
    This approach leverages Perplexity's search capabilities and OpenAI's contextual understanding
    for more accurate fact checking of images.
    """

    def __init__(self):
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Log warnings if API keys are missing
        if not self.perplexity_api_key or self.perplexity_api_key == "your_perplexity_api_key_here":
            logger.warning("PERPLEXITY_API_KEY environment variable not set or has default value")
        
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable not set")
        
        self.perplexity_api_url = "https://api.perplexity.ai/chat/completions"
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
        
        # Reliability tiers for domains - kept from original implementation
        self.reliability_tiers = {
            "high": [
                "reuters.com", "apnews.com", "bbc.com", "npr.org", "politifact.com",
                "factcheck.org", "snopes.com", "aap.com.au", "afp.com", "bbc.co.uk"
            ],
            "medium": [
                "nytimes.com", "washingtonpost.com", "cnn.com", "nbcnews.com",
                "abcnews.go.com", "theguardian.com", "usatoday.com", "perplexity.ai",
                "instagram.com", "cbsnews.com", "foxnews.com", "time.com", "economist.com"
            ],
            "low": [
                "tabloids", "social-media", "unverified-blogs"
            ]
        }
        logger.info("Enhanced FactChecker initialized with two-stage analysis")

    async def check(self, content_context: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Perform two-stage fact checking:
        1. Use Perplexity to gather relevant information about the image
        2. Feed that information to OpenAI for deeper analysis and fact checking

        Args:
            content_context: List of dictionaries, each containing "title" and "description"
        
        Returns:
            dict: Results including related fact-checks and a reliability score
        """
        logger.info("Starting enhanced fact check process")
        try:
            # Return neutral results if API keys are not available
            if not self.perplexity_api_key or self.perplexity_api_key == "your_perplexity_api_key_here":
                logger.warning("Skipping fact check due to missing or default Perplexity API key")
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "Fact checking skipped - Perplexity API key not configured"
                }
                
            if not self.openai_api_key:
                logger.warning("Skipping fact check due to missing OpenAI API key")
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "Fact checking skipped - OpenAI API key not configured"
                }

            # Validate and combine content context
            if not content_context:
                logger.warning("No content context provided.")
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "No content context provided"
                }
            
            # Create search query from context
            search_query = self._create_search_query(content_context)
            if not search_query or len(search_query) < 10:
                logger.warning("Insufficient text for fact checking. Combined text < 10 characters.")
                return {
                    "score": 50,  # Neutral score
                    "related_fact_checks": [],
                    "message": "Insufficient text for fact checking"
                }
                
            logger.info(f"Created search query: {search_query}")
            
            # STAGE 1: Use Perplexity to gather relevant information
            perplexity_results = await self._search_perplexity(search_query)
            logger.info(f"Perplexity search returned {len(perplexity_results)} results")
            
            # If Perplexity search failed, return neutral score
            if not perplexity_results or (isinstance(perplexity_results, dict) and perplexity_results.get("error")):
                error_msg = perplexity_results.get("error", "Unknown Perplexity search error")
                logger.warning(f"Perplexity search failed: {error_msg}")
                return {
                    "score": 50,
                    "related_fact_checks": [],
                    "message": f"Perplexity search failed: {error_msg}"
                }
            
            # STAGE 2: Use OpenAI to analyze the Perplexity results
            analysis_results = await self._analyze_with_openai(search_query, perplexity_results)
            logger.info("OpenAI analysis completed successfully")
            
            # Extract structured information from OpenAI's analysis
            try:
                # Extract the main components from the analysis
                fact_checks = analysis_results.get("fact_checks", [])
                score = analysis_results.get("score", 50)
                
                logger.info(f"Analysis returned {len(fact_checks)} fact checks with score: {score}")
                
                return {
                    "score": score,
                    "related_fact_checks": fact_checks,
                    "analysis_summary": analysis_results.get("summary", ""),
                    "search_query": search_query,
                    "perplexity_result_count": len(perplexity_results),
                }
            except Exception as e:
                logger.error(f"Error extracting results from OpenAI analysis: {str(e)}")
                return {
                    "score": 50,
                    "error": str(e),
                    "related_fact_checks": self._create_generic_info_sources("Image verification"),
                    "message": f"Error during OpenAI analysis: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Enhanced fact checking error: {str(e)}")
            return {
                "score": 50,  # Neutral score on error
                "error": str(e),
                "related_fact_checks": self._create_generic_info_sources("Image verification"),
                "message": f"Error during fact checking: {str(e)}"
            }

    def _create_search_query(self, content_context: List[Dict[str, str]]) -> str:
        """Create an optimized search query from the content context"""
        combined_text = []
        for item in content_context:
            t = item.get("title", "").strip()
            d = item.get("description", "").strip()
            if t and len(t) > 3:
                combined_text.append(t)
            if d and len(d) > 3:
                combined_text.append(d)
        
        # Join all content with spaces
        full_text = " ".join(combined_text)
        
        # Create a more targeted search query
        query = f"{full_text} fact check verification authentic"
        
        # If the query is too long, extract key terms
        if len(query) > 300:
            query = self._extract_key_terms(full_text) + " image authentic fact check verification"
            
        return query

    def _extract_key_terms(self, text: str) -> str:
        """Extract key terms from text for better searching"""
        # Simple extraction of longer words which are more likely to be significant
        words = [w for w in text.split() if len(w) >= 4]
        
        # Get the most unique words that might help identify the image
        # Prioritize proper nouns (capitalized words not at the start of sentences)
        proper_nouns = []
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper():
                proper_nouns.append(word)
                
        # If we have proper nouns, prioritize them
        if proper_nouns and len(proper_nouns) >= 3:
            key_words = proper_nouns[:min(7, len(proper_nouns))]
        else:
            key_words = words[:min(7, len(words))]
            
        return " ".join(key_words)

    async def _search_perplexity(self, query: str) -> Dict[str, Any]:
        """
        Search using Perplexity API to gather relevant information about the image in a structured format.
        
        Args:
            query: Search query string
            
        Returns:
            dict: Raw search results with detailed information
        """
        logger.info(f"Searching Perplexity with query: {query}")
        try:
            if not self.perplexity_api_key or self.perplexity_api_key == "your_perplexity_api_key_here":
                logger.warning("Skipping Perplexity search due to missing API key")
                return {"error": "Perplexity API key not configured"}

            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a fact-checking research assistant that retrieves information about images or news stories. "
                            "When providing information, use this structured format:\n\n"
                            "ARTICLE 1:\n"
                            "SOURCE: [Publication or website name]\n"
                            "URL: [URL if available]\n"
                            "CLAIM: [What this source states about the image/story]\n"
                            "DATE: [Publication date if available]\n\n"
                            "ARTICLE 2:\n"
                            "SOURCE: [Different publication]\n"
                            "URL: [URL if available]\n"
                            "CLAIM: [Content from this source]\n"
                            "DATE: [Publication date if available]\n\n"
                            "And so on for each source you find...\n\n"
                            "Finally, include a SUMMARY section with key points about the image/story verification status.\n\n"
                            "Focus on finding fact-checking articles, news reports, and authoritative sources. "
                            "Search extensively to determine if the subject is authentic, manipulated, or misrepresented."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Find detailed information about whether this image or news story is authentic or fake: {query}"
                    }
                ],
                "options": {
                    "search_focus": "internet"
                },
                "max_tokens": 1500
            }
            
            logger.debug(f"Perplexity API request payload: {json.dumps(payload)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.perplexity_api_url,
                    headers=headers,
                    json=payload,
                    timeout=20
                ) as response:
                    response_text = await response.text()
                    logger.debug(f"Perplexity API response status: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"Perplexity API error: {response.status} - {response_text[:500]}")
                        return {"error": f"Perplexity API error: {response.status}"}
                    
                    result = json.loads(response_text)
                    
                    # Extract the raw content from Perplexity's response
                    perplexity_content = ""
                    if "choices" in result:
                        for choice in result["choices"]:
                            msg = choice.get("message", {})
                            perplexity_content += msg.get("content", "")
                    
                    logger.info("Successfully retrieved information from Perplexity")
                    
                    # Return the raw content for OpenAI to parse
                    return {
                        "raw_content": perplexity_content,
                        "result_found": bool(perplexity_content.strip()),
                        "query": query
                    }
            
            logger.debug(f"Perplexity API request payload: {json.dumps(payload)}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.perplexity_api_url,
                    headers=headers,
                    json=payload,
                    timeout=20  # Increased timeout for more thorough search
                ) as response:
                    response_text = await response.text()
                    logger.debug(f"Perplexity API response status: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"Perplexity API error: {response.status} - {response_text[:500]}")
                        return {"error": f"Perplexity API error: {response.status}"}
                    
                    result = json.loads(response_text)
                    
                    # Extract all available information
                    sources = []
                    content = ""
                    
                    if "choices" in result:
                        for choice in result["choices"]:
                            msg = choice.get("message", {})
                            
                            # Extract citations if available
                            if "citations" in msg:
                                sources.extend(msg["citations"])
                            elif "content_citations" in msg:
                                sources.extend(msg["content_citations"])
                                
                            # Always capture the full content
                            content += msg.get("content", "")
                    
                    # If no structured citations but we have content, extract information from content
                    if not sources and content:
                        logger.info("No structured citations found, extracting from content")
                        content_source = self._extract_citations_from_content(content)
                        sources.append(content_source)
                        
                        # Also extract any URLs mentioned in the content
                        urls = self._extract_urls_from_text(content)
                        for url in urls:
                            domain = self._extract_domain(url)
                            if domain:
                                sources.append({
                                    "url": url,
                                    "title": f"Source from {domain}",
                                    "snippet": "URL extracted from content"
                                })
                    
                    # Enrich sources with additional information
                    enriched_sources = []
                    for source in sources:
                        if not source:
                            continue
                            
                        url = source.get("url", "")
                        if not url:
                            continue
                            
                        domain = self._extract_domain(url)
                        reliability = self._determine_source_reliability(domain)
                        
                        enriched_source = {
                            "url": url,
                            "title": source.get("title", f"Source from {domain}"),
                            "snippet": source.get("snippet", ""),
                            "source": domain,
                            "reliability": reliability
                        }
                        enriched_sources.append(enriched_source)
                    
                    # Add the full content as a source as well, but only if we couldn't extract any real sources
                    if content and not enriched_sources:
                        # Try harder to extract actual sources from the content before falling back to Perplexity
                        mentioned_sources = self._extract_all_mentioned_sources(content)
                        
                        if mentioned_sources:
                            # If we found mentions of actual sources, add those instead of attributing to Perplexity
                            for source_name in mentioned_sources[:3]:  # Limit to top 3 sources
                                domain = source_name.lower().replace(" ", "")
                                if "." not in domain:
                                    domain = domain + ".com"
                                
                                reliability = self._determine_source_reliability(domain)
                                enriched_sources.append({
                                    "content": f"Information from {source_name}",
                                    "source": source_name,
                                    "reliability": reliability,
                                    "url": f"https://www.google.com/search?q={source_name.replace(' ', '+')}"
                                })
                        else:
                            # As a last resort, use Perplexity as the source
                            enriched_sources.append({
                                "content": content,
                                "source": "aggregated sources",
                                "reliability": "medium",
                                "url": "https://perplexity.ai/"
                            })
                    
                    logger.info(f"Enriched {len(enriched_sources)} sources with reliability information")
                    return enriched_sources
                    
        except Exception as e:
            logger.error(f"Error in Perplexity search: {str(e)}")
            return {"error": str(e)}

    async def _analyze_with_openai(self, query: str, perplexity_content: str) -> Dict[str, Any]:
        """
        Use OpenAI to parse and analyze the structured information from Perplexity.
        
        Args:
            query: Original search query
            perplexity_content: Raw content from Perplexity search
            
        Returns:
            dict: Structured analysis results including fact checks and score
        """
        logger.info("Starting OpenAI parsing and analysis of Perplexity results")
        try:
            if not self.openai_api_key:
                logger.warning("Skipping OpenAI analysis due to missing API key")
                return {"error": "OpenAI API key not configured"}

            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the system prompt
            system_prompt = """
You are a fact-checking expert who analyzes text about images or news stories to determine authenticity.

Your task is to parse the provided information (from a search service) into a structured format, and then analyze the authenticity of the subject.

The search service has returned information in a semi-structured format with multiple articles and a summary.

FIRST, extract all sources and their claims from the text.
THEN, analyze these sources to determine the authenticity of the subject.

Return your analysis as a structured JSON object with these fields:
- "score": A number from 0 to 100 representing the authenticity score (0 = definitively false, 100 = definitively authentic, 50 = unclear)
- "summary": A concise summary of your fact-check findings
- "fact_checks": An array of fact check results, each containing:
  - "title": A descriptive title for this fact check
  - "source": The source domain or name of the news organization
  - "url": The source URL (if mentioned)
  - "description": A summary of what this source claims about the subject
  - "rating": One of ["True", "Mostly True", "Partly True", "Unverified", "Partly False", "Mostly False", "False"]
  - "reliability": The reliability of the source ("high", "medium", or "low")

SCORING GUIDELINES:
- If multiple reliable sources confirm the image is FAKE, MANIPULATED, or MISREPRESENTED, score BELOW 40.
- If the image shows a real event but with false context, score between 20-35.
- If the image is an AI-generated fake, score between 0-20.
- If the image's authenticity is disputed or unclear, score around 50.
- Only if the image is verified as authentic by reliable sources should the score be above 70.

Be objective, thorough, and only draw conclusions supported by the provided information.
"""
            
            # Create the user message with the query and perplexity content
            user_message = f"""
Here is information about this query: "{query}"

This is the information I've gathered:

{perplexity_content}

Please parse this information into a structured format and analyze the authenticity of the subject.
"""

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            
            logger.debug("Sending request to OpenAI API")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.openai_api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    response_text = await response.text()
                    logger.debug(f"OpenAI API response status: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"OpenAI API error: {response.status} - {response_text[:500]}")
                        return {"error": f"OpenAI API error: {response.status}"}
                    
                    try:
                        result = json.loads(response_text)
                        
                        # Extract the analysis from the OpenAI response
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            try:
                                analysis = json.loads(content)
                                logger.info(f"OpenAI analysis completed with score: {analysis.get('score', 'unknown')}")
                                return analysis
                            except json.JSONDecodeError as json_err:
                                logger.error(f"Failed to parse OpenAI response as JSON: {str(json_err)}")
                                return {"error": "Invalid JSON in OpenAI response", "raw_content": content[:1000]}
                        else:
                            logger.error("Unexpected response format from OpenAI")
                            return {"error": "Unexpected response format from OpenAI"}
                    except Exception as e:
                        logger.error(f"Error processing OpenAI response: {str(e)}")
                        return {"error": f"Error processing OpenAI response: {str(e)}"}
                    
        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {str(e)}")
            return {"error": str(e)}

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2
            }
            
            logger.debug("Sending request to OpenAI API")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.openai_api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    response_text = await response.text()
                    logger.debug(f"OpenAI API response status: {response.status}")
                    
                    if response.status != 200:
                        logger.error(f"OpenAI API error: {response.status} - {response_text[:500]}")
                        return {"error": f"OpenAI API error: {response.status}"}
                    
                    result = json.loads(response_text)
                    
                    # Extract the analysis from the OpenAI response
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        try:
                            analysis = json.loads(content)
                            logger.info(f"OpenAI analysis completed with score: {analysis.get('score', 'unknown')}")
                            return analysis
                        except json.JSONDecodeError as json_err:
                            logger.error(f"Failed to parse OpenAI response as JSON: {str(json_err)}")
                            return {"error": "Invalid JSON in OpenAI response", "raw_content": content[:1000]}
                    else:
                        logger.error("Unexpected response format from OpenAI")
                        return {"error": "Unexpected response format from OpenAI"}
                    
        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {str(e)}")
            return {"error": str(e)}

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """Extract URLs from text content"""
        url_pattern = r'https?://[^\s)"]+'
        return re.findall(url_pattern, text)

    def _extract_citations_from_content(self, content, url="https://perplexity.ai/search"):
        """Extract citations from the content text itself"""
        # Start with a generic source, but we'll try to find a better one
        citation_source = "aggregated sources"
        citation_reliability = "medium"
        
        # Clean the content text
        content_text = content[:800] + "..." if len(content) > 800 else content
        
        # Try to extract actual sources from content
        mentioned_sources = self._extract_all_mentioned_sources(content)
        
        # Use the first mentioned source if available
        if mentioned_sources:
            citation_source = mentioned_sources[0]
            logger.debug(f"Extracted source from content: {citation_source}")
            # Determine reliability of extracted source
            for tier, domains in self.reliability_tiers.items():
                if any(trusted_domain.lower() in citation_source.lower() for trusted_domain in domains):
                    citation_reliability = tier
                    logger.debug(f"Source {citation_source} matched reliability tier: {tier}")
                    break
        
        # Create the citation with the best source attribution
        citation = {
            "url": url,
            "title": "Analysis from " + (citation_source if citation_source != "aggregated sources" else "multiple sources"),
            "source": citation_source,
            "snippet": content_text,
            "reliability": citation_reliability
        }
        
        return citation
        
    def _extract_all_mentioned_sources(self, content):
        """Extract all mentioned sources from content text"""
        mentioned_sources = []
        
        # Look for source attributions with various indicators
        source_indicators = ["according to", "reported by", "from", "source:", "by", "published in", "article in"]
        
        for indicator in source_indicators:
            pattern = f"{indicator} ([A-Za-z0-9 ]+)"
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) > 3 and len(match) < 30:  # Reasonable source name length
                    mentioned_sources.append(match.strip())
        
        # Extract any media outlet names that might be mentioned
        media_outlets = [
            "Reuters", "Associated Press", "AP", "BBC", "CNN", 
            "New York Times", "NY Times", "Washington Post", "The Guardian", 
            "Snopes", "FactCheck.org", "PolitiFact", "NBC", "CBS", 
            "ABC News", "Fox News", "NPR", "USA Today", "Wall Street Journal",
            "The Telegraph", "The Times", "Daily Mail", "The Sun", "The Independent",
            "The Atlantic", "NBC News", "CBS News", "ABC"
        ]
        
        for outlet in media_outlets:
            # Use more precise pattern matching to avoid false positives
            pattern = r'\b' + re.escape(outlet) + r'\b'
            if re.search(pattern, content, re.IGNORECASE):
                mentioned_sources.append(outlet)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(mentioned_sources))

    def _calculate_reliability_score(self, fact_checks: List[Dict[str, Any]]) -> float:
        """
        Calculate overall reliability score based on found fact-checks.
        This revised approach weights each fact-check individually based on its
        reliability and adjusts the score so that negative ratings have a heavier impact.
        
        Args:
            fact_checks: List of extracted fact-checks
            
        Returns:
            float: Reliability score (0-100)
        """
        logger.info("Calculating reliability score based on fact-checks")
        if not fact_checks:
            logger.warning("No fact-checks found, returning neutral score of 50.0")
            return 50.0

        score = 50.0  # Start with a neutral baseline

        # Process each fact-check individually
        for fc in fact_checks:
            reliability = fc.get("reliability", "low")
            rating = fc.get("rating", "Unverified")
            
            # Determine weight based on reliability tier:
            # high = 3, medium = 2, low = 1
            weight = {"high": 3, "medium": 2, "low": 1}.get(reliability, 1)
            
            if rating in ["True", "Mostly True"]:
                # Positive evidence: add 2 points per weight unit
                delta = weight * 2
                score += delta
                logger.debug(f"Added {delta} points for rating '{rating}' with {reliability} reliability")
            elif rating in ["False", "Mostly False"]:
                # Negative evidence: subtract 4 points per weight unit
                delta = weight * 4
                score -= delta
                logger.debug(f"Subtracted {delta} points for rating '{rating}' with {reliability} reliability")
            elif rating in ["Partly True", "Partly False"]:
                # Partial evidence is adjusted slightly
                if "True" in rating:
                    delta = weight * 1
                    score += delta
                    logger.debug(f"Added {delta} points for partly true rating with {reliability} reliability")
                else:
                    delta = weight * 1
                    score -= delta
                    logger.debug(f"Subtracted {delta} points for partly false rating with {reliability} reliability")
            else:
                logger.debug(f"No adjustment for rating '{rating}'")

        # Clamp the final score between 0 and 100
        final_score = max(0, min(100, score))
        logger.info(f"Final reliability score: {final_score}")
        return final_score

    def _determine_source_reliability(self, domain: str) -> str:
        """
        Determine the reliability of a source based on domain.
        
        Args:
            domain: Website domain
            
        Returns:
            str: Reliability tier ('high', 'medium', or 'low')
        """
        for tier, domains in self.reliability_tiers.items():
            if any(trusted_domain in domain.lower() for trusted_domain in domains):
                return tier
        
        return "low"

    def _extract_domain(self, url: str) -> str:
        """
        Extract domain from URL.
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception as e:
            logger.error(f"Error extracting domain: {str(e)}")
            return url

    def _create_generic_info_sources(self, context: str) -> List[Dict[str, Any]]:
        """
        Create generic information sources when no fact-checks are found
        """
        # These are placeholders that will still render in the UI
        return [
            {
                "title": "Information about image verification technologies",
                "url": "https://www.reuters.com/fact-check/",
                "source": "reuters.com",
                "description": "Digital media can be manipulated in various ways. Consider checking multiple reliable sources for verification.",
                "rating": "Information Source",
                "reliability": "high"
            },
            {
                "title": "Resources for fact-checking visual content",
                "url": "https://factcheck.org",
                "source": "factcheck.org",
                "description": "When verifying images, consider checking metadata, reverse image search, and consulting established fact-checking resources.",
                "rating": "Information Source",
                "reliability": "high"
            }
        ]