import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TrustScoreCalculator:
    """Calculates overall trust score based on all verification components."""
    
    def __init__(self):
        # Weights for each component in the final score
        self.weights = {
            "metadata": 0.15,
            "reverse_image": 0.25,
            "deepfake": 0.25,
            "photoshop": 0.25,
            "fact_check": 0.10
        }
        logger.info("TrustScoreCalculator initialized with weights: %s", self.weights)
    
    def calculate(self, metadata_results, reverse_image_results, 
                 deepfake_results, photoshop_results, fact_check_results):
        """
        Calculate the overall trust score and component scores.
        
        Args:
            metadata_results: Results from metadata analysis
            reverse_image_results: Results from reverse image search
            deepfake_results: Results from deepfake detection
            photoshop_results: Results from photoshop detection
            fact_check_results: Results from fact checking
        
        Returns:
            tuple: (trust_score, component_scores, summary, key_findings)
        """
        logger.info("Starting trust score calculation")
        
        # Extract scores from each component
        metadata_score = metadata_results.get("score", 0)
        logger.debug("Metadata score: %s", metadata_score)
        reverse_image_score = reverse_image_results.get("score", 0)
        logger.debug("Reverse image score: %s", reverse_image_score)
        deepfake_score = deepfake_results.get("score", 0)
        logger.debug("Deepfake score: %s", deepfake_score)
        photoshop_score = photoshop_results.get("score", 0)
        logger.debug("Photoshop score: %s", photoshop_score)
        fact_check_score = fact_check_results.get("score", 0)
        logger.debug("Fact check score: %s", fact_check_score)
        
        # Calculate weighted overall score
        trust_score = (
            self.weights["metadata"] * metadata_score +
            self.weights["reverse_image"] * reverse_image_score +
            self.weights["deepfake"] * deepfake_score +
            self.weights["photoshop"] * photoshop_score +
            self.weights["fact_check"] * fact_check_score
        )
        logger.debug("Weighted trust score before rounding: %s", trust_score)
        trust_score = round(trust_score, 1)
        logger.info("Final trust score calculated: %s", trust_score)
        
        # Component scores dictionary
        component_scores = {
            "metadata": metadata_score,
            "reverse_image": reverse_image_score,
            "deepfake": deepfake_score,
            "photoshop": photoshop_score,
            "fact_check": fact_check_score
        }
        
        # Generate summary and key findings
        summary = self._generate_summary(trust_score, component_scores)
        logger.info("Summary generated: %s", summary)
        key_findings = self._generate_key_findings(
            metadata_results, 
            reverse_image_results,
            deepfake_results,
            photoshop_results,
            fact_check_results
        )
        logger.info("Key findings generated: %s", key_findings)
        
        return trust_score, component_scores, summary, key_findings
    
    def _generate_summary(self, trust_score, component_scores):
        """Generate a summary based on the trust score."""
        logger.debug("Generating summary for trust score: %s", trust_score)
        if trust_score >= 80:
            summary = "This image appears to be authentic with high confidence. Most verification checks passed successfully."
        elif trust_score >= 60:
            summary = "This image shows some signs of potential manipulation or inconsistencies, but many verification checks passed."
        elif trust_score >= 40:
            summary = "This image has several suspicious characteristics that suggest it may be manipulated or misrepresented."
        else:
            summary = "This image shows strong evidence of manipulation, forgery, or misrepresentation. It should not be trusted."
        logger.debug("Summary: %s", summary)
        return summary
    
    def _generate_key_findings(self, metadata_results, reverse_image_results,
                              deepfake_results, photoshop_results, fact_check_results):
        """Generate key findings based on component results."""
        logger.debug("Generating key findings based on component results")
        findings = []
        
        # Add metadata findings
        if metadata_results.get("anomalies"):
            for anomaly in metadata_results["anomalies"][:3]:  # Limit to top 3
                finding = f"Metadata issue: {anomaly['description']}"
                findings.append(finding)
                logger.debug("Added metadata finding: %s", finding)
        
        # Add reverse image search findings
        if reverse_image_results.get("earliest_source"):
            earliest = reverse_image_results["earliest_source"]
            finding = f"Earliest source: {earliest['date']} from {earliest['site']}"
            findings.append(finding)
            logger.debug("Added reverse image finding: %s", finding)
        
        # Add deepfake detection findings
        if deepfake_results.get("is_deepfake", False):
            confidence = deepfake_results.get('confidence', 0)
            finding = f"Deepfake detection: {confidence}% confidence this is AI-generated"
            findings.append(finding)
            logger.debug("Added deepfake finding: %s", finding)
        
        # Add photoshop detection findings
        if photoshop_results.get("manipulated_regions"):
            regions = len(photoshop_results["manipulated_regions"])
            finding = f"Found {regions} potentially edited region(s) in the image"
            findings.append(finding)
            logger.debug("Added photoshop finding: %s", finding)
        
        # Add fact check findings
        if fact_check_results.get("related_fact_checks"):
            for check in fact_check_results["related_fact_checks"][:2]:  # Limit to top 2
                finding = f"Fact check: {check['title']} - {check['rating']}"
                findings.append(finding)
                logger.debug("Added fact check finding: %s", finding)
        
        logger.info("Total key findings generated: %d", len(findings))
        return findings