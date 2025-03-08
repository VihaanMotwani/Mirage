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
        # Extract scores from each component
        metadata_score = metadata_results.get("score", 0)
        reverse_image_score = reverse_image_results.get("score", 0)
        deepfake_score = deepfake_results.get("score", 0)
        photoshop_score = photoshop_results.get("score", 0)
        fact_check_score = fact_check_results.get("score", 0)
        
        # Calculate weighted overall score
        trust_score = (
            self.weights["metadata"] * metadata_score +
            self.weights["reverse_image"] * reverse_image_score +
            self.weights["deepfake"] * deepfake_score +
            self.weights["photoshop"] * photoshop_score +
            self.weights["fact_check"] * fact_check_score
        )
        
        # Round to 1 decimal place
        trust_score = round(trust_score, 1)
        
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
        key_findings = self._generate_key_findings(
            metadata_results, 
            reverse_image_results,
            deepfake_results,
            photoshop_results,
            fact_check_results
        )
        
        return trust_score, component_scores, summary, key_findings
    
    def _generate_summary(self, trust_score, component_scores):
        """Generate a summary based on the trust score."""
        if trust_score >= 80:
            return "This image appears to be authentic with high confidence. Most verification checks passed successfully."
        elif trust_score >= 60:
            return "This image shows some signs of potential manipulation or inconsistencies, but many verification checks passed."
        elif trust_score >= 40:
            return "This image has several suspicious characteristics that suggest it may be manipulated or misrepresented."
        else:
            return "This image shows strong evidence of manipulation, forgery, or misrepresentation. It should not be trusted."
    
    def _generate_key_findings(self, metadata_results, reverse_image_results,
                              deepfake_results, photoshop_results, fact_check_results):
        """Generate key findings based on component results."""
        findings = []
        
        # Add metadata findings
        if metadata_results.get("anomalies"):
            for anomaly in metadata_results["anomalies"][:3]:  # Limit to top 3
                findings.append(f"Metadata issue: {anomaly['description']}")
        
        # Add reverse image search findings
        if reverse_image_results.get("earliest_source"):
            findings.append(f"Earliest source: {reverse_image_results['earliest_source']['date']} from {reverse_image_results['earliest_source']['site']}")
        
        # Add deepfake detection findings
        if deepfake_results.get("is_deepfake", False):
            findings.append(f"Deepfake detection: {deepfake_results.get('confidence', 0)}% confidence this is AI-generated")
        
        # Add photoshop detection findings
        if photoshop_results.get("manipulated_regions"):
            regions = len(photoshop_results["manipulated_regions"])
            findings.append(f"Found {regions} potentially edited region(s) in the image")
        
        # Add fact check findings
        if fact_check_results.get("related_fact_checks"):
            for check in fact_check_results["related_fact_checks"][:2]:  # Limit to top 2
                findings.append(f"Fact check: {check['title']} - {check['rating']}")
        
        return findings