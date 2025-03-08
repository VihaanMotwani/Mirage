/**
 * API service for communicating with the backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Verify an image using the backend API
 * 
 * @param {File|string} image - Image file or URL
 * @param {string} sourceType - 'upload' or 'url'
 * @returns {Promise<Object>} Verification results
 */
export async function verifyImage(image, sourceType) {
  try {
    const formData = new FormData();
    formData.append('source_type', sourceType);
    
    if (sourceType === 'upload' && image instanceof File) {
      formData.append('image', image);
    } else if (sourceType === 'url' && typeof image === 'string') {
      formData.append('image_url', image);
    } else {
      throw new Error('Invalid image or source type');
    }
    
    console.log(`Sending verification request to ${API_BASE_URL}/api/verify`);
    
    const response = await fetch(`${API_BASE_URL}/api/verify`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      // Try to get error details from response
      let errorDetail = 'Unknown error';
      try {
        const errorData = await response.json();
        errorDetail = errorData.detail || errorData.message || String(response.status);
      } catch (e) {
        errorDetail = `HTTP error ${response.status}`;
      }
      
      throw new Error(`Verification failed: ${errorDetail}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API error:', error);
    throw error;
  }
}

/**
 * Get verification history
 * 
 * @param {number} limit - Maximum number of items to return
 * @returns {Promise<Array>} Verification history
 */
export async function getVerificationHistory(limit = 10) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/history?limit=${limit}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch history: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API error:', error);
    throw error;
  }
}

/**
 * Check API health
 * 
 * @returns {Promise<Object>} Health status
 */
export async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API health check error:', error);
    throw error;
  }
}