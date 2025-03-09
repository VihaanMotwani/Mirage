import { useState } from 'react';
import { UploadCloud, Link, AlertTriangle, Info } from 'lucide-react';

export default function ImageUpload({ onImageSubmit }) {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadMethod, setUploadMethod] = useState('file'); 
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    processFile(selectedFile);
  };

  const processFile = (selectedFile) => {
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setError(null);
    
    // Generate preview
    const reader = new FileReader();
    reader.onload = () => {
      setPreview(reader.result);
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleUrlChange = (e) => {
    setImageUrl(e.target.value);
    setError(null);
    // Clear preview when URL changes
    if (preview && uploadMethod === 'url') {
      setPreview(null);
    }
  };

  const handleURLPreview = () => {
    if (!imageUrl) return;
    
    // Only update preview if URL is valid
    try {
      new URL(imageUrl);
      setPreview(imageUrl);
    } catch (e) {
      setError("Please enter a valid URL");
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      const API_URL = 'http://localhost:8000';
      
      const formData = new FormData();
      
      if (uploadMethod === 'file' && file) {
        formData.append('source_type', 'upload');
        formData.append('image', file);
        console.log("Sending file:", file.name, file.type, file.size);
      } else if (uploadMethod === 'url' && imageUrl) {
        formData.append('source_type', 'url');
        formData.append('image_url', imageUrl);
        console.log("Sending URL:", imageUrl);
      } else {
        throw new Error('No image provided');
      }
      
      const endpoint = `${API_URL}/api/verify`;
      console.log(`Sending verification request to ${endpoint}`);
      
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      
      console.log("Response status:", response.status);
      
      if (!response.ok) {
        // Try to get error details from response
        let errorDetail = 'Unknown error';
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorData.message || String(response.status);
          console.error("API error:", errorData);
        } catch (e) {
          errorDetail = `HTTP error ${response.status}`;
          console.error("Error parsing error response:", e);
        }
        
        throw new Error(`Verification failed: ${errorDetail}`);
      }
      
      const result = await response.json();
      console.log("Received results:", result);
      
      onImageSubmit(result);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message || 'Failed to verify image');
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-white p-8 rounded-lg shadow-md mb-12">
      <div className="flex space-x-4 mb-6">
        <button
          className={`flex-1 py-3 rounded-md flex items-center justify-center gap-2 transition-all ${
            uploadMethod === 'file' 
              ? 'bg-blue-600 text-white shadow-md' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => setUploadMethod('file')}
          type="button"
        >
          <UploadCloud size={18} />
          <span className="font-medium">Upload File</span>
        </button>
        <button
          className={`flex-1 py-3 rounded-md flex items-center justify-center gap-2 transition-all ${
            uploadMethod === 'url' 
              ? 'bg-blue-600 text-white shadow-md' 
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
          onClick={() => setUploadMethod('url')}
          type="button"
        >
          <Link size={18} />
          <span className="font-medium">Image URL</span>
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 rounded-md flex items-start">
          <AlertTriangle className="text-red-500 mr-2 mt-0.5 flex-shrink-0" size={18} />
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {uploadMethod === 'file' && (
          <div 
            className={`border-2 border-dashed rounded-lg transition-all ${
              isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
            } p-8 text-center`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer block">
              <UploadCloud className="mx-auto h-14 w-14 text-gray-400" />
              <p className="mt-3 text-sm text-gray-600 font-medium">
                Drag and drop or click to upload
              </p>
              <p className="mt-1 text-xs text-gray-500">
                JPEG, PNG, WebP, and other common formats accepted
              </p>
            </label>
            
            {preview && (
              <div className="mt-6 border rounded-md p-2 bg-white">
                <img src={preview} alt="Preview" className="max-h-64 mx-auto object-contain" />
                {file && (
                  <p className="text-xs text-gray-500 mt-2">
                    {file.name} ({Math.round(file.size/1024)} KB)
                  </p>
                )}
              </div>
            )}
          </div>
        )}
        
        {uploadMethod === 'url' && (
          <div className="space-y-4">
            <div className="flex items-center border border-gray-300 rounded-lg overflow-hidden focus-within:ring-2 focus-within:ring-blue-500 focus-within:border-blue-500">
              <div className="p-3 bg-gray-100">
                <Link className="h-5 w-5 text-gray-500" />
              </div>
              <input
                type="url"
                placeholder="Paste image URL here"
                value={imageUrl}
                onChange={handleUrlChange}
                className="flex-1 p-3 outline-none text-black"
              />
              <button 
                type="button"
                onClick={handleURLPreview}
                className="px-4 py-2 text-blue-600 font-medium text-sm hover:bg-blue-50"
              >
                Preview
              </button>
            </div>
            
            {preview && uploadMethod === 'url' && (
              <div className="border rounded-md p-2 bg-white">
                <img 
                  src={preview} 
                  alt="URL Preview" 
                  className="max-h-64 mx-auto object-contain"
                  onError={() => {
                    setError("Failed to load image from URL");
                    setPreview(null);
                  }} 
                />
              </div>
            )}
          </div>
        )}

        <div className="mt-2 flex items-start text-xs text-gray-500">
          <Info size={14} className="mr-1 flex-shrink-0 mt-0.5" />
          <p>By uploading, you agree to our terms and privacy policy.</p>
        </div>

        <button
          type="submit"
          disabled={isLoading || (uploadMethod === 'file' && !file) || (uploadMethod === 'url' && !imageUrl)}
          className="w-full mt-8 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-md disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors font-medium shadow-md flex items-center justify-center"
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing Image...
            </span>
          ) : 'Verify This Image'}
        </button>
      </form>
      
      {isLoading && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
          <div className="flex mb-2">
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div className="bg-blue-600 h-2.5 rounded-full animate-pulse w-3/4"></div>
            </div>
          </div>
          <p className="text-sm font-medium text-blue-800">Analysis in progress...</p>
          <p className="text-xs text-blue-700 mt-1">
            We're checking metadata, source credibility, manipulation detection, and context. This may take 30-60 seconds.
          </p>
        </div>
      )}
    </div>
  );
}