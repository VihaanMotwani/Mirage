import { useState } from 'react';
import { UploadCloud, Link } from 'lucide-react';

export default function ImageUpload({ onImageSubmit }) {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadMethod, setUploadMethod] = useState('file'); // 'file' or 'url'
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
    
    // Generate preview
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleUrlChange = (e) => {
    setImageUrl(e.target.value);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      // For direct debugging, let's do the fetch directly here
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
          className={`flex-1 py-2 rounded-md flex items-center justify-center gap-2 ${
            uploadMethod === 'file' 
              ? 'bg-primary-600 text-white' 
              : 'bg-gray-200 text-gray-700'
          }`}
          onClick={() => setUploadMethod('file')}
          type="button"
        >
          <UploadCloud size={16} />
          <span>Upload File</span>
        </button>
        <button
          className={`flex-1 py-2 rounded-md flex items-center justify-center gap-2 ${
            uploadMethod === 'url' 
              ? 'bg-primary-600 text-white' 
              : 'bg-gray-200 text-gray-700'
          }`}
          onClick={() => setUploadMethod('url')}
          type="button"
        >
          <Link size={16} />
          <span>Image URL</span>
        </button>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit}>
        {uploadMethod === 'file' && (
          <div className="border-2 border-dashed border-gray-300 p-6 rounded-md text-center">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <UploadCloud className="mx-auto h-12 w-12 text-gray-400" />
              <p className="mt-2 text-sm text-gray-600">
                Drag and drop or click to upload
              </p>
            </label>
            
            {preview && (
              <div className="mt-4">
                <img src={preview} alt="Preview" className="max-h-64 mx-auto" />
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
          <div className="flex items-center border border-gray-300 rounded-md overflow-hidden">
            <div className="p-3 bg-gray-100">
              <Link className="h-5 w-5 text-gray-500" />
            </div>
            <input
              type="url"
              placeholder="Paste image URL here"
              value={imageUrl}
              onChange={handleUrlChange}
              className="flex-1 p-3 outline-none"
            />
          </div>
        )}

        <button
          type="submit"
          disabled={isLoading || (uploadMethod === 'file' && !file) || (uploadMethod === 'url' && !imageUrl)}
          className="w-full mt-8 bg-primary-600 hover:bg-primary-700 text-black py-3 px-4 rounded-md disabled:bg-primary-300 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing...
            </span>
          ) : 'Verify Image'}
        </button>
      </form>
      
      {isLoading && (
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
          <p className="text-sm font-medium text-blue-800 mb-2">Analysis in progress...</p>
          <p className="text-xs text-blue-700">Image verification may take up to 30-60 seconds. Please be patient.</p>
        </div>
      )}
    </div>
  );
}