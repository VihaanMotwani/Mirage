'use client'

import { useState } from 'react';
import { UploadCloud, Link as LinkIcon, Camera } from 'lucide-react';

export default function Home() {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadMethod, setUploadMethod] = useState('file'); // 'file' or 'url'
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    
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
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      // This is a placeholder. In a real app, you would send this to your API
      // For now, we'll simulate a response after a delay
      setTimeout(() => {
        setResults({
          trust_score: 75.5,
          metadata_score: 85.2,
          reverse_image_score: 70.0,
          deepfake_score: 90.3,
          photoshop_score: 65.8,
          fact_check_score: 60.0,
          summary: "This image appears to be authentic with moderate confidence. Most verification checks passed successfully.",
          key_findings: [
            "Metadata is consistent with original camera data",
            "Found 3 matching sources, oldest from 2023-05-15",
            "No signs of AI generation detected",
            "Possible minor adjustments in bottom-right corner"
          ]
        });
        setIsLoading(false);
      }, 2000);
      
    } catch (error) {
      console.error('Error:', error);
      setIsLoading(false);
    }
  };

  const getTrustColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="space-y-8">
      <div className="text-center max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">Image Verification Tool</h1>
        <p className="text-lg text-gray-600">
          Upload an image to verify its authenticity through metadata analysis, 
          reverse image search, deepfake detection, and more.
        </p>
      </div>

      {!results ? (
        <div className="max-w-2xl mx-auto bg-white p-8 rounded-lg shadow-md">
          <div className="flex space-x-4 mb-6">
            <button
              className={`flex-1 py-2 rounded-md flex items-center justify-center gap-2 ${
                uploadMethod === 'file' 
                  ? 'bg-primary-600 text-white' 
                  : 'bg-gray-200 text-gray-700'
              }`}
              onClick={() => setUploadMethod('file')}
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
            >
              <LinkIcon size={16} />
              <span>Image URL</span>
            </button>
            <button
              className={`flex-1 py-2 rounded-md flex items-center justify-center gap-2 ${
                uploadMethod === 'camera' 
                  ? 'bg-primary-600 text-white' 
                  : 'bg-gray-200 text-gray-700'
              }`}
              onClick={() => setUploadMethod('camera')}
            >
              <Camera size={16} />
              <span>Camera</span>
            </button>
          </div>

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
                    <img src={preview} alt="Preview" className="max-h-40 mx-auto" />
                  </div>
                )}
              </div>
            )}
            
            {uploadMethod === 'url' && (
              <div className="flex items-center border border-gray-300 rounded-md overflow-hidden">
                <div className="p-3 bg-gray-100">
                  <LinkIcon className="h-5 w-5 text-gray-500" />
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
            
            {uploadMethod === 'camera' && (
              <div className="border-2 border-gray-300 p-6 rounded-md text-center">
                <Camera className="mx-auto h-12 w-12 text-gray-400" />
                <p className="mt-2 text-sm text-gray-600">
                  Camera capture coming soon
                </p>
              </div>
            )}

            <button
              type="submit"
              disabled={isLoading || (uploadMethod === 'file' && !file) || (uploadMethod === 'url' && !imageUrl)}
              className="w-full mt-6 bg-primary-600 hover:bg-primary-700 text-white py-3 px-4 rounded-md disabled:bg-primary-300 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? 'Analyzing...' : 'Verify Image'}
            </button>
          </form>
        </div>
      ) : (
        <div className="max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-md">
          <div className="flex justify-between items-center mb-8">
            <h2 className="text-2xl font-bold">Image Verification Results</h2>
            <div className={`text-3xl font-bold ${getTrustColor(results.trust_score)}`}>
              {results.trust_score.toFixed(1)}%
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
            {[
              { label: 'Metadata', score: results.metadata_score },
              { label: 'Source Check', score: results.reverse_image_score },
              { label: 'Deepfake', score: results.deepfake_score },
              { label: 'Photoshop', score: results.photoshop_score },
              { label: 'Fact Check', score: results.fact_check_score }
            ].map((item, index) => (
              <div key={index} className="text-center p-4 border rounded-lg">
                <div className={`text-2xl font-bold ${getTrustColor(item.score)}`}>
                  {item.score.toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">{item.label}</div>
              </div>
            ))}
          </div>
          
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-4">Verification Summary</h3>
            <p className="mb-4">{results.summary}</p>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Key Findings:</h4>
              <ul className="list-disc pl-5 space-y-2">
                {results.key_findings.map((finding, index) => (
                  <li key={index}>{finding}</li>
                ))}
              </ul>
            </div>
          </div>
          
          <div className="flex justify-end">
            <button 
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 py-2 px-4 rounded-md"
              onClick={() => {
                setResults(null);
                setFile(null);
                setImageUrl('');
                setPreview(null);
              }}
            >
              Analyze Another Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}