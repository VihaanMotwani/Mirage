import { useState } from 'react';
import { UploadCloud, Link } from 'lucide-react';

export default function ImageUpload({ onImageSubmit }) {
  const [file, setFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadMethod, setUploadMethod] = useState('file'); // 'file' or 'url'
  const [preview, setPreview] = useState(null);

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
      let formData = new FormData();
      
      if (uploadMethod === 'file' && file) {
        formData.append('image', file);
        formData.append('source_type', 'upload');
      } else if (uploadMethod === 'url' && imageUrl) {
        formData.append('image_url', imageUrl);
        formData.append('source_type', 'url');
      } else {
        throw new Error('No image provided');
      }
      
      const response = await fetch('http://localhost:8000/api/verify', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Server error');
      }
      
      const data = await response.json();
      onImageSubmit(data);
    } catch (error) {
      console.error('Error:', error);
      // Handle error state here
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto bg-white p-8 rounded-lg shadow-md">
      <div className="flex space-x-4 mb-6">
        <button
          className={`flex-1 py-2 rounded-md ${uploadMethod === 'file' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => setUploadMethod('file')}
        >
          Upload File
        </button>
        <button
          className={`flex-1 py-2 rounded-md ${uploadMethod === 'url' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => setUploadMethod('url')}
        >
          Image URL
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        {uploadMethod === 'file' ? (
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
        ) : (
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
          disabled={isLoading || (!file && !imageUrl)}
          className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-md disabled:bg-blue-300"
        >
          {isLoading ? 'Analyzing...' : 'Verify Image'}
        </button>
      </form>
    </div>
  );
}