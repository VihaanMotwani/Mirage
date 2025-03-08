'use client'

import { useState, useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import ImageUpload from '@/components/ImageUpload';
import ResultsDashboard from '@/components/ResultsDashboard';
import { checkApiHealth } from '@/lib/api';

export default function Home() {
  const [results, setResults] = useState(null);
  const [apiStatus, setApiStatus] = useState({ healthy: true, message: '' });
  
  // Check API health on load
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await checkApiHealth();
        setApiStatus({ healthy: true, message: '' });
      } catch (error) {
        console.error('API health check failed:', error);
        setApiStatus({ 
          healthy: false, 
          message: 'Unable to connect to the backend API. Please ensure the server is running.'
        });
      }
    };
    
    checkHealth();
  }, []);
  
  const handleImageSubmit = (data) => {
    setResults(data);
    // In a real app, you might want to scroll to the results
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };
  
  const handleAnalyzeAnother = () => {
    setResults(null);
  };

  return (
    <div className="space-y-8">
      <div className="text-center max-w-3xl mx-auto">
        <h1 className="text-3xl font-bold mb-4">Mirage</h1>
        <p className="text-lg text-gray-600 mb-6">
          Verify image authenticity through metadata analysis, 
          reverse image search, deepfake detection, and more.
        </p>
        
        {!apiStatus.healthy && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-md flex items-center gap-3 max-w-xl mx-auto">
            <AlertTriangle className="text-amber-500 shrink-0" size={20} />
            <p className="text-amber-700 text-sm">{apiStatus.message}</p>
          </div>
        )}
      </div>

      {!results ? (
        <ImageUpload onImageSubmit={handleImageSubmit} />
      ) : (
        <ResultsDashboard 
          results={results} 
          onAnalyzeAnother={handleAnalyzeAnother} 
        />
      )}
    </div>
  );
}