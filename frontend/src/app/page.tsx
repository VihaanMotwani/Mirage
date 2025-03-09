'use client'
import { useState, useEffect } from 'react';
import { AlertTriangle, Shield, ArrowUpCircle } from 'lucide-react';
import ImageUpload from '@/components/ImageUpload';
import ResultsDashboard from '@/components/ResultsDashboard';
import { checkApiHealth } from '@/lib/api';

export default function Home() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState({ healthy: true, message: '' });

  // Check API health on load
  useEffect(() => {
    const checkHealth = async () => {
      try {
        setLoading(true);
        await checkApiHealth();
        setApiStatus({ healthy: true, message: '' });
      } catch (error) {
        console.error('API health check failed:', error);
        setApiStatus({
          healthy: false,
          message: 'Unable to connect to the backend API. Please ensure the server is running.'
        });
      } finally {
        setLoading(false);
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
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white">
      <header className="border-b border-gray-100 bg-white sticky top-0 z-10 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Shield className="text-indigo-600" />
            <h1 className="text-2xl font-bold text-gray-900">Mirage</h1>
          </div>
          {!results && !loading && (
            <p className="text-sm text-gray-500 hidden md:block">Verify image authenticity</p>
          )}
          {results && (
            <button
              onClick={handleAnalyzeAnother}
              className="text-sm text-indigo-600 hover:text-indigo-800 flex items-center gap-1"
            >
              <ArrowUpCircle size={16} />
              Analyze another image
            </button>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {!results ? (
          <div className="space-y-8">
            <div className="text-center max-w-3xl mx-auto">
              <h2 className="text-3xl font-bold mb-4 text-gray-900">Mirage</h2>
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
            <div className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
              <ImageUpload onImageSubmit={handleImageSubmit} />
            </div>
          </div>
        ) : (
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
            <ResultsDashboard
              results={results}
              onAnalyzeAnother={handleAnalyzeAnother}
            />
          </div>
        )}
      </main>

      <footer className="border-t border-gray-100 py-6 mt-12">
        <div className="max-w-6xl mx-auto px-4 text-center text-sm text-gray-500">
          <p>© {new Date().getFullYear()} Mirage. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}