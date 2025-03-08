import { useState } from 'react';

export default function ResultsDashboard({ results }) {
  const [activeTab, setActiveTab] = useState('summary');
  
  if (!results) return null;
  
  const { 
    trust_score,
    metadata_score,
    reverse_image_score,
    deepfake_score,
    photoshop_score,
    fact_check_score,
    metadata_results,
    reverse_image_results,
    deepfake_results,
    photoshop_results,
    fact_check_results
  } = results;
  
  const getTrustColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };
  
  return (
    <div className="w-full max-w-4xl mx-auto bg-white p-8 rounded-lg shadow-md">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-2xl font-bold">Image Verification Results</h2>
        <div className={`text-3xl font-bold ${getTrustColor(trust_score)}`}>
          {trust_score}%
        </div>
      </div>
      
      <div className="grid grid-cols-5 gap-4 mb-8">
        {[
          { label: 'Metadata', score: metadata_score },
          { label: 'Source Check', score: reverse_image_score },
          { label: 'Deepfake', score: deepfake_score },
          { label: 'Photoshop', score: photoshop_score },
          { label: 'Fact Check', score: fact_check_score }
        ].map((item, index) => (
          <div key={index} className="text-center p-4 border rounded-lg">
            <div className={`text-2xl font-bold ${getTrustColor(item.score)}`}>
              {item.score}%
            </div>
            <div className="text-sm text-gray-600">{item.label}</div>
          </div>
        ))}
      </div>
      
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex space-x-8">
          {['summary', 'metadata', 'source', 'manipulation', 'factcheck'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </nav>
      </div>
      
      <div className="py-4">
        {activeTab === 'summary' && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Verification Summary</h3>
            <p className="mb-4">{results.summary}</p>
            <div className="bg-gray-100 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Key Findings:</h4>
              <ul className="list-disc pl-5 space-y-2">
                {results.key_findings.map((finding, index) => (
                  <li key={index}>{finding}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
        
        {activeTab === 'metadata' && (
          <div>
            <h3 className="text-lg font-semibold mb-4">Metadata Analysis</h3>
            {/* Metadata details would go here */}
          </div>
        )}
        
        {/* Additional tab content would be implemented here */}
      </div>
    </div>
  );
}