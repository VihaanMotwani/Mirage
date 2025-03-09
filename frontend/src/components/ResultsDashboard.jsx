import { useState } from 'react';

function DynamicDataDisplay({ data }) {
  // Handle different data types
  if (data === null || data === undefined) {
    return <span className="text-gray-400 italic">None</span>;
  }
  
  if (typeof data === 'boolean') {
    return (
      <span className={data ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
        {data.toString()}
      </span>
    );
  }
  
  if (typeof data === 'number') {
    return <span className="text-indigo-600 font-medium">{data}</span>;
  }
  
  if (typeof data === 'string') {
    // Check if it's a URL
    if (data.match(/^https?:\/\//i)) {
      return (
        <a 
          href={data} 
          target="_blank" 
          rel="noopener noreferrer" 
          className="text-blue-600 hover:text-blue-800 underline break-all"
        >
          {data}
        </a>
      );
    }
    return <span className="break-words">{data}</span>;
  }
  
  // Handle arrays
  if (Array.isArray(data)) {
    if (data.length === 0) {
      return <span className="text-gray-400 italic">Empty array</span>;
    }
    
    return (
      <ul className="pl-5 space-y-2 list-disc">
        {data.map((item, index) => (
          <li key={index} className="text-gray-700">
            <DynamicDataDisplay data={item} />
          </li>
        ))}
      </ul>
    );
  }
  
  // Handle objects
  if (typeof data === 'object') {
    const entries = Object.entries(data);
    if (entries.length === 0) {
      return <span className="text-gray-400 italic">Empty object</span>;
    }
    
    return (
      <div className="pl-4 border-l-2 border-gray-200 mt-2 mb-2">
        {entries.map(([key, value]) => (
          <div key={key} className="mb-2">
            <span className="font-medium text-gray-700">{key}: </span>
            <DynamicDataDisplay data={value} />
          </div>
        ))}
      </div>
    );
  }
  
  // Fallback
  return <span>{String(data)}</span>;
}

// Enhanced KeyValueTable component
function KeyValueTable({ data }) {
  if (!data || Object.keys(data).length === 0) {
    return <div className="bg-gray-50 p-6 rounded-lg text-center text-gray-500">No data available</div>;
  }
  
  return (
    <div className="overflow-x-auto mt-4 rounded-lg border border-gray-200 shadow-sm">
      <table className="min-w-full divide-y divide-gray-200">
        <tbody className="bg-white divide-y divide-gray-200">
          {Object.entries(data).map(([key, value]) => (
            <tr key={key} className="hover:bg-gray-50 transition-colors duration-150">
              <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-700 bg-gray-50 border-r border-gray-200 w-1/3">
                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </td>
              <td className="px-6 py-4 text-gray-800">
                <DynamicDataDisplay data={value} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// Score badge component for consistent score visualization
function ScoreBadge({ score, label, size = "normal" }) {
  const getTrustColor = (score) => {
    if (score >= 80) return { bg: 'bg-green-100', text: 'text-green-800', ring: 'ring-green-600/20' };
    if (score >= 60) return { bg: 'bg-yellow-100', text: 'text-yellow-800', ring: 'ring-yellow-600/20' };
    return { bg: 'bg-red-100', text: 'text-red-800', ring: 'ring-red-600/20' };
  };

  const { bg, text, ring } = getTrustColor(score);
  
  const sizeClasses = size === "large" 
    ? "text-4xl p-6" 
    : "text-2xl p-4";

  return (
    <div className={`flex flex-col items-center justify-center ${bg} rounded-lg ring-1 ${ring} shadow-sm transition-all duration-300 hover:shadow-md`}>
      <div className={`font-bold ${text} ${sizeClasses}`}>
        {score.toFixed(1)}%
      </div>
      {label && <div className="text-sm font-medium text-gray-600 mt-1">{label}</div>}
    </div>
  );
}

export default function ResultsDashboard({ results }) {
  const [activeTab, setActiveTab] = useState('summary');

  if (!results) {
    return (
      <div className="w-full max-w-4xl mx-auto p-8 rounded-lg shadow-lg bg-white text-center">
        <div className="animate-pulse space-y-8">
          <div className="h-8 bg-gray-200 rounded w-3/4 mx-auto"></div>
          <div className="grid grid-cols-5 gap-4">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="h-64 bg-gray-200 rounded"></div>
        </div>
        <p className="text-gray-500 mt-8">Loading results...</p>
      </div>
    );
  }

  const {
    trust_score,
    metadata_score,
    reverse_image_score,
    deepfake_score,
    photoshop_score,
    fact_check_score,
    summary,
    key_findings,
    metadata_results,
    reverse_image_results,
    deepfake_results,
    photoshop_results,
    fact_check_results
  } = results;

  // Define the tabs with icons
  const tabs = [
    { key: 'summary', label: 'Summary', icon: '📋' },
    { key: 'metadata', label: 'Metadata', icon: '🔍' },
    { key: 'source', label: 'Source Check', icon: '🔎' },
    { key: 'manipulation', label: 'Manipulation', icon: '🖼️' },
    { key: 'factcheck', label: 'Fact Check', icon: '✓' }
  ];

  return (
    <div className="w-full max-w-4xl mx-auto bg-gradient-to-br from-white to-gray-50 p-8 rounded-xl shadow-lg border border-gray-100">
      {/* Header with animated gradient border */}
      <div className="relative mb-8 p-6 rounded-lg bg-white shadow-sm overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 opacity-75 blur-xl -z-10"></div>
        <div className="absolute inset-0 bg-white opacity-90 z-0"></div>
        
        <div className="relative z-10 flex justify-between items-center">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Image Verification Results
          </h2>
          <ScoreBadge score={trust_score} size="large" />
        </div>
      </div>

      {/* Score Cards - with hover effects and transitions */}
      <div className="grid grid-cols-5 gap-4 mb-8">
        {[
          { label: 'Metadata', score: metadata_score },
          { label: 'Source', score: reverse_image_score },
          { label: 'Deepfake', score: deepfake_score },
          { label: 'Photoshop', score: photoshop_score },
          { label: 'Fact Check', score: fact_check_score }
        ].map((item, index) => (
          <ScoreBadge key={index} score={item.score} label={item.label} />
        ))}
      </div>

      {/* Tabs - with active indicator animation */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex space-x-1 overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`py-4 px-4 flex items-center space-x-2 font-medium text-sm transition-all duration-300 border-b-2 ${
                activeTab === tab.key
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span className="text-lg">{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content - with card design for sections */}
      <div className="py-4">
        {activeTab === 'summary' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">Verification Summary</h3>
              <p className="mb-4 text-gray-700 leading-relaxed">{summary}</p>
            </div>
            
            <div className="bg-blue-50 p-6 rounded-lg shadow-sm border border-blue-100">
              <h4 className="font-medium mb-4 text-blue-800">Key Findings:</h4>
              {key_findings && key_findings.length > 0 ? (
                <ul className="list-disc pl-5 space-y-3 text-gray-700">
                  {key_findings.map((finding, index) => (
                    <li key={index} className="leading-relaxed">{finding}</li>
                  ))}
                </ul>
              ) : (
                <p className="text-gray-500 italic">No key findings available</p>
              )}
            </div>
          </div>
        )}

        {activeTab === 'metadata' && (
          <div className="animate-fadeIn">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">Metadata Analysis</h3>
              <p className="text-gray-600 mb-4">
                Technical metadata extracted from the image file.
              </p>
            </div>
            <KeyValueTable data={metadata_results} />
          </div>
        )}

        {activeTab === 'source' && (
          <div className="animate-fadeIn">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">Reverse Image Search</h3>
              <p className="text-gray-600 mb-4">
                Results from searching for similar images across the web.
              </p>
            </div>
            
            {reverse_image_results?.matched_sources && reverse_image_results.matched_sources.length > 0 ? (
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                <h4 className="font-medium mb-4 text-gray-800">Matched Sources:</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-700">
                  {reverse_image_results.matched_sources.map((source, index) => (
                    <li key={index} className="py-1">
                      <DynamicDataDisplay data={source} />
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <KeyValueTable data={reverse_image_results} />
            )}
          </div>
        )}

        {activeTab === 'manipulation' && (
          <div className="space-y-6 animate-fadeIn">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">Image Manipulation Analysis</h3>
              <p className="text-gray-600 mb-4">
                Detection of AI-generated content and digital manipulation.
              </p>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <h4 className="text-lg font-semibold mb-4 flex items-center text-gray-800">
                <span className="mr-2">🤖</span> Deepfake Analysis
              </h4>
              <KeyValueTable data={deepfake_results} />
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
              <h4 className="text-lg font-semibold mb-4 flex items-center text-gray-800">
                <span className="mr-2">✂️</span> Photoshop Analysis
              </h4>
              <KeyValueTable data={photoshop_results} />
            </div>
          </div>
        )}

        {activeTab === 'factcheck' && (
          <div className="animate-fadeIn">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-100 mb-6">
              <h3 className="text-xl font-semibold mb-4 text-gray-800">Fact Check Details</h3>
              <p className="text-gray-600 mb-4">
                Verification of claims and context related to the image.
              </p>
            </div>
            <KeyValueTable data={fact_check_results} />
          </div>
        )}
      </div>
      
      {/* Add custom CSS for animations */}
      <style jsx>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out forwards;
        }
      `}</style>
    </div>
  );
}