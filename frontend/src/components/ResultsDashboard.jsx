import { useState } from 'react';

function KeyValueTable({ data }) {
  if (!data || Object.keys(data).length === 0) {
    return <p className="text-gray-600">No data available.</p>;
  }
  return (
    <div className="overflow-x-auto mt-4">
      <table className="min-w-full border border-gray-200">
        <tbody>
          {Object.entries(data).map(([key, value]) => (
            <tr key={key} className="border-b border-gray-200">
              <td className="px-4 py-2 font-semibold text-gray-600">{key}</td>
              <td className="px-4 py-2 text-gray-800">
                {typeof value === 'object' ? JSON.stringify(value, null, 2) : value}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

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
    summary,
    key_findings,
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

  // Define the tabs
  const tabs = [
    { key: 'summary', label: 'Summary' },
    { key: 'metadata', label: 'Metadata' },
    { key: 'source', label: 'Source Check' },
    { key: 'manipulation', label: 'Manipulation' },
    { key: 'factcheck', label: 'Fact Check' }
  ];

  return (
    <div className="w-full max-w-4xl mx-auto bg-gradient-to-br from-white to-gray-50 p-8 rounded-lg shadow-lg">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold text-gray-800">Image Verification Results</h2>
        <div className={`text-4xl font-bold ${getTrustColor(trust_score)}`}>
          {/* Round trust score to 1 decimal place */}
          {trust_score.toFixed(1)}%
        </div>
      </div>

      {/* Score Cards */}
      <div className="grid grid-cols-5 gap-4 mb-8">
        {[
          { label: 'Metadata', score: metadata_score },
          { label: 'Source Check', score: reverse_image_score },
          { label: 'Deepfake', score: deepfake_score },
          { label: 'Photoshop', score: photoshop_score },
          { label: 'Fact Check', score: fact_check_score }
        ].map((item, index) => (
          <div
            key={index}
            className="text-center p-4 bg-white rounded-lg border shadow-sm"
          >
            <div className={`text-2xl font-bold ${getTrustColor(item.score)}`}>
              {/* Round each score to 1 decimal place */}
              {item.score.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600 mt-1">{item.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors duration-300 ${
                activeTab === tab.key
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="py-4">
        {activeTab === 'summary' && (
          <div>
            <h3 className="text-xl font-semibold mb-4">Verification Summary</h3>
            <p className="mb-4 text-gray-700">{summary}</p>
            <div className="bg-gray-100 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Key Findings:</h4>
              <ul className="list-disc pl-5 space-y-2 text-gray-700">
                {key_findings?.map((finding, index) => (
                  <li key={index}>{finding}</li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'metadata' && (
          <div>
            <h3 className="text-xl font-semibold mb-4">Metadata Analysis</h3>
            <KeyValueTable data={metadata_results} />
          </div>
        )}

        {activeTab === 'source' && (
          <div>
            <h3 className="text-xl font-semibold mb-4">Reverse Image Search Details</h3>
            {reverse_image_results?.matched_sources ? (
              <div>
                <h4 className="font-medium mb-2">Matched Sources:</h4>
                <ul className="list-disc pl-5 space-y-2 text-gray-700">
                  {reverse_image_results.matched_sources.map((source, index) => (
                    <li key={index}>{source}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <KeyValueTable data={reverse_image_results} />
            )}
          </div>
        )}

        {activeTab === 'manipulation' && (
          <div>
            <h3 className="text-xl font-semibold mb-4">Image Manipulation Analysis</h3>
            <div>
              <h4 className="text-lg font-semibold mt-4 mb-2">Deepfake Analysis</h4>
              <KeyValueTable data={deepfake_results} />
            </div>
            <div>
              <h4 className="text-lg font-semibold mt-6 mb-2">Photoshop Analysis</h4>
              <KeyValueTable data={photoshop_results} />
            </div>
          </div>
        )}

        {activeTab === 'factcheck' && (
          <div>
            <h3 className="text-xl font-semibold mb-4">Fact Check Details</h3>
            <KeyValueTable data={fact_check_results} />
          </div>
        )}
      </div>
    </div>
  );
}