import React from 'react';

interface PredictionProps {
  prediction: string | null;
  confidence: number | null;
  error?: string | null;
}

export const Prediction: React.FC<PredictionProps> = ({ prediction, confidence, error }) => {
  // Format confidence as percentage
  const confidencePercent = confidence ? Math.round(confidence * 100) : 0;
  
  // Color coding based on confidence
  const getConfidenceColor = (score: number) => {
    if (score >= 90) return 'text-green-500';
    if (score >= 70) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="w-full max-w-md mx-auto mt-6 p-6 bg-white rounded-2xl shadow-lg border border-gray-100 dark:bg-gray-800 dark:border-gray-700">
      {error ? (
        <div className="text-center text-red-500 font-medium">
            <p>Error: {error}</p>
        </div>
      ) : (
        <div className="text-center space-y-4">
            <div>
                <h3 className="text-sm uppercase tracking-wider text-gray-500 font-semibold mb-1">
                    Detected Sign
                </h3>
                <div className="flex justify-center items-center h-24 mb-2">
                     {prediction ? (
                        <span className="text-7xl font-bold text-gray-900 dark:text-white">
                            {prediction}
                        </span>
                     ) : (
                        <span className="text-4xl text-gray-300 dark:text-gray-600">
                             --
                        </span>
                     )}
                </div>
            </div>

            {/* Confidence Bar */}
            {prediction && (
                <div className="w-full">
                    <div className="flex justify-between items-end mb-1">
                        <span className="text-xs font-medium text-gray-500">Confidence</span>
                        <span className={`text-sm font-bold ${getConfidenceColor(confidencePercent)}`}>
                            {confidencePercent}%
                        </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700 overflow-hidden">
                        <div 
                            className={`h-2.5 rounded-full transition-all duration-300 ease-out ${
                                confidencePercent >= 90 ? 'bg-green-500' : 
                                confidencePercent >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${confidencePercent}%` }}
                        ></div>
                    </div>
                </div>
            )}
            
            {!prediction && (
                <p className="text-sm text-gray-400 italic">
                    Waiting for input...
                </p>
            )}
        </div>
      )}
    </div>
  );
};
