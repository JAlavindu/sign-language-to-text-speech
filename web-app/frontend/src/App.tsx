import React, { useState, useEffect, useCallback } from 'react';
import { Camera } from './components/Camera';
import { Prediction } from './components/Prediction';
import { predictSign, checkHealth } from './services/api';

function App() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isBackendReady, setIsBackendReady] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    checkHealth().then(setIsBackendReady);
  }, []);

  const handleCapture = useCallback(async (imageSrc: string) => {
    try {
      setError(null);
      const result = await predictSign(imageSrc);
      setPrediction(result.prediction);
      setConfidence(result.confidence);
    } catch (err) {
      setError('Failed to get prediction');
    }
  }, []);

  const toggleCapture = () => {
    setIsCapturing(prev => !prev);
    if (!isCapturing) {
        // Reset state when starting
        setPrediction(null);
        setConfidence(null);
        setError(null);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans selection:bg-indigo-500 selection:text-white dark:bg-gray-900 dark:text-gray-100">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200 dark:bg-gray-900/80 dark:border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
           {/* Logo / Title */}
           <div className="flex items-center gap-2">
             <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
               <span className="text-white font-bold text-lg">ASL</span>
             </div>
             <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
               SignSense
             </h1>
           </div>
           
           {/* Backend Status */}
           <div className="flex items-center gap-2">
             <div className={`w-2.5 h-2.5 rounded-full ${isBackendReady ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`}></div>
             <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                {isBackendReady ? 'System Online' : 'Connecting to Server...'}
             </span>
           </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="pt-24 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="flex flex-col lg:flex-row gap-8 lg:gap-12 items-start justify-center">
            
            {/* Left Column: Camera */}
            <div className="w-full lg:w-1/2 flex flex-col items-center space-y-6">
                <div className="w-full rounded-2xl overflow-hidden shadow-2xl ring-1 ring-gray-900/5 dark:ring-white/10">
                     <Camera 
                        onCapture={handleCapture}
                        isCapturing={isCapturing}
                        interval={500}
                     />
                </div>

                <button
                    onClick={toggleCapture}
                    disabled={!isBackendReady}
                    className={`
                        group relative w-full sm:w-auto px-8 py-4 rounded-full font-bold text-lg tracking-wide shadow-lg transition-all transform hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed
                        ${isCapturing 
                            ? 'bg-red-500 hover:bg-red-600 text-white shadow-red-500/30' 
                            : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-500/30'
                        }
                    `}
                >
                    <span className="flex items-center justify-center gap-2">
                        {isCapturing ? (
                            <>
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                                </svg>
                                Stop Translation
                            </>
                        ) : (
                            <>
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                </svg>
                                Start Translation
                            </>
                        )}
                    </span>
                </button>
            </div>

            {/* Right Column: Prediction Results */}
            <div className="w-full lg:w-5/12 space-y-6">
                <div className="prose dark:prose-invert">
                    <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
                        Live Translation
                    </h2>
                    <p className="text-gray-500 dark:text-gray-400">
                        Perform a sign language gesture in front of the camera to translate it into text in real-time.
                    </p>
                </div>

                <div className="transform transition-all">
                    <Prediction 
                        prediction={prediction}
                        confidence={confidence}
                        error={error}
                    />
                </div>
                
                {/* Instructions or Quick Help */}
                <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-sm border border-gray-100 dark:border-gray-700">
                    <h3 className="font-semibold mb-2 flex items-center gap-2">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-indigo-500" viewBox="0 0 20 20" fill="currentColor">
                          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                        </svg>
                        Tips for Best Results
                    </h3>
                    <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                        <li className="flex items-start gap-2">
                            <span className="mt-1 block h-1 w-1 rounded-full bg-indigo-500"></span>
                            Ensure your hand is clearly visible and well-lit.
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="mt-1 block h-1 w-1 rounded-full bg-indigo-500"></span>
                            Keep your background simple and uncluttered.
                        </li>
                        <li className="flex items-start gap-2">
                            <span className="mt-1 block h-1 w-1 rounded-full bg-indigo-500"></span>
                            Position your hand in the center of the frame.
                        </li>
                    </ul>
                </div>

            </div>
        </div>
      </main>
    </div>
  );
}

export default App;
