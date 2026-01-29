import React, { useState, useEffect, useCallback, useRef } from "react";
import { Camera } from "./components/Camera";
import { Prediction } from "./components/Prediction";
import { predictSign, checkHealth } from "./services/api";

function App() {
  const [isCapturing, setIsCapturing] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [sentence, setSentence] = useState<string>("");
  const lastSignRef = useRef<string | null>(null);
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

      // Low confidence filter (e.g., 80%) prevents false positives from background noise
      if (result.confidence < 0.8) {
        setPrediction(null);
        setConfidence(result.confidence);
        // Optional: Reset debouncer if confidence drops (assume hand moved/dropped)
        // lastSignRef.current = null;
        return;
      }

      const sign = result.prediction;
      setPrediction(sign);
      setConfidence(result.confidence);

      // Sentence Construction Logic
      if (sign) {
        const lastSign = lastSignRef.current;

        // Only add if different from last sign (debouncing)
        // Exception: "NOTHING" resets the debounce
        if (sign !== lastSign) {
          if (sign === "NOTHING") {
            lastSignRef.current = null;
          } else {
            // Valid new command/character
            if (sign === "SPACE") {
              setSentence((prev) => prev + " ");
            } else if (sign === "DEL") {
              setSentence((prev) => prev.slice(0, -1));
            } else {
              setSentence((prev) => prev + sign);
            }
            lastSignRef.current = sign;
          }
        }
      }
    } catch (err) {
      setError("Failed to get prediction");
    }
  }, []);

  const toggleCapture = () => {
    setIsCapturing((prev) => !prev);
    if (!isCapturing) {
      // Reset state when starting
      setPrediction(null);
      setConfidence(null);
      setError(null);
      lastSignRef.current = null;
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-indigo-500 selection:text-white dark:bg-slate-900 dark:text-slate-100">
      {/* Background decoration */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl transform -translate-y-1/2"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl transform translate-y-1/2"></div>
      </div>

      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-slate-200 dark:bg-slate-900/80 dark:border-slate-800 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo / Title */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl shadow-lg shadow-indigo-500/20 flex items-center justify-center transform transition-transform hover:scale-105">
              <span className="text-white font-bold text-lg leading-none">
                ASL
              </span>
            </div>
            <h1 className="text-2xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
              SignSense
            </h1>
          </div>

          {/* Backend Status */}
          <div className="flex items-center gap-3 bg-slate-100 dark:bg-slate-800 px-4 py-2 rounded-full border border-slate-200 dark:border-slate-700">
            <div className={`relative flex h-3 w-3`}>
              <span
                className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isBackendReady ? "bg-green-400" : "bg-red-400"}`}
              ></span>
              <span
                className={`relative inline-flex rounded-full h-3 w-3 ${isBackendReady ? "bg-green-500" : "bg-red-500"}`}
              ></span>
            </div>
            <span
              className={`text-sm font-semibold ${isBackendReady ? "text-slate-700 dark:text-slate-200" : "text-slate-500 dark:text-slate-400"}`}
            >
              {isBackendReady ? "System Online" : "Connecting..."}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 pt-28 pb-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="flex flex-col lg:flex-row gap-8 lg:gap-16 items-start justify-center">
          {/* Left Column: Camera */}
          <div className="w-full lg:w-1/2 flex flex-col items-center space-y-8">
            <div className="w-full relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-[2rem] blur opacity-25 group-hover:opacity-50 transition duration-1000 group-hover:duration-200"></div>
              <div className="relative w-full rounded-[1.75rem] overflow-hidden shadow-2xl bg-slate-900 border border-slate-800">
                <Camera
                  onCapture={handleCapture}
                  isCapturing={isCapturing}
                  interval={100}
                />
              </div>
            </div>

            <div className="flex flex-col items-center w-full">
              <button
                onClick={toggleCapture}
                disabled={!isBackendReady}
                className={`
                            relative overflow-hidden w-full sm:w-auto px-10 py-4 rounded-2xl font-bold text-lg tracking-wide shadow-xl transition-all duration-300 transform hover:-translate-y-1 hover:shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed
                            ${
                              isCapturing
                                ? "bg-red-500 hover:bg-red-600 text-white shadow-red-500/30 ring-4 ring-red-500/20"
                                : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-500/30 ring-4 ring-indigo-500/20"
                            }
                        `}
              >
                <span className="relative z-10 flex items-center justify-center gap-3">
                  {isCapturing ? (
                    <>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6 animate-pulse"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2.5}
                          d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2.5}
                          d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
                        />
                      </svg>
                      Stop Translation
                    </>
                  ) : (
                    <>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-6 w-6"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2.5}
                          d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2.5}
                          d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      Start Translation
                    </>
                  )}
                </span>
              </button>
              {!isBackendReady && (
                <p className="mt-4 text-sm text-slate-500 animate-pulse">
                  Waiting for server connection...
                </p>
              )}
            </div>
          </div>

          {/* Right Column: Prediction Results */}
          <div className="w-full lg:w-5/12 space-y-8">
            <div className="space-y-2">
              <h2 className="text-3xl font-extrabold text-slate-900 dark:text-white tracking-tight">
                Live Translation
              </h2>
              <p className="text-lg text-slate-600 dark:text-slate-400 leading-relaxed">
                Sign gestures are translated into text in real-time using our
                advanced AI model.
              </p>
            </div>

            <div className="transform transition-all hover:scale-[1.02] duration-300">
              <Prediction
                prediction={prediction}
                confidence={confidence}
                error={error}
              />
            </div>

            {/* Sentence Display */}
            <div className="bg-white dark:bg-slate-800 rounded-2xl p-6 shadow-lg border border-slate-200 dark:border-slate-700 transition-all">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                  Constructed Sentence
                </h3>
                <button
                  onClick={() => setSentence("")}
                  className="text-sm px-4 py-1.5 text-red-600 bg-red-50 hover:bg-red-100 dark:bg-red-900/20 dark:hover:bg-red-900/40 dark:text-red-300 rounded-lg transition-colors font-medium"
                >
                  Clear
                </button>
              </div>
              <div className="min-h-[4rem] p-4 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-700 font-mono text-xl text-slate-800 dark:text-slate-200 break-words shadow-inner">
                {sentence}
                {!sentence && (
                  <span className="text-slate-400 italic text-base">
                    Start signing to build a sentence...
                  </span>
                )}
                {sentence && (
                  <span className="animate-pulse inline-block w-2.5 h-5 bg-indigo-500 ml-1 align-middle"></span>
                )}
              </div>
            </div>

            {/* Instructions or Quick Help */}
            <div className="bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm rounded-2xl p-8 shadow-sm border border-slate-200 dark:border-slate-700 transition-colors">
              <h3 className="font-bold text-lg mb-4 flex items-center gap-2 text-slate-900 dark:text-white">
                <div className="p-1.5 bg-indigo-100 dark:bg-indigo-900/50 rounded-lg text-indigo-600 dark:text-indigo-400">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                Tips for Best Results
              </h3>
              <ul className="space-y-4">
                {[
                  "Ensure your hand is clearly visible and well-lit.",
                  "Keep your background simple and uncluttered.",
                  "Position your hand in the center of the frame.",
                  "Hold the sign steady for a moment.",
                ].map((tip, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-3 text-slate-600 dark:text-slate-300"
                  >
                    <span className="mt-1.5 flex h-2 w-2 rounded-full bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]"></span>
                    <span className="text-sm font-medium">{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
