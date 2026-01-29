import React, { useCallback, useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";

interface CameraProps {
  onCapture: (imageSrc: string) => void;
  isCapturing: boolean;
  interval?: number; // Time in ms between captures
}

const videoConstraints = {
  width: 512,
  height: 512,
  aspectRatio: 1,
  facingMode: "user",
};

export const Camera: React.FC<CameraProps> = ({
  onCapture,
  isCapturing,
  interval = 500,
}) => {
  const webcamRef = useRef<Webcam>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const capture = useCallback(() => {
    if (webcamRef.current && isCameraReady) {
      // getScreenshot returns a base64 string
      // Resize to model input size (224x224) to reduce bandwidth and latency
      const imageSrc = webcamRef.current.getScreenshot({
        width: 224,
        height: 224,
      });
      if (imageSrc) {
        onCapture(imageSrc);
      }
    }
  }, [onCapture, isCameraReady]);

  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval>;

    if (isCapturing && isCameraReady) {
      // Initial capture immediately
      capture();
      // Schedule subsequent captures
      intervalId = setInterval(capture, interval);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isCapturing, interval, capture, isCameraReady]);

  const handleUserMedia = () => {
    setIsCameraReady(true);
    setCameraError(null);
  };

  const handleUserMediaError = (error: string | DOMException) => {
    setIsCameraReady(false);
    setCameraError("Could not access camera. Please allow permissions.");
    console.error(error);
  };

  return (
    <div className="relative w-full max-w-md mx-auto aspect-square rounded-3xl overflow-hidden shadow-2xl bg-gray-900 border border-gray-800">
      {/* Loading State */}
      {!isCameraReady && !cameraError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-gray-900 text-gray-400">
          <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-indigo-500 mb-4"></div>
          <p className="text-sm font-medium">Initializing camera...</p>
        </div>
      )}

      {/* Error State */}
      {cameraError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-gray-900 px-6 text-center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-12 w-12 text-red-500 mb-3"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
          <p className="text-white font-medium mb-1">Camera Error</p>
          <p className="text-sm text-gray-400">{cameraError}</p>
        </div>
      )}

      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        videoConstraints={videoConstraints}
        className={`w-full h-full object-cover transform scale-x-[-1] transition-opacity duration-500 ${isCameraReady ? "opacity-100" : "opacity-0"}`}
        onUserMedia={handleUserMedia}
        onUserMediaError={handleUserMediaError}
      />

      {/* Guide Overlay - Always visible when ready to help positioning */}
      {isCameraReady && (
        <div className="absolute inset-0 pointer-events-none">
          {/* Viewfinder corners */}
          <div className="absolute top-8 left-8 w-16 h-16 border-t-4 border-l-4 border-white/20 rounded-tl-xl transition-all duration-300"></div>
          <div className="absolute top-8 right-8 w-16 h-16 border-t-4 border-r-4 border-white/20 rounded-tr-xl transition-all duration-300"></div>
          <div className="absolute bottom-8 left-8 w-16 h-16 border-b-4 border-l-4 border-white/20 rounded-bl-xl transition-all duration-300"></div>
          <div className="absolute bottom-8 right-8 w-16 h-16 border-b-4 border-r-4 border-white/20 rounded-br-xl transition-all duration-300"></div>

          {/* Center focus area with explicit instruction */}
          {!isCapturing && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-64 h-64 border-2 border-dashed border-white/30 rounded-2xl flex items-center justify-center bg-black/20 backdrop-blur-[2px]">
                <p className="text-white font-semibold text-sm bg-black/50 px-4 py-2 rounded-full shadow-sm">
                  Place Hand Here
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Status Indicator */}
      <div className="absolute top-4 right-4 z-20">
        {isCapturing ? (
          <div className="flex items-center gap-2 bg-red-500/90 text-white px-3 py-1.5 rounded-full shadow-lg shadow-red-500/20 backdrop-blur-sm transition-all duration-300">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-white"></span>
            </span>
            <span className="text-xs font-bold tracking-wider uppercase">
              Live
            </span>
          </div>
        ) : isCameraReady ? (
          <div className="flex items-center gap-2 bg-black/50 text-white px-3 py-1.5 rounded-full backdrop-blur-md border border-white/10 transition-all duration-300">
            <div className="w-2 h-2 rounded-full bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.6)]"></div>
            <span className="text-xs font-medium">Ready</span>
          </div>
        ) : null}
      </div>

      {/* Active Scan Effect - Subtle Pulse */}
      {isCapturing && (
        <div className="absolute inset-0 z-0 pointer-events-none border-[3px] border-red-500/20 rounded-3xl animate-pulse"></div>
      )}
    </div>
  );
};
