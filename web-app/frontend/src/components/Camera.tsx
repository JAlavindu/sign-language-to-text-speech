import React, { useCallback, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';

interface CameraProps {
  onCapture: (imageSrc: string) => void;
  isCapturing: boolean;
  interval?: number; // Time in ms between captures
}

const videoConstraints = {
  width: 512,
  height: 512,
  aspectRatio: 1,
  facingMode: "user"
};

export const Camera: React.FC<CameraProps> = ({ 
  onCapture, 
  isCapturing, 
  interval = 500 
}) => {
  const webcamRef = useRef<Webcam>(null);

  const capture = useCallback(() => {
    if (webcamRef.current) {
      // getScreenshot returns a base64 string
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        onCapture(imageSrc);
      }
    }
  }, [onCapture]);

  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval>;
    
    if (isCapturing) {
      // Initial capture immediately
      capture();
      // Schedule subsequent captures
      intervalId = setInterval(capture, interval);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isCapturing, interval, capture]);

  return (
    <div className="relative w-full max-w-md mx-auto aspect-square rounded-2xl overflow-hidden shadow-2xl bg-gray-900 ring-4 ring-gray-800">
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          className="w-full h-full object-cover transform scale-x-[-1]" // Mirror the video
        />
        
        {/* Overlay Grid (optional, helps with centering) */}
        {!isCapturing && (
             <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-30">
                <div className="w-48 h-48 border-2 border-dashed border-white rounded-lg"></div>
             </div>
        )}

        {/* Status Indicator */}
        <div className="absolute top-4 right-4">
            {isCapturing ? (
                 <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/10">
                    <span className="relative flex h-3 w-3">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
                    </span>
                    <span className="text-white text-xs font-semibold tracking-wide">LIVE</span>
                </div>
            ) : (
                <div className="bg-black/60 backdrop-blur-sm px-3 py-1.5 rounded-full border border-white/10">
                    <span className="text-gray-300 text-xs font-medium">Ready</span>
                </div>
            )}
        </div>
    </div>
  );
};
