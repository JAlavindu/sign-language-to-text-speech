import axios from 'axios';

// Ensure this matches your FastAPI URL
const API_URL = 'http://localhost:8000'; 

export interface PredictionResponse {
  prediction: string;
  confidence: number;
  all_classes?: string[];
}

export const predictSign = async (imageSrc: string): Promise<PredictionResponse> => {
  try {
    // 1. Convert Base64 image to Blob
    const response = await fetch(imageSrc);
    const blob = await response.blob();

    // 2. Create FormData
    const formData = new FormData();
    // 'file' must match the parameter name in your FastAPI endpoint: 
    // async def predict(file: UploadFile = File(...))
    formData.append('file', blob, 'capture.jpg');

    // 3. Send POST request
    const { data } = await axios.post<PredictionResponse>(`${API_URL}/predict`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return data;
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
};

export const checkHealth = async (): Promise<boolean> => {
    try {
        await axios.get(`${API_URL}/`);
        return true;
    } catch (e) {
        return false;
    }
}
