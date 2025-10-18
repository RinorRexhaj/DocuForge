import axios, { AxiosError } from "axios";
import type { AnalysisResult, ApiResponse, UploadedFile } from "../types";

// Backend API base URL - Update this to match your backend server
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

// Simplified API call - no authentication tokens needed
// Frontend Auth0 is only for user login/logout UI
export const analyzeImage = async (file: File): Promise<AnalysisResult> => {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await axios.post<AnalysisResult>(
      `${API_BASE_URL}/predict`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 60000, // 60 second timeout
      }
    );

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<{ detail?: string }>;

      throw new Error(
        axiosError.response?.data?.detail ||
          "Failed to analyze image. Please check if the backend server is running."
      );
    }
    throw new Error("An unexpected error occurred during image analysis.");
  }
};

export const getHistory = async () => {
  try {
    const response = await axios.get<UploadedFile[]>(`${API_BASE_URL}/history`);

    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<{ detail?: string }>;

      throw new Error(
        axiosError.response?.data?.detail ||
          "Failed to retrieve analysis history. Please check if the backend server is running."
      );
    }
    throw new Error(
      "An unexpected error occurred while fetching analysis history."
    );
  }
};

// Mock response for testing without backend
// export const getMockAnalysisResult = async (
//   file: File
// ): Promise<ApiResponse> => {
//   // Simulate network delay
//   await new Promise((resolve) => setTimeout(resolve, 2000));

//   // Generate a mock base64 image (1x1 transparent pixel)
//   const mockBase64 =
//     "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

//   return {
//     probability: Math.random() * 0.8 + 0.2, // Random probability between 0.2 and 1.0
//     heatmap: mockBase64,
//     mask: mockBase64,
//     tampered_regions: mockBase64,
//     prediction: Math.random() > 0.5 ? "Tampered" : "Clean",
//   };
// };
