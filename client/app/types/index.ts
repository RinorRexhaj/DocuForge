export interface AnalysisResult {
  probability: number;
  confidence: number;
  heatmap: string; // base64 encoded image
  mask: string; // base64 encoded image
  tampered_regions: string; // base64 encoded image
  prediction?: string;
}

export interface UploadedFile {
  id: string;
  filename: string;
  created_at: string;
  prediction: "forged" | "authentic";
  confidence: number;
}

export interface ApiResponse {
  confidence: number;
  heatmap: string;
  mask: string;
  tampered_regions: string;
  prediction: string;
  error?: string;
}
