"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileImage, X, AlertCircle } from "lucide-react";
import { analyzeImage } from "../services/api";
import type { AnalysisResult } from "../types";
import Button from "./ui/Button";
import Card from "./ui/Card";

interface FileUploadProps {
  onAnalysisComplete: (
    result: AnalysisResult,
    fileName: string,
    imageUrl: string
  ) => void;
  onAnalysisStart: () => void;
  isLoading: boolean;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ACCEPTED_FILE_TYPES = {
  "image/jpeg": [".jpg", ".jpeg"],
  "image/png": [".png"],
};

export default function FileUpload({
  onAnalysisComplete,
  onAnalysisStart,
  isLoading,
}: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useMockData, setUseMockData] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(null);

    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      if (rejection.file.size > MAX_FILE_SIZE) {
        setError("File size exceeds 10MB limit");
      } else {
        setError("Invalid file type. Please upload JPG or PNG images only.");
      }
      return;
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);

      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_FILE_TYPES,
    maxSize: MAX_FILE_SIZE,
    multiple: false,
  });

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setError(null);
    onAnalysisStart();

    try {
      // Simple API call - no token needed
      // Auth0 is only for UI (showing user info in navbar)
      const result = await analyzeImage(selectedFile);

      onAnalysisComplete(result, selectedFile.name, previewUrl || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
      onAnalysisComplete(
        {
          probability: 0,
          confidence: 0,
          heatmap: "",
          mask: "",
          tampered_regions: "",
        },
        selectedFile.name,
        previewUrl || ""
      );
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setError(null);
  };

  return (
    <Card className="p-8">
      <h2 className="text-2xl font-semibold text-gray-900 dark:text-white mb-6">
        Upload Document Image
      </h2>

      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-12 text-center cursor-pointer
            transition-all duration-200 ease-in-out
            ${
              isDragActive
                ? "border-blue-500 bg-blue-50 dark:bg-blue-900/20"
                : "border-gray-300 dark:border-gray-600 hover:border-blue-400 dark:hover:border-blue-500"
            }
          `}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center space-y-4">
            <div className="bg-blue-100 dark:bg-blue-900/30 p-4 rounded-full">
              <Upload className="h-12 w-12 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900 dark:text-white">
                {isDragActive
                  ? "Drop your image here"
                  : "Drag & drop an image here"}
              </p>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                or click to browse
              </p>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500 space-y-1">
              <p>Supported formats: JPG, PNG</p>
              <p>Maximum file size: 10MB</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* File Preview */}
          <div className="relative bg-gray-50 dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
            <button
              onClick={handleRemoveFile}
              className="absolute top-2 right-2 p-1 bg-red-500 hover:bg-red-600 rounded-full text-white transition-colors"
              aria-label="Remove file"
            >
              <X className="h-4 w-4" />
            </button>

            <div className="flex items-start space-x-4">
              {previewUrl && (
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-32 h-32 object-cover rounded border border-gray-300 dark:border-gray-600"
                />
              )}
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2 mb-2">
                  <FileImage className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
                  <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {selectedFile.name}
                  </p>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
          </div>

          {/* Mock Data Toggle */}
          {/* <div className="flex items-center space-x-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
            <input
              type="checkbox"
              id="mockData"
              checked={useMockData}
              onChange={(e) => setUseMockData(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
            />
            <label
              htmlFor="mockData"
              className="text-sm text-gray-700 dark:text-gray-300"
            >
            </label>
          </div> */}

          {/* Analyze Button */}
          <Button
            onClick={handleAnalyze}
            disabled={isLoading}
            fullWidth
            size="lg"
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
                Analyzing Image...
              </>
            ) : (
              "Analyze for Tampering"
            )}
          </Button>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg flex items-start space-x-3">
          <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm font-medium text-red-800 dark:text-red-200">
              Analysis Error
            </p>
            <p className="text-sm text-red-700 dark:text-red-300 mt-1">
              {error}
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}
