"use client";

import { Clock, RotateCw, XCircle, CheckCircle } from "lucide-react";
import type { UploadedFile } from "../types";
import Card from "./ui/Card";
import Button from "./ui/Button";
import dayjs from "dayjs";

interface FileHistoryListProps {
  files: UploadedFile[];
  onReanalyze: (fileId: string) => void;
}

export default function FileHistoryList({
  files,
  onReanalyze,
}: FileHistoryListProps) {
  const formatTimestamp = (timestamp: string) => {
    // UTC format to DD/MM HH:mm
    const formatted = dayjs(timestamp).format("DD/MM HH:mm");
    return formatted;
  };

  return (
    <Card className="p-4">
      <div className="flex items-center space-x-2 mb-3">
        <Clock className="h-4 w-4 text-gray-600 dark:text-gray-400" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          Analysis History
        </h2>
      </div>

      {files.length === 0 ? (
        <div className="text-center py-6 text-gray-500 dark:text-gray-400">
          <p className="text-sm">No analyses yet</p>
          <p className="text-xs mt-1">Upload an image to get started</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {files.map((file) => (
            <div
              key={file.id}
              className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-600 transition-colors"
            >
              <div className="flex items-start justify-between mb-1.5">
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-gray-900 dark:text-white truncate">
                    {file.filename}
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5">
                    {formatTimestamp(file.created_at)}
                  </p>
                </div>
                <div
                  className={`
                  ml-2 px-2 py-1 rounded text-xs font-semibold whitespace-nowrap
                  ${
                    file.prediction === "forged"
                      ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                      : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
                  }
                `}
                >
                  {file.prediction === "forged" ? (
                    <XCircle className="h-4 w-4" />
                  ) : (
                    <CheckCircle className="h-4 w-4" />
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Confidence: {(file.confidence * 100).toFixed(1)}%
                </div>
                {/* <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => onReanalyze(file.id)}
                >
                  <RotateCw className="h-3 w-3 mr-1" />
                  Re-analyze
                </Button> */}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
