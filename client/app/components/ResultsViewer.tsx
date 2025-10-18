"use client";

import { useState } from "react";
import { Download, Maximize2, XCircleIcon, CheckCircle2 } from "lucide-react";
import type { AnalysisResult } from "../types";
import Card from "./ui/Card";
import Tabs from "./ui/Tabs";
import Button from "./ui/Button";

interface ResultsViewerProps {
  result: AnalysisResult;
  originalImageUrl?: string | null;
  showLeftSide?: boolean;
}

type TabType = "original" | "heatmap" | "mask" | "tampered_regions";

export default function ResultsViewer({
  result,
  originalImageUrl,
  showLeftSide = true,
}: ResultsViewerProps) {
  const [activeTab, setActiveTab] = useState<TabType>("heatmap");

  const isProbablyTampered = result.probability >= 0.5;
  const confidencePercentage = isProbablyTampered
    ? (result.probability * 100).toFixed(1)
    : ((1 - result.probability) * 100).toFixed(1);

  const tabs = [
    { id: "original", label: "Original Image", image: originalImageUrl || "" },
    { id: "heatmap", label: "Heatmap Overlay", image: result.heatmap },
    { id: "mask", label: "Tampered Mask", image: result.mask },
    {
      id: "tampered_regions",
      label: "Tampered Regions",
      image: result.tampered_regions,
    },
  ];

  const downloadImage = (base64Data: string, filename: string) => {
    const link = document.createElement("a");
    link.href = base64Data;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const currentImage = tabs.find((tab) => tab.id === activeTab)?.image || "";

  // Left Side: Original Image and Tabs
  if (showLeftSide) {
    return (
      <Card className="p-4 h-full flex flex-col">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          Image Viewer
        </h2>

        {/* Tabs */}
        <Tabs
          tabs={tabs.map((tab) => ({ id: tab.id, label: tab.label }))}
          activeTab={activeTab}
          onChange={(id) => setActiveTab(id as TabType)}
        />

        {/* Image Display */}
        <div className="mt-4 flex-1 min-h-0">
          <div className="relative bg-gray-100 dark:bg-gray-800 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 h-full flex items-center justify-center">
            {currentImage ? (
              <div className="relative group w-full h-full flex items-center justify-center p-2">
                <img
                  src={
                    activeTab === "original"
                      ? currentImage
                      : `data:image/jpeg;base64,${currentImage}`
                  }
                  alt={`Analysis result - ${activeTab}`}
                  className="max-w-full max-h-full object-contain"
                />
                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => window.open(currentImage, "_blank")}
                  >
                    <Maximize2 className="h-4 w-4 mr-1" />
                    Full Size
                  </Button>
                </div>
              </div>
            ) : (
              <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
                <p>No image data available</p>
              </div>
            )}
          </div>
        </div>

        {/* Analysis Description - Compact */}
        <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
          <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-1 text-xs">
            Understanding the Results
          </h3>
          <div className="text-xs text-blue-800 dark:text-blue-200 space-y-0.5">
            <p>
              <strong>Original:</strong> The uploaded image
            </p>
            <p>
              <strong>Heatmap:</strong> Tampering probability (red = high)
            </p>
            <p>
              <strong>Mask:</strong> Detected areas (white = tampered)
            </p>
            <p>
              <strong>Regions:</strong> Isolated suspicious areas
            </p>
          </div>
        </div>
      </Card>
    );
  }

  // Right Side: Confidence and Verdict
  return (
    <Card className="p-4">
      <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
        Analysis Results
      </h2>

      {/* Verdict Badge */}
      <div className="mb-4">
        <div
          className={`
            w-full flex items-center justify-center gap-2 text-center px-4 py-3 rounded-lg font-semibold text-base
            ${
              isProbablyTampered
                ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"
                : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300"
            }
          `}
        >
          {isProbablyTampered ? (
            <XCircleIcon className="h-4 w-4" />
          ) : (
            <CheckCircle2 className="h-4 w-4" />
          )}
          {isProbablyTampered ? "Forgery Detected" : "Appears Authentic"}
        </div>
      </div>

      {/* Confidence Score */}
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 border border-gray-200 dark:border-gray-700 mb-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {isProbablyTampered ? "Forgery" : "Authenticity"} Confidence
          </span>
          <span className="text-xl font-bold text-gray-900 dark:text-white">
            {confidencePercentage}%
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ease-out ${
              isProbablyTampered ? "bg-red-600" : "bg-green-600"
            }`}
            style={{ width: `${confidencePercentage}%` }}
          />
        </div>
      </div>

      {/* Download Buttons - Inline */}
      <div>
        <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Download Results
        </h3>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            onClick={() =>
              downloadImage(
                `data:image/jpeg;base64,${tabs[1].image}`,
                "heatmap.png"
              )
            }
            disabled={!tabs[1].image}
            size="sm"
          >
            <Download className="h-3 w-3 mr-1.5" />
            Heatmap
          </Button>
          <Button
            variant="outline"
            onClick={() =>
              downloadImage(
                `data:image/jpeg;base64,${tabs[2].image}`,
                "mask.png"
              )
            }
            disabled={!tabs[2].image}
            size="sm"
          >
            <Download className="h-3 w-3 mr-1.5" />
            Mask
          </Button>
          <Button
            variant="outline"
            onClick={() =>
              downloadImage(
                `data:image/jpeg;base64,${tabs[3].image}`,
                "tampered_regions.png"
              )
            }
            disabled={!tabs[3].image}
            size="sm"
          >
            <Download className="h-3 w-3 mr-1.5" />
            Regions
          </Button>
        </div>
      </div>
    </Card>
  );
}
