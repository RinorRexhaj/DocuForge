"use client";

import { useState, useEffect } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import FileUpload from "./components/FileUpload";
import ResultsViewer from "./components/ResultsViewer";
import FileHistoryList from "./components/FileHistoryList";
import AuthNavbar from "./components/AuthNavbar";
// import WelcomeBanner from "./components/WelcomeBanner";
// import AuthDebug from "./components/AuthDebug";
import { Moon, Sun, Upload, History, Plus } from "lucide-react";
import type { AnalysisResult, UploadedFile } from "./types";
import { getHistory } from "./services/api";

export default function Home() {
  const { isAuthenticated, isLoading: authLoading, user } = useAuth0();
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  );
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [darkMode, setDarkMode] = useState(false);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(true);

  // Debug authentication state
  //   useEffect(() => {
  //     console.log("Auth State:", { isAuthenticated, authLoading, user });
  //   }, [isAuthenticated, authLoading, user]);

  // Initialize dark mode from localStorage on mount
  useEffect(() => {
    const storedDarkMode = localStorage.getItem("darkMode");
    if (storedDarkMode !== null) {
      const isDark = storedDarkMode === "true";
      setDarkMode(isDark);
      if (isDark) {
        document.documentElement.classList.add("dark");
      }
    }
    getHistoryFiles();
  }, []);

  const getHistoryFiles = async () => {
    try {
      const history = await getHistory();
      setUploadedFiles(history);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  const handleAnalysisComplete = (
    result: AnalysisResult,
    fileName: string,
    imageUrl: string
  ) => {
    setAnalysisResult(result);
    setIsLoading(false);
    setOriginalImageUrl(imageUrl);

    // Add to uploaded files history
    const newFile: UploadedFile = {
      id: Date.now().toString(),
      filename: fileName,
      created_at: new Date().toISOString(),
      prediction: result.probability >= 0.5 ? "forged" : "authentic",
      confidence: result.confidence,
    };
    setUploadedFiles((prev) => [newFile, ...prev]);
  };

  const handleAnalysisStart = () => {
    setIsLoading(true);
    setAnalysisResult(null);
  };

  const handleNewAnalysis = () => {
    setAnalysisResult(null);
    setOriginalImageUrl(null);
    setIsLoading(false);
  };

  const handleReanalyze = (fileId: string) => {
    // console.log("Re-analyze file:", fileId);
    // Implementation for re-analyzing a file from history
  };

  const toggleDarkMode = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem("darkMode", String(newDarkMode));

    if (newDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  // Show loading state while checking authentication
  //   if (authLoading) {
  //     return (
  //       <main className="min-h-screen flex flex-col">
  //         <AuthNavbar />
  //         <AuthDebug />
  //         <div className="flex-1 flex items-center justify-center">
  //           <div className="text-center">
  //             <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
  //             <p className="mt-4 text-gray-600 dark:text-gray-400">Loading...</p>
  //           </div>
  //         </div>
  //       </main>
  //     );
  //   }

  // Show welcome banner if not authenticated
  //   if (!isAuthenticated) {
  //     return (
  //       <main className="min-h-screen flex flex-col">
  //         <AuthNavbar />
  //         <AuthDebug />
  //         <WelcomeBanner />
  //       </main>
  //     );
  //   }

  // Show dashboard if authenticated
  return (
    <main className="min-h-screen flex flex-col">
      {/* Auth Navbar */}
      {/* <AuthNavbar /> */}
      {/* <AuthDebug /> */}

      {/* Compact Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <div className="flex items-center justify-between">
            {/* Logo and Title */}
            <div className="flex items-center space-x-2">
              <div className="bg-blue-600 px-1.5 py-1 rounded-lg">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6 text-white"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                DocuForge
              </h1>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center space-x-2">
              {analysisResult && (
                <>
                  <button
                    onClick={handleNewAnalysis}
                    className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5"
                    title="New Analysis"
                  >
                    <Plus className="h-4 w-4" />
                    <span className="hidden sm:inline">New Analysis</span>
                  </button>
                  <button
                    onClick={() => setShowHistory(!showHistory)}
                    className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center space-x-1.5 ${
                      showHistory
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                    }`}
                    title="Upload History"
                  >
                    <History className="h-4 w-4" />
                    <span className="hidden sm:inline">History</span>
                  </button>
                </>
              )}
              <button
                onClick={toggleDarkMode}
                className="px-2 py-1.5 rounded-lg bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                aria-label="Toggle dark mode"
                title="Toggle Theme"
              >
                {darkMode ? (
                  <Sun className="h-5 w-5 text-gray-700 dark:text-gray-300" />
                ) : (
                  <Moon className="h-5 w-5 text-gray-700 dark:text-gray-300" />
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-4">
        {!analysisResult ? (
          // Upload Section - Full Width when no results
          <FileUpload
            onAnalysisComplete={handleAnalysisComplete}
            onAnalysisStart={handleAnalysisStart}
            isLoading={isLoading}
          />
        ) : (
          // Results Layout
          <div className="h-full">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
              {/* Left Side: Original Image and Tabs */}
              <div className="flex flex-col">
                <ResultsViewer
                  result={analysisResult}
                  originalImageUrl={originalImageUrl}
                  showLeftSide={true}
                />
              </div>

              {/* Right Side: Confidence/Verdict and History */}
              <div className="flex flex-col gap-4">
                {/* Confidence and Verdict Section */}
                <ResultsViewer
                  result={analysisResult}
                  originalImageUrl={originalImageUrl}
                  showLeftSide={false}
                />

                {/* File History - Conditional */}
                {showHistory && (
                  <FileHistoryList
                    files={uploadedFiles}
                    onReanalyze={handleReanalyze}
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 py-3">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-center text-gray-600 dark:text-gray-400 text-xs">
            © 2025 DocuForge. AI-powered document analysis and tampering
            detection.
          </p>
        </div>
      </footer>
    </main>
  );
}
