"use client";

import { useAuth0 } from "@auth0/auth0-react";
import { Shield, Scan, Lock, CheckCircle, AlertTriangle } from "lucide-react";

export default function WelcomeBanner() {
  const { loginWithRedirect } = useAuth0();

  return (
    <div className="min-h-screen flex flex-col">
      {/* Hero Section */}
      <div className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl w-full space-y-8 text-center">
          {/* Logo and Title */}
          <div className="space-y-4">
            <div className="flex justify-center">
              <div className="bg-blue-600 p-4 rounded-2xl shadow-lg">
                <Shield className="h-16 w-16 text-white" />
              </div>
            </div>
            <h1 className="text-5xl font-bold text-gray-900 dark:text-white">
              DocuForge
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300">
              AI-Powered Document Forgery Detection
            </p>
          </div>

          {/* Description */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8 space-y-6">
            <p className="text-lg text-gray-700 dark:text-gray-300">
              Protect your documents with advanced AI technology. DocuForge uses
              state-of-the-art machine learning to detect tampering, forgery,
              and manipulation in digital documents.
            </p>

            {/* Features */}
            <div className="grid md:grid-cols-3 gap-6 mt-8">
              <div className="space-y-3">
                <div className="flex justify-center">
                  <Scan className="h-10 w-10 text-blue-600" />
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Advanced Detection
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Detects blur inconsistencies, color manipulation, and
                  copy-move forgeries
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex justify-center">
                  <AlertTriangle className="h-10 w-10 text-orange-600" />
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Visual Analysis
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Get heatmaps and masks showing exactly where tampering was
                  detected
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex justify-center">
                  <CheckCircle className="h-10 w-10 text-green-600" />
                </div>
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Instant Results
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Upload your document and receive detailed analysis in seconds
                </p>
              </div>
            </div>

            {/* CTA Button */}
            <div className="pt-6">
              <button
                onClick={() => loginWithRedirect()}
                className="inline-flex items-center space-x-3 bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 rounded-xl text-lg font-semibold transition-all transform hover:scale-105 shadow-lg"
              >
                <Lock className="h-6 w-6" />
                <span>Login to Get Started</span>
              </button>
              <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
                Secure authentication powered by Auth0
              </p>
            </div>
          </div>

          {/* Additional Info */}
          <div className="bg-blue-50 dark:bg-gray-900 rounded-xl p-6">
            <h3 className="font-semibold text-gray-900 dark:text-white mb-3">
              How It Works
            </h3>
            <div className="flex flex-col md:flex-row justify-center items-center space-y-4 md:space-y-0 md:space-x-8 text-sm text-gray-600 dark:text-gray-400">
              <div className="flex items-center space-x-2">
                <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center font-bold text-xs">
                  1
                </span>
                <span>Login securely</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center font-bold text-xs">
                  2
                </span>
                <span>Upload your document</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center font-bold text-xs">
                  3
                </span>
                <span>Get instant analysis</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
