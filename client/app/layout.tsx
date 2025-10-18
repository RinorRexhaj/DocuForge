import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Auth0ProviderWithNavigate from "./components/Auth0ProviderWithNavigate";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "DocuForge - Document Forgery Detection",
  description: "AI-powered document tampering detection and analysis",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Auth0ProviderWithNavigate>
          <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
            {children}
          </div>
        </Auth0ProviderWithNavigate>
      </body>
    </html>
  );
}
