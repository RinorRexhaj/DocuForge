"use client";

import { useAuth0 } from "@auth0/auth0-react";
import { useEffect } from "react";

export default function AuthDebug() {
  const auth = useAuth0();

  useEffect(() => {
    console.group("🔍 Auth0 Debug Info");
    console.log("isAuthenticated:", auth.isAuthenticated);
    console.log("isLoading:", auth.isLoading);
    console.log("user:", auth.user);
    console.log("error:", auth.error);
    console.groupEnd();
  }, [auth.isAuthenticated, auth.isLoading, auth.user, auth.error]);

  if (!process.env.NODE_ENV || process.env.NODE_ENV === "development") {
    return (
      <div className="fixed bottom-4 right-4 bg-black text-white p-4 rounded-lg shadow-lg text-xs font-mono max-w-sm z-50">
        <div className="font-bold mb-2">🔐 Auth0 Debug</div>
        <div>Loading: {auth.isLoading ? "⏳ Yes" : "✅ No"}</div>
        <div>Authenticated: {auth.isAuthenticated ? "✅ Yes" : "❌ No"}</div>
        <div>User: {auth.user?.email || "None"}</div>
        {auth.error && (
          <div className="mt-2 text-red-400">Error: {auth.error.message}</div>
        )}
      </div>
    );
  }

  return null;
}
