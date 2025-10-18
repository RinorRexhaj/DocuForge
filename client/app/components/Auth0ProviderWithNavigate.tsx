"use client";

import { Auth0Provider } from "@auth0/auth0-react";
import { useRouter } from "next/navigation";
import { ReactNode } from "react";

interface Auth0ProviderWithNavigateProps {
  children: ReactNode;
}

export default function Auth0ProviderWithNavigate({
  children,
}: Auth0ProviderWithNavigateProps) {
  const router = useRouter();

  const domain = process.env.NEXT_PUBLIC_AUTH0_DOMAIN!;
  const clientId = process.env.NEXT_PUBLIC_AUTH0_CLIENT_ID!;
  const redirectUri = process.env.NEXT_PUBLIC_AUTH0_REDIRECT_URI!;
  const audience = process.env.NEXT_PUBLIC_AUTH0_AUDIENCE;

  if (!domain || !clientId || !redirectUri) {
    console.error("Auth0 environment variables are missing!");
    return <>{children}</>;
  }

  const onRedirectCallback = (appState?: any) => {
    // Force a router refresh after login to update the UI
    router.push(appState?.returnTo || "/");
    router.refresh();
  };

  // Build authorization params conditionally
  const authorizationParams: any = {
    redirect_uri: redirectUri,
    scope: "openid profile email",
  };

  // Only add audience if it's defined
  if (audience) {
    authorizationParams.audience = audience;
  }

  return (
    <Auth0Provider
      domain={domain}
      clientId={clientId}
      authorizationParams={authorizationParams}
      onRedirectCallback={onRedirectCallback}
      useRefreshTokens={true}
      cacheLocation="localstorage"
    >
      {children}
    </Auth0Provider>
  );
}
