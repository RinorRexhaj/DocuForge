"""
Auth0 Authentication for DocuForge API
======================================
This module provides Auth0 JWT token verification for FastAPI endpoints.

Setup:
1. Create an Auth0 account and application
2. Set environment variables:
   - AUTH0_DOMAIN: Your Auth0 domain (e.g., your-tenant.auth0.com)
   - AUTH0_API_AUDIENCE: Your API identifier
   - AUTH0_ALGORITHMS: JWT algorithms (default: RS256)

Usage:
    from api.auth import get_current_user
    
    @app.get("/protected")
    async def protected_route(user: dict = Depends(get_current_user)):
        return {"message": f"Hello {user['sub']}"}
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import requests
import os
from functools import lru_cache
import time

# Auth0 configuration from environment variables
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE", "")
AUTH0_ALGORITHMS = os.getenv("AUTH0_ALGORITHMS", "RS256").split(",")

# Security scheme
security = HTTPBearer()


class Auth0JWTBearer:
    """
    Auth0 JWT Bearer token validator.
    Verifies JWT tokens against Auth0's public keys.
    """
    
    def __init__(self, domain: str, api_audience: str, algorithms: list = None):
        self.domain = domain
        self.api_audience = api_audience
        self.algorithms = algorithms or ["RS256"]
        self._jwks_cache = None
        self._jwks_cache_time = 0
        self._cache_duration = 3600  # Cache JWKS for 1 hour
        
        # Validate configuration
        if not self.domain:
            raise ValueError("AUTH0_DOMAIN environment variable is required")
        if not self.api_audience:
            raise ValueError("AUTH0_API_AUDIENCE environment variable is required")
    
    @lru_cache(maxsize=128)
    def _get_jwks(self) -> dict:
        """
        Fetch JSON Web Key Set (JWKS) from Auth0.
        Results are cached for performance.
        """
        current_time = time.time()
        
        # Return cached JWKS if still valid
        if self._jwks_cache and (current_time - self._jwks_cache_time) < self._cache_duration:
            return self._jwks_cache
        
        # Fetch fresh JWKS
        jwks_url = f"https://{self.domain}/.well-known/jwks.json"
        try:
            response = requests.get(jwks_url, timeout=10)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._jwks_cache_time = current_time
            return self._jwks_cache
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Unable to fetch Auth0 JWKS: {str(e)}"
            )
    
    def _get_signing_key(self, token: str) -> str:
        """
        Extract the signing key from JWKS based on the token's kid.
        
        Args:
            token: JWT token string
            
        Returns:
            Signing key for token verification
        """
        try:
            # Decode token header without verification
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            
            if not kid:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing 'kid' in header",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Get JWKS
            jwks = self._get_jwks()
            
            # Find matching key
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    # Construct RSA key
                    from jose.utils import base64url_decode
                    return {
                        "kty": key.get("kty"),
                        "kid": key.get("kid"),
                        "use": key.get("use"),
                        "n": key.get("n"),
                        "e": key.get("e"),
                    }
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to find appropriate signing key",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token header: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def verify_token(self, credentials: HTTPAuthorizationCredentials) -> dict:
        """
        Verify JWT token and return decoded payload.
        
        Args:
            credentials: HTTP Authorization credentials
            
        Returns:
            Decoded JWT payload containing user information
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        token = credentials.credentials
        
        try:
            # Get signing key
            signing_key = self._get_signing_key(token)
            
            # Verify and decode token
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=self.algorithms,
                audience=self.api_audience,
                issuer=f"https://{self.domain}/"
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTClaimsError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token claims: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Token verification failed: {str(e)}"
            )


# Initialize Auth0 JWT Bearer
try:
    auth0_bearer = Auth0JWTBearer(
        domain=AUTH0_DOMAIN,
        api_audience=AUTH0_API_AUDIENCE,
        algorithms=AUTH0_ALGORITHMS
    )
except ValueError as e:
    print(f"⚠️  Warning: Auth0 not configured - {e}")
    print("   Set AUTH0_DOMAIN and AUTH0_API_AUDIENCE environment variables to enable authentication")
    auth0_bearer = None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Dependency to get the current authenticated user from JWT token.
    
    Usage:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"user_id": user["sub"]}
    
    Returns:
        dict: Decoded JWT payload containing user information:
            - sub: User ID (subject)
            - email: User email (if available)
            - permissions: List of permissions (if using RBAC)
            - etc.
    
    Raises:
        HTTPException: If authentication fails
    """
    if not auth0_bearer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured. Please set AUTH0_DOMAIN and AUTH0_API_AUDIENCE."
        )
    
    return await auth0_bearer.verify_token(credentials)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[dict]:
    """
    Optional authentication dependency.
    Returns user info if authenticated, None otherwise.
    
    Usage:
        @app.get("/public-or-private")
        async def route(user: Optional[dict] = Depends(get_current_user_optional)):
            if user:
                return {"message": f"Hello {user['sub']}"}
            return {"message": "Hello anonymous"}
    
    Returns:
        dict or None: User information if authenticated, None otherwise
    """
    if not credentials or not auth0_bearer:
        return None
    
    try:
        return await auth0_bearer.verify_token(credentials)
    except HTTPException:
        return None


def require_permissions(*required_permissions: str):
    """
    Decorator to require specific permissions for an endpoint.
    Requires Auth0 RBAC (Role-Based Access Control) to be configured.
    
    Usage:
        @app.get("/admin")
        @require_permissions("read:admin", "write:admin")
        async def admin_route(user: dict = Depends(get_current_user)):
            return {"message": "Admin access granted"}
    
    Args:
        *required_permissions: Permission strings to require
    
    Returns:
        Dependency function that checks permissions
    """
    async def permission_checker(user: dict = Depends(get_current_user)) -> dict:
        user_permissions = user.get("permissions", [])
        
        for permission in required_permissions:
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {permission}"
                )
        
        return user
    
    return permission_checker


# Helper function to check if auth is enabled
def is_auth_enabled() -> bool:
    """
    Check if Auth0 authentication is properly configured.
    
    Returns:
        bool: True if auth is enabled, False otherwise
    """
    return auth0_bearer is not None


# Helper function to get user info from token (without validation)
def decode_token_unsafe(token: str) -> dict:
    """
    Decode JWT token WITHOUT verification (for debugging only).
    DO NOT use this for authentication!
    
    Args:
        token: JWT token string
        
    Returns:
        dict: Decoded token payload
    """
    try:
        return jwt.get_unverified_claims(token)
    except Exception as e:
        return {"error": str(e)}
