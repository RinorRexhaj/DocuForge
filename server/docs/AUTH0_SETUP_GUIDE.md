# Auth0 Authentication Setup Guide

This guide explains how to set up Auth0 authentication for the DocuForge API to protect your endpoints with JWT-based authentication.

## Overview

The DocuForge API supports Auth0 authentication to secure endpoints. When enabled:

- ✅ `/predict` - **Protected** (requires valid JWT token)
- ✅ `/detect-tampering` - **Protected** (requires valid JWT token)
- 🌐 `/` (root) - Public
- 🌐 `/health` - Public
- 📚 `/docs` - Public (Swagger UI)

## Quick Start

### 1. Create Auth0 Account

1. Go to [https://auth0.com](https://auth0.com) and sign up for a free account
2. Create a new tenant (or use an existing one)
3. Note your **domain** (e.g., `your-tenant.auth0.com`)

### 2. Create an API in Auth0

1. In the Auth0 dashboard, navigate to **Applications > APIs**
2. Click **+ Create API**
3. Fill in the details:
   - **Name**: `DocuForge API` (or any name you prefer)
   - **Identifier**: `https://docuforge-api` (this will be your audience)
   - **Signing Algorithm**: `RS256` (recommended)
4. Click **Create**
5. Copy the **Identifier** - this is your `AUTH0_API_AUDIENCE`

### 3. Configure Environment Variables

Create a `.env` file in the `server/` directory:

```bash
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_API_AUDIENCE=https://docuforge-api
AUTH0_ALGORITHMS=RS256
```

Or set them as system environment variables:

**Windows (PowerShell):**

```powershell
$env:AUTH0_DOMAIN = "your-tenant.auth0.com"
$env:AUTH0_API_AUDIENCE = "https://docuforge-api"
$env:AUTH0_ALGORITHMS = "RS256"
```

**Linux/Mac:**

```bash
export AUTH0_DOMAIN="your-tenant.auth0.com"
export AUTH0_API_AUDIENCE="https://docuforge-api"
export AUTH0_ALGORITHMS="RS256"
```

### 4. Install Dependencies

Make sure you have the required packages installed:

```bash
pip install -r requirements/requirements_api.txt
```

This includes:

- `python-jose[cryptography]>=3.3.0` - JWT token verification
- `requests>=2.31.0` - JWKS fetching from Auth0

### 5. Start the Server

```bash
cd server
python api/main.py
```

You should see:

```
🔐 Auth0 Authentication: ENABLED
   Domain: your-tenant.auth0.com
   Audience: https://docuforge-api
```

## Testing Authentication

### Option 1: Using Auth0's Test Tool

1. In Auth0 dashboard, go to **APIs > DocuForge API > Test**
2. Copy the provided access token
3. Use it in your requests:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -F "file=@path/to/image.jpg"
```

### Option 2: Create a Test Application

1. In Auth0 dashboard, go to **Applications > Applications**
2. Click **+ Create Application**
3. Choose **Machine to Machine Applications**
4. Select your **DocuForge API**
5. Authorize the application with required permissions
6. Copy the **Client ID** and **Client Secret**

Get a token programmatically:

```python
import requests

def get_access_token():
    url = f"https://{AUTH0_DOMAIN}/oauth/token"
    payload = {
        "client_id": "YOUR_CLIENT_ID",
        "client_secret": "YOUR_CLIENT_SECRET",
        "audience": "https://docuforge-api",
        "grant_type": "client_credentials"
    }
    response = requests.post(url, json=payload)
    return response.json()["access_token"]

# Use the token
token = get_access_token()
headers = {"Authorization": f"Bearer {token}"}
files = {"file": open("image.jpg", "rb")}
response = requests.post("http://localhost:8000/predict", headers=headers, files=files)
print(response.json())
```

### Option 3: Using Swagger UI

1. Navigate to http://localhost:8000/docs
2. Click the **Authorize** button (lock icon)
3. Enter your token in the format: `Bearer YOUR_TOKEN`
4. Click **Authorize**
5. Now you can test endpoints directly from Swagger

## API Response with Authentication

When authenticated, responses include the user ID:

```json
{
  "prediction": "forged",
  "probability": 0.89,
  "confidence": 0.89,
  "filename": "document.jpg",
  "heatmap": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "tampered_regions": "data:image/png;base64,...",
  "user_id": "auth0|507f1f77bcf86cd799439011"
}
```

## Error Responses

### 401 Unauthorized - Missing Token

```json
{
  "detail": "Not authenticated"
}
```

### 401 Unauthorized - Invalid Token

```json
{
  "detail": "Invalid token: Token is expired"
}
```

### 403 Forbidden - Insufficient Permissions (if using RBAC)

```json
{
  "detail": "Insufficient permissions"
}
```

## Disabling Authentication (Development Only)

For development/testing, you can run without authentication:

1. Don't set `AUTH0_DOMAIN` and `AUTH0_API_AUDIENCE` environment variables
2. Start the server - it will run in **open mode**

```
⚠️  Auth0 Authentication: DISABLED
   Running in OPEN mode (no authentication required)
   Set AUTH0_DOMAIN and AUTH0_API_AUDIENCE to enable Auth0
```

**⚠️ WARNING**: Only use open mode for local development. Always enable authentication in production!

## Advanced Configuration

### Role-Based Access Control (RBAC)

To add permissions-based access control:

```python
from api.auth import get_current_user, require_permissions

@app.post("/admin/users")
async def admin_endpoint(
    user: dict = Depends(require_permissions(["admin:access"]))
):
    # Only users with "admin:access" permission can access this
    return {"message": "Admin access granted"}
```

Configure permissions in Auth0:

1. Go to **APIs > DocuForge API > Permissions**
2. Add permissions (e.g., `admin:access`, `read:documents`, `write:documents`)
3. Assign permissions to users or roles in **User Management**

### Optional Authentication (Hybrid Endpoints)

For endpoints that work with or without authentication:

```python
from api.auth import get_current_user_optional

@app.get("/public-data")
async def hybrid_endpoint(
    user: dict = Depends(get_current_user_optional)
):
    if user:
        # User is authenticated - provide enhanced data
        return {"data": "full_data", "user_id": user['sub']}
    else:
        # Anonymous user - provide limited data
        return {"data": "limited_data"}
```

### Custom Token Validation

Modify `server/api/auth.py` to add custom claims validation:

```python
def verify_token(self, token: str) -> dict:
    # ... existing code ...

    # Add custom validation
    if payload.get("custom_claim") != "expected_value":
        raise HTTPException(
            status_code=403,
            detail="Custom claim validation failed"
        )

    return payload
```

## Troubleshooting

### Issue: "Could not find JWKS"

**Solution**: Check your `AUTH0_DOMAIN` is correct and accessible. Test: `https://YOUR_DOMAIN/.well-known/jwks.json`

### Issue: "Invalid token: Token is expired"

**Solution**: Get a new token. Auth0 tokens typically expire after 24 hours.

### Issue: "Invalid audience"

**Solution**: Ensure `AUTH0_API_AUDIENCE` matches the API identifier in Auth0 dashboard.

### Issue: "Invalid signature"

**Solution**: Token might be from wrong Auth0 tenant. Verify `AUTH0_DOMAIN` and token source match.

### Issue: "Import 'jose' could not be resolved"

**Solution**: Install dependencies: `pip install python-jose[cryptography]`

## Security Best Practices

1. ✅ **Always use HTTPS in production** - Never send tokens over HTTP
2. ✅ **Keep tokens secure** - Store in httpOnly cookies or secure storage, never in localStorage
3. ✅ **Use short-lived tokens** - Configure token expiration in Auth0 (recommended: 1-24 hours)
4. ✅ **Implement refresh tokens** - For long-running applications
5. ✅ **Use RBAC** - Assign minimum necessary permissions to users
6. ✅ **Monitor failed attempts** - Enable Auth0 anomaly detection
7. ✅ **Keep dependencies updated** - Regularly update `python-jose` and other security packages
8. ❌ **Never commit .env file** - Add `.env` to `.gitignore`
9. ❌ **Don't log tokens** - Redact tokens from application logs

## Frontend Integration

### React + Auth0 React SDK

```bash
npm install @auth0/auth0-react
```

```javascript
// src/index.js
import { Auth0Provider } from "@auth0/auth0-react";

<Auth0Provider
  domain="your-tenant.auth0.com"
  clientId="YOUR_CLIENT_ID"
  authorizationParams={{
    redirect_uri: window.location.origin,
    audience: "https://docuforge-api",
  }}
>
  <App />
</Auth0Provider>;
```

```javascript
// src/api/client.js
import { useAuth0 } from "@auth0/auth0-react";

export function useApi() {
  const { getAccessTokenSilently } = useAuth0();

  const predictDocument = async (file) => {
    const token = await getAccessTokenSilently();

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
      },
      body: formData,
    });

    return response.json();
  };

  return { predictDocument };
}
```

## Resources

- [Auth0 Documentation](https://auth0.com/docs)
- [Auth0 Python SDK](https://github.com/auth0/auth0-python)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT.io](https://jwt.io/) - Decode and inspect JWT tokens

## Support

If you encounter issues:

1. Check the server logs for detailed error messages
2. Verify your Auth0 configuration matches the environment variables
3. Test your token at [jwt.io](https://jwt.io)
4. Review Auth0 dashboard logs for authentication attempts
