# Auth0 Quick Reference

## Setup (5 minutes)

1. **Create Auth0 API**
   - Go to: https://auth0.com → APIs → Create API
   - Name: `DocuForge API`
   - Identifier: `https://docuforge-api`
2. **Configure Environment**

   ```bash
   # Create .env file in server/ directory
   AUTH0_DOMAIN=your-tenant.auth0.com
   AUTH0_API_AUDIENCE=https://docuforge-api
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements/requirements_api.txt
   ```

4. **Start Server**
   ```bash
   cd server
   python api/main.py
   ```

## Testing

### Get Test Token (Auth0 Dashboard)

1. Go to: APIs → DocuForge API → Test
2. Copy the access token
3. Use in requests

### cURL Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -F "file=@test.jpg"
```

### Python Example

```python
import requests

headers = {"Authorization": "Bearer YOUR_TOKEN_HERE"}
files = {"file": open("test.jpg", "rb")}
response = requests.post(
    "http://localhost:8000/predict",
    headers=headers,
    files=files
)
print(response.json())
```

### JavaScript Example

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${token}`,
  },
  body: formData,
})
  .then((res) => res.json())
  .then((data) => console.log(data));
```

## Endpoints

| Endpoint            | Auth Required | Method | Description                |
| ------------------- | ------------- | ------ | -------------------------- |
| `/`                 | ❌ No         | GET    | API information            |
| `/health`           | ❌ No         | GET    | Health check + auth status |
| `/docs`             | ❌ No         | GET    | Swagger UI                 |
| `/predict`          | ✅ Yes        | POST   | Document prediction        |
| `/detect-tampering` | ✅ Yes        | POST   | Tampering detection        |

## Response Format (with Auth)

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

## Common Errors

| Status | Error                    | Solution                     |
| ------ | ------------------------ | ---------------------------- |
| 401    | Not authenticated        | Add Authorization header     |
| 401    | Token is expired         | Get new token from Auth0     |
| 403    | Insufficient permissions | Check Auth0 user permissions |
| 400    | Invalid file type        | Use jpg, png, bmp, or tiff   |
| 503    | Model not loaded         | Restart server               |

## Disable Auth (Dev Only)

Don't set environment variables:

```bash
# No AUTH0_DOMAIN or AUTH0_API_AUDIENCE set
python api/main.py
```

Output:

```
🔐 Authentication: DISABLED ⚠️
   Running in OPEN mode (no authentication required)
```

## Get Token Programmatically

### Machine-to-Machine

```python
import requests

def get_token(domain, client_id, client_secret, audience):
    url = f"https://{domain}/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "audience": audience,
        "grant_type": "client_credentials"
    }
    response = requests.post(url, json=payload)
    return response.json()["access_token"]

token = get_token(
    domain="your-tenant.auth0.com",
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    audience="https://docuforge-api"
)
```

## Environment Variables

| Variable             | Required | Example                 | Description           |
| -------------------- | -------- | ----------------------- | --------------------- |
| `AUTH0_DOMAIN`       | Yes\*    | `your-tenant.auth0.com` | Auth0 tenant domain   |
| `AUTH0_API_AUDIENCE` | Yes\*    | `https://docuforge-api` | API identifier        |
| `AUTH0_ALGORITHMS`   | No       | `RS256`                 | JWT signing algorithm |

\*Required for auth to be enabled. If not set, API runs in open mode.

## Frontend Integration (React)

```bash
npm install @auth0/auth0-react
```

```jsx
// App.js
import { Auth0Provider, useAuth0 } from "@auth0/auth0-react";

function App() {
  return (
    <Auth0Provider
      domain="your-tenant.auth0.com"
      clientId="YOUR_CLIENT_ID"
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: "https://docuforge-api",
      }}
    >
      <MainApp />
    </Auth0Provider>
  );
}

// API call component
function UploadComponent() {
  const { getAccessTokenSilently } = useAuth0();

  const uploadFile = async (file) => {
    const token = await getAccessTokenSilently();

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: formData,
    });

    return response.json();
  };

  return <FileUploader onUpload={uploadFile} />;
}
```

## Troubleshooting

**Server won't start?**

```bash
pip install python-jose[cryptography] requests
```

**"Could not find JWKS"?**

- Check AUTH0_DOMAIN is correct
- Test: `curl https://YOUR_DOMAIN/.well-known/jwks.json`

**Token not working?**

- Check token expiration (decode at jwt.io)
- Verify audience matches AUTH0_API_AUDIENCE
- Ensure issuer matches AUTH0_DOMAIN

**Still having issues?**

- Check server logs for detailed errors
- Review Auth0 dashboard logs
- See full guide: `docs/AUTH0_SETUP_GUIDE.md`

## Security Checklist

- [ ] Use HTTPS in production (required!)
- [ ] Don't commit .env file (add to .gitignore)
- [ ] Use short-lived tokens (24 hours max)
- [ ] Enable Auth0 anomaly detection
- [ ] Configure CORS properly (no wildcard in production)
- [ ] Keep dependencies updated
- [ ] Monitor failed authentication attempts
- [ ] Never log token values
