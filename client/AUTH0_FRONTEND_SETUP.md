# Auth0 Frontend Setup Guide

This guide will help you configure Auth0 authentication in the DocuForge Next.js frontend.

## Prerequisites

- Auth0 account (free tier works fine)
- Node.js and npm installed
- Backend API already configured with Auth0

## Step 1: Install Dependencies

```bash
npm install
```

The `@auth0/auth0-react` package has already been added to package.json.

## Step 2: Configure Auth0 Application

### Create Application in Auth0

1. Go to [Auth0 Dashboard](https://manage.auth0.com)
2. Navigate to **Applications > Applications**
3. Click **+ Create Application**
4. Fill in:
   - **Name**: `DocuForge Web App`
   - **Application Type**: `Single Page Application`
   - Click **Create**

### Configure Application Settings

1. In your application settings, find:

   - **Domain**: `dev-aam3u7jezgeelchn.us.auth0.com` (already in your .env.local)
   - **Client ID**: `0QjVH5EkTuU9Y8B2waXwAQFhByJvlSGx` (already in your .env.local)

2. **Allowed Callback URLs** (IMPORTANT!):

   ```
   http://localhost:3000
   ```

3. **Allowed Logout URLs**:

   ```
   http://localhost:3000
   ```

4. **Allowed Web Origins**:

   ```
   http://localhost:3000
   ```

5. **Allowed Origins (CORS)**:

   ```
   http://localhost:3000
   ```

6. Click **Save Changes**

## Step 3: Verify Environment Variables

Your `.env.local` file should already have:

```bash
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000

# Auth0 Configuration
NEXT_PUBLIC_AUTH0_DOMAIN=dev-aam3u7jezgeelchn.us.auth0.com
NEXT_PUBLIC_AUTH0_CLIENT_ID=0QjVH5EkTuU9Y8B2waXwAQFhByJvlSGx
NEXT_PUBLIC_AUTH0_REDIRECT_URI=http://localhost:3000
NEXT_PUBLIC_AUTH0_AUDIENCE=https://docuforge-api
```

**Important**: Make sure `NEXT_PUBLIC_AUTH0_AUDIENCE` matches the API identifier you created in the backend setup!

## Step 4: Start the Application

```bash
npm run dev
```

The app will run at: http://localhost:3000

## Step 5: Test Authentication

### Test Login Flow

1. Open http://localhost:3000
2. You should see a **Login** button in the top-right corner
3. Click **Login**
4. You'll be redirected to Auth0's login page
5. Sign up or log in with your credentials
6. After successful authentication, you'll be redirected back to the app
7. You should see your name and email in the top-right corner

### Test File Upload with Auth

1. After logging in, upload a document image
2. Click **Analyze Document**
3. The request will automatically include your Auth0 token
4. You should see the analysis results

### Test Logout

1. Click the **Logout** button in the top-right
2. You'll be logged out and redirected back to the homepage
3. The **Login** button should reappear

## What Was Implemented

### Components Created/Modified

1. **`Auth0ProviderWithNavigate.tsx`** (NEW)

   - Wraps the entire app with Auth0 context
   - Handles authentication state
   - Manages token refresh

2. **`AuthNavbar.tsx`** (NEW)

   - Displays login/logout buttons
   - Shows user information when authenticated
   - Clean, modern UI with Tailwind CSS

3. **`FileUpload.tsx`** (MODIFIED)

   - Now uses `useAuth0()` hook
   - Automatically includes access token in API requests
   - Shows authentication errors clearly

4. **`api.ts`** (MODIFIED)

   - Accepts optional `getAccessToken` function
   - Adds `Authorization: Bearer <token>` header
   - Handles 401/403 authentication errors

5. **`layout.tsx`** (MODIFIED)

   - Wraps app with `Auth0ProviderWithNavigate`

6. **`page.tsx`** (MODIFIED)
   - Added `AuthNavbar` component at the top

## How It Works

### Authentication Flow

```
1. User clicks "Login"
   ↓
2. Redirected to Auth0 login page
   ↓
3. User enters credentials
   ↓
4. Auth0 validates and redirects back to app
   ↓
5. App receives authentication state + user info
   ↓
6. User can now upload documents
```

### API Request Flow (Authenticated)

```
1. User uploads document
   ↓
2. FileUpload calls getAccessTokenSilently()
   ↓
3. Auth0 returns JWT access token (cached)
   ↓
4. Token added to request headers: Authorization: Bearer <token>
   ↓
5. Backend validates token
   ↓
6. Backend processes request and returns response with user_id
```

### Token Management

- **Automatic Token Refresh**: Auth0 SDK handles token refresh automatically
- **Token Caching**: Tokens are cached in localStorage
- **Silent Authentication**: Uses refresh tokens to get new access tokens without re-login

## Troubleshooting

### Issue: "Callback URL mismatch" Error

**Solution**: Make sure you've added `http://localhost:3000` to:

- Allowed Callback URLs
- Allowed Logout URLs
- Allowed Web Origins
- Allowed Origins (CORS)

In Auth0 Dashboard: Applications > Your App > Settings

### Issue: "Invalid audience" Error

**Solution**: Verify that:

1. `NEXT_PUBLIC_AUTH0_AUDIENCE` in frontend matches API identifier in backend
2. In Auth0 Dashboard: APIs > Your API > Settings > Identifier
3. Both should be the same (e.g., `https://docuforge-api`)

### Issue: "Authentication failed" or "401 Unauthorized"

**Solution**:

1. Make sure backend is running with Auth0 enabled
2. Check backend logs for Auth0 configuration
3. Verify backend `.env` has `AUTH0_DOMAIN` and `AUTH0_API_AUDIENCE`
4. Test getting a token manually:
   ```javascript
   const { getAccessTokenSilently } = useAuth0();
   const token = await getAccessTokenSilently();
   console.log(token);
   ```

### Issue: Can't find module '@auth0/auth0-react'

**Solution**: Install dependencies

```bash
npm install
```

### Issue: Login redirects to wrong URL

**Solution**: Check `NEXT_PUBLIC_AUTH0_REDIRECT_URI` in `.env.local` matches your current URL

## Production Deployment

When deploying to production, update:

### 1. Environment Variables

```bash
NEXT_PUBLIC_API_URL=https://api.your-domain.com
NEXT_PUBLIC_AUTH0_REDIRECT_URI=https://your-domain.com
```

### 2. Auth0 Application Settings

Add production URLs to:

- Allowed Callback URLs: `https://your-domain.com`
- Allowed Logout URLs: `https://your-domain.com`
- Allowed Web Origins: `https://your-domain.com`
- Allowed Origins (CORS): `https://your-domain.com`

### 3. Backend CORS Configuration

Update backend `main.py` to allow your production domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Update this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Security Best Practices

1. ✅ **Never commit `.env.local`** - Already in `.gitignore`
2. ✅ **Use HTTPS in production** - Auth0 requires it
3. ✅ **Use short-lived tokens** - Configured in Auth0 API settings
4. ✅ **Implement token refresh** - Already handled by Auth0 SDK
5. ✅ **Validate tokens on backend** - Already implemented
6. ✅ **Use proper CORS settings** - Configure in production

## Testing Without Authentication (Development Only)

If you want to test without authentication temporarily:

1. Remove Auth0 environment variables from backend
2. Backend will run in "open mode"
3. Frontend will work without tokens

**WARNING**: Only for local development. Never deploy without authentication!

## Additional Resources

- [Auth0 React SDK Documentation](https://auth0.com/docs/libraries/auth0-react)
- [Auth0 Next.js Quickstart](https://auth0.com/docs/quickstart/spa/nextjs)
- [Auth0 Dashboard](https://manage.auth0.com)

## Support

If you encounter issues:

1. Check browser console for errors
2. Check Auth0 Dashboard > Monitoring > Logs for authentication attempts
3. Verify all environment variables are set correctly
4. Make sure backend is running and Auth0 is enabled
5. Test with Auth0's built-in test tool in Dashboard

## Summary

✅ Auth0 authentication fully integrated
✅ Automatic token management
✅ Secure API requests with JWT tokens
✅ User login/logout functionality
✅ Error handling for authentication failures
✅ Production-ready configuration

Your frontend is now ready to authenticate users and make secure API calls to the backend!
