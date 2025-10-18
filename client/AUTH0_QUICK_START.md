# Auth0 Frontend Integration - Quick Reference

## ✅ Installation Complete

Auth0 has been successfully integrated into your DocuForge Next.js frontend!

## 🚀 Quick Start

### 1. Configure Auth0 Application (One-time setup)

Go to [Auth0 Dashboard](https://manage.auth0.com):

1. **Applications** → **Applications** → **+ Create Application**
2. Name: `DocuForge Web App`, Type: `Single Page Application`
3. In **Settings**, add these URLs:
   - **Allowed Callback URLs**: `http://localhost:3000`
   - **Allowed Logout URLs**: `http://localhost:3000`
   - **Allowed Web Origins**: `http://localhost:3000`
   - **Allowed Origins (CORS)**: `http://localhost:3000`
4. Click **Save Changes**

### 2. Start the Application

```bash
cd client
npm run dev
```

### 3. Test It Out

1. Open http://localhost:3000
2. Click **Login** (top-right corner)
3. Sign up or log in
4. Upload a document and analyze it
5. Your request will automatically include authentication!

## 📁 What Was Created/Modified

### New Files

- ✅ `app/components/Auth0ProviderWithNavigate.tsx` - Auth0 wrapper component
- ✅ `app/components/AuthNavbar.tsx` - Login/logout navbar
- ✅ `AUTH0_FRONTEND_SETUP.md` - Complete setup guide

### Modified Files

- ✅ `package.json` - Added @auth0/auth0-react dependency
- ✅ `.env.local` - Added Auth0 environment variables
- ✅ `app/layout.tsx` - Wrapped with Auth0Provider
- ✅ `app/page.tsx` - Added AuthNavbar component
- ✅ `app/components/FileUpload.tsx` - Added Auth0 token handling
- ✅ `app/services/api.ts` - Added token authentication to API calls

## 🔐 How Authentication Works

### Before (No Auth)

```
User uploads file → API call → Backend processes → Returns result
```

### After (With Auth)

```
User logs in → Gets JWT token
   ↓
User uploads file → Token added to request → Backend validates token
   ↓
Backend processes → Returns result with user_id
```

## 🎯 Key Features

✅ **Automatic Token Management** - Auth0 SDK handles everything
✅ **Token Refresh** - Tokens auto-refresh without re-login
✅ **Secure API Calls** - All requests include JWT Bearer token
✅ **User Information** - Display user name/email in navbar
✅ **Error Handling** - Clear messages for auth failures
✅ **Modern UI** - Clean login/logout interface

## 📋 Environment Variables

Your `.env.local` file now has:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_AUTH0_DOMAIN=dev-aam3u7jezgeelchn.us.auth0.com
NEXT_PUBLIC_AUTH0_CLIENT_ID=0QjVH5EkTuU9Y8B2waXwAQFhByJvlSGx
NEXT_PUBLIC_AUTH0_REDIRECT_URI=http://localhost:3000
NEXT_PUBLIC_AUTH0_AUDIENCE=https://docuforge-api
```

**Important**: Make sure `NEXT_PUBLIC_AUTH0_AUDIENCE` matches your backend API identifier!

## 🧪 Testing

### Test Authentication

```typescript
// In any component
import { useAuth0 } from "@auth0/auth0-react";

const { isAuthenticated, user, getAccessTokenSilently } = useAuth0();

console.log("Authenticated:", isAuthenticated);
console.log("User:", user);

// Get token
const token = await getAccessTokenSilently();
console.log("Token:", token);
```

### Test API Call with Token

1. Log in
2. Upload an image
3. Check browser Network tab
4. Look for `/predict` request
5. Check Headers → Authorization: Bearer <token>

## ⚠️ Common Issues

### "Callback URL mismatch"

**Fix**: Add `http://localhost:3000` to all URL fields in Auth0 Dashboard

### "Invalid audience"

**Fix**: Verify `NEXT_PUBLIC_AUTH0_AUDIENCE` matches backend API identifier

### "401 Unauthorized"

**Fix**: Make sure backend is running with Auth0 enabled (`AUTH0_DOMAIN` and `AUTH0_API_AUDIENCE` set)

### Login button not showing

**Fix**: Check browser console for errors, verify environment variables are set

## 🔄 User Flow

### Login Flow

1. User clicks **Login** button
2. Redirected to Auth0 login page
3. User enters credentials (or signs up)
4. Auth0 validates credentials
5. Redirected back to app
6. User information displayed in navbar
7. Token stored in localStorage

### Upload Flow (Authenticated)

1. User selects document
2. Clicks **Analyze Document**
3. `FileUpload` component calls `getAccessTokenSilently()`
4. Auth0 returns cached token (or refreshes if expired)
5. Token added to request: `Authorization: Bearer <token>`
6. Backend validates token against Auth0
7. Backend processes document
8. Response includes `user_id` for audit trail

### Logout Flow

1. User clicks **Logout** button
2. Auth0 clears authentication state
3. Tokens removed from localStorage
4. User redirected to homepage
5. **Login** button reappears

## 📦 Dependencies Added

```json
{
  "@auth0/auth0-react": "^2.2.4"
}
```

This package provides:

- `Auth0Provider` - Context provider for authentication
- `useAuth0()` - React hook for accessing auth state
- `getAccessTokenSilently()` - Get JWT tokens
- `loginWithRedirect()` - Trigger login flow
- `logout()` - Log user out

## 🎨 UI Components

### AuthNavbar

- Shows **Login** button when not authenticated
- Shows **User info** + **Logout** button when authenticated
- Displays user name and email
- Modern, responsive design

### Auth0ProviderWithNavigate

- Wraps entire app
- Configures Auth0 with environment variables
- Handles navigation after login/logout
- Enables token refresh and caching

## 🔒 Security Features

✅ **JWT Token Validation** - Backend verifies every token
✅ **Token Refresh** - Automatic refresh before expiration
✅ **Secure Storage** - Tokens in localStorage (httpOnly cookies in production)
✅ **HTTPS Required** - Auth0 enforces HTTPS in production
✅ **CORS Protection** - Configure allowed origins in Auth0
✅ **Audience Validation** - Tokens only work for your API

## 📊 API Response Changes

### Before Authentication

```json
{
  "prediction": "forged",
  "probability": 0.89,
  "confidence": 0.89,
  "filename": "document.jpg",
  "heatmap": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "tampered_regions": "data:image/png;base64,..."
}
```

### With Authentication

```json
{
  "prediction": "forged",
  "probability": 0.89,
  "confidence": 0.89,
  "filename": "document.jpg",
  "heatmap": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "tampered_regions": "data:image/png;base64,...",
  "user_id": "auth0|507f1f77bcf86cd799439011" // ← NEW
}
```

## 🚢 Production Deployment

When deploying:

### 1. Update Environment Variables

```bash
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_AUTH0_REDIRECT_URI=https://yourdomain.com
```

### 2. Update Auth0 Settings

Add production URLs to Auth0 Dashboard:

- Allowed Callback URLs
- Allowed Logout URLs
- Allowed Web Origins
- Allowed Origins (CORS)

### 3. Enable HTTPS

Auth0 requires HTTPS in production - no exceptions!

## 📚 Full Documentation

For detailed setup instructions, see:

- **`AUTH0_FRONTEND_SETUP.md`** - Complete setup guide
- **Backend**: `server/docs/AUTH0_SETUP_GUIDE.md` - Backend setup

## ✨ Next Steps

1. ✅ Install dependencies: `npm install` (Already done!)
2. ⏳ Configure Auth0 Application URLs (Do this now!)
3. ⏳ Start the app: `npm run dev`
4. ⏳ Test login/logout functionality
5. ⏳ Test document upload with authentication

## 🎉 Summary

Your frontend is now fully integrated with Auth0! Users can:

- ✅ Log in with Auth0
- ✅ Upload documents (authenticated)
- ✅ Receive analysis results with user tracking
- ✅ Log out securely

All API requests automatically include JWT tokens, and the backend validates them before processing.

**Ready to test? Just configure the Auth0 Application URLs and start the dev server!**
