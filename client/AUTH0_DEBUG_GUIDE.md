# Auth0 Login State Not Updating - Troubleshooting Guide

## What We Changed

### 1. Updated `Auth0ProviderWithNavigate.tsx`

- Added `router.refresh()` after login to force UI update
- Made audience optional (won't break if not set)
- Better error handling

### 2. Added Debug Components

- **`AuthDebug.tsx`** - Shows real-time auth status in bottom-right corner
- **Console logging** - Logs auth state changes to browser console
- **Added to page.tsx** - Visible on all screens

## How to Debug

### Step 1: Restart the Dev Server

```bash
# Stop current server (Ctrl+C in terminal)
npm run dev
```

### Step 2: Clear Browser Cache & Storage

1. Open **Developer Tools** (F12)
2. Go to **Application** tab
3. Under **Storage**, click "Clear site data"
4. Or manually delete:
   - Local Storage → `auth0spa`
   - Session Storage → all items
   - Cookies → all from `localhost:3000` and `auth0.com`

### Step 3: Test Login Flow

1. Open http://localhost:3000
2. Open **Console** tab in DevTools (F12)
3. Look for "🔍 Auth0 Debug Info" logs
4. Look at bottom-right corner for debug panel
5. Click "Login to Get Started"
6. Complete Auth0 login
7. Watch console and debug panel

## What to Look For

### In Console (F12 → Console)

```javascript
🔍 Auth0 Debug Info
  isAuthenticated: false/true
  isLoading: false/true
  user: { email: "...", ... } or undefined
  error: undefined or error message
```

### In Debug Panel (Bottom-Right)

```
🔐 Auth0 Debug
Loading: ✅ No / ⏳ Yes
Authenticated: ✅ Yes / ❌ No
User: email@example.com / None
```

## Common Issues & Solutions

### Issue 1: Stuck on "Loading: ⏳ Yes"

**Cause**: Auth0 is trying to get authentication state but failing
**Solution**:

1. Clear browser storage (see Step 2 above)
2. Check Auth0 callback URLs are configured
3. Verify environment variables are correct

### Issue 2: "Authenticated: ❌ No" after login

**Cause**: Auth0 redirect callback not working
**Solutions**:

1. **Check Callback URL in Auth0**:
   - Go to Auth0 Dashboard
   - Applications → Your App → Settings
   - Allowed Callback URLs: `http://localhost:3000`
2. **Check .env.local**:

   ```bash
   NEXT_PUBLIC_AUTH0_REDIRECT_URI=http://localhost:3000
   ```

   (Make sure it's EXACTLY `http://localhost:3000` with no trailing slash)

3. **Hard Refresh**: Ctrl+Shift+R or Cmd+Shift+R

### Issue 3: Error in console about audience

**Cause**: Auth0 API not configured or audience mismatch
**Solutions**:

**Option A: Create the API in Auth0**

1. Go to Auth0 Dashboard → Applications → APIs
2. Create API with identifier: `https://docuforge-api`
3. Make sure backend also uses this audience

**Option B: Remove audience temporarily** (for testing only)
Update `.env.local`:

```bash
# Comment out or remove this line
# NEXT_PUBLIC_AUTH0_AUDIENCE=https://docuforge-api
```

Then restart: `npm run dev`

### Issue 4: Page doesn't update after login

**Cause**: React state not refreshing
**Solution**: We added `router.refresh()` - restart server to apply changes

```bash
npm run dev
```

### Issue 5: Infinite redirect loop

**Cause**: Auth0 callback not completing
**Solutions**:

1. Clear browser storage
2. Check for errors in console
3. Verify redirect URI matches exactly in Auth0 settings

## Manual Testing Steps

### Test 1: Fresh Login

1. Clear all browser storage
2. Go to http://localhost:3000
3. Should see: "Authenticated: ❌ No"
4. Click "Login to Get Started"
5. Auth0 login page appears
6. Enter credentials
7. Redirected back to app
8. Should see: "Authenticated: ✅ Yes"
9. Should see: User email in debug panel
10. Should see: Dashboard with upload component

### Test 2: Refresh After Login

1. Login successfully (Test 1)
2. Refresh page (F5)
3. Should still show: "Authenticated: ✅ Yes"
4. Should still see: Dashboard (not welcome banner)

### Test 3: Logout

1. Login successfully
2. Click "Logout" button
3. Should see: "Authenticated: ❌ No"
4. Should see: Welcome banner (not dashboard)

## Quick Fixes Checklist

- [ ] Restart dev server: `npm run dev`
- [ ] Clear browser storage (Application → Clear site data)
- [ ] Hard refresh (Ctrl+Shift+R)
- [ ] Check console for errors
- [ ] Verify Auth0 callback URL: `http://localhost:3000` (exact match)
- [ ] Verify .env.local has correct values
- [ ] Check Auth0 Application Type: "Single Page Application"

## Environment Variables Verification

Your `.env.local` should have:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_AUTH0_DOMAIN=dev-aam3u7jezgeelchn.us.auth0.com
NEXT_PUBLIC_AUTH0_CLIENT_ID=0QjVH5EkTuU9Y8B2waXwAQFhByJvlSGx
NEXT_PUBLIC_AUTH0_REDIRECT_URI=http://localhost:3000
NEXT_PUBLIC_AUTH0_AUDIENCE=https://docuforge-api  # Optional - can comment out for testing
```

## Auth0 Dashboard Settings to Verify

### Application Settings

- **Application Type**: Single Page Application
- **Allowed Callback URLs**: `http://localhost:3000`
- **Allowed Logout URLs**: `http://localhost:3000`
- **Allowed Web Origins**: `http://localhost:3000`
- **Allowed Origins (CORS)**: `http://localhost:3000`

### API Settings (if using audience)

- **API exists** with identifier: `https://docuforge-api`
- **Application authorized** to access the API (APIs tab in Application)

## Advanced Debugging

### Check localStorage

1. Open DevTools (F12)
2. Application → Local Storage → http://localhost:3000
3. Look for keys starting with `@@auth0spajs@@`
4. Should contain authentication tokens after login

### Check Network Tab

1. Open DevTools → Network tab
2. Login
3. Look for requests to:
   - `dev-aam3u7jezgeelchn.us.auth0.com/authorize` (redirect to login)
   - `dev-aam3u7jezgeelchn.us.auth0.com/oauth/token` (get tokens)
4. Check for errors (red status codes)

### React DevTools (if installed)

1. Install React DevTools extension
2. Open Components tab
3. Find `Auth0Provider`
4. Check props and state

## Still Not Working?

If none of the above works, try this nuclear option:

### Complete Reset

```bash
# 1. Stop dev server
# 2. Delete .next folder
rm -rf .next

# 3. Clear npm cache
npm cache clean --force

# 4. Reinstall dependencies
rm -rf node_modules
npm install

# 5. Restart
npm run dev
```

Then:

1. Clear ALL browser data for localhost:3000
2. Use incognito/private window
3. Try login fresh

## Contact Info

If still having issues, share:

1. Console errors (screenshot or text)
2. Debug panel output
3. Network tab errors
4. Auth0 Dashboard settings (sanitized - no secrets!)

## Remove Debug Components (After Fixing)

Once login works, remove debug components:

### In `page.tsx`

Remove:

```tsx
import AuthDebug from "./components/AuthDebug";

// And remove all <AuthDebug /> instances
```

### Delete file

```bash
rm app/components/AuthDebug.tsx
```
