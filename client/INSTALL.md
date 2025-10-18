# DocuForge Client - Complete Installation Guide

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Testing](#testing)
6. [Deployment](#deployment)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Node.js** (v18.0.0 or higher)

   - Download from: https://nodejs.org/
   - Verify installation:
     ```powershell
     node --version
     npm --version
     ```

2. **Git** (optional, for version control)

   - Download from: https://git-scm.com/

3. **Code Editor** (recommended)
   - VS Code: https://code.visualstudio.com/

### System Requirements

- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 500MB for node_modules and build files

---

## Installation Steps

### Option 1: Automated Setup (Recommended)

```powershell
# Navigate to client directory
cd c:\Users\PC\Desktop\Apps\DocuForge\client

# Run setup script
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

The script will:

- ✅ Check Node.js installation
- ✅ Install all dependencies
- ✅ Create environment configuration
- ✅ Display next steps

### Option 2: Manual Setup

#### Step 1: Navigate to Directory

```powershell
cd c:\Users\PC\Desktop\Apps\DocuForge\client
```

#### Step 2: Install Dependencies

```powershell
npm install
```

This installs:

- **next**: React framework (14.2.3)
- **react** & **react-dom**: UI library (18.3.1)
- **typescript**: Type safety (5.4.5)
- **tailwindcss**: Styling (3.4.3)
- **axios**: HTTP client (1.6.8)
- **react-dropzone**: File uploads (14.2.3)
- **zustand**: State management (4.5.2)
- **lucide-react**: Icons (0.378.0)
- **clsx**: Utility for classes (2.1.1)

#### Step 3: Environment Configuration

```powershell
# Copy example file
copy .env.local.example .env.local

# Edit with your preferred editor
notepad .env.local
```

Set the backend URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Configuration

### Environment Variables

Create `.env.local` in the client root:

```env
# Backend API URL (required)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Optional: Custom port (default is 3000)
PORT=3000

# Optional: Enable strict mode
NEXT_STRICT_MODE=true
```

### TypeScript Configuration

The `tsconfig.json` is pre-configured with:

- Strict type checking
- Path aliases (`@/` → root directory)
- ES2017 target
- Module resolution for Next.js

### Tailwind Configuration

The `tailwind.config.ts` includes:

- Custom color scheme
- Dark mode support (class-based)
- Custom animations
- Extended theme values

---

## Running the Application

### Development Mode

```powershell
npm run dev
```

**Output**:

```
  ▲ Next.js 14.2.3
  - Local:        http://localhost:3000
  - Ready in 2.5s
```

Access the app at: **http://localhost:3000**

**Features in Development Mode**:

- Hot module replacement (HMR)
- Fast refresh
- Detailed error messages
- Source maps enabled

### Production Mode

#### Build the Application

```powershell
npm run build
```

This creates an optimized production build in `.next/` directory.

#### Start Production Server

```powershell
npm start
```

Production mode includes:

- Minified JavaScript
- Optimized images
- Server-side rendering
- Static page generation

### Custom Port

```powershell
# Windows PowerShell
$env:PORT=3001; npm run dev

# Or edit package.json
"dev": "next dev -p 3001"
```

---

## Testing

### 1. Test Without Backend (Mock Mode)

1. Start the dev server: `npm run dev`
2. Open http://localhost:3000
3. Upload an image
4. Check "Use mock data"
5. Click "Analyze for Tampering"
6. View mock results

### 2. Test With Backend

#### Start Backend Server

```powershell
cd ../server
python api/start_server.py
```

Verify backend is running:

```powershell
curl http://localhost:8000/docs
```

#### Test Frontend

1. Start frontend: `npm run dev`
2. Upload an image
3. Uncheck "Use mock data"
4. Click "Analyze for Tampering"
5. View real analysis results

### 3. Test Responsive Design

**Desktop** (Default):

- Open http://localhost:3000

**Tablet**:

- Open browser DevTools (F12)
- Toggle device toolbar
- Select iPad or similar

**Mobile**:

- Select iPhone or similar device
- Test drag-and-drop
- Check scrolling and buttons

### 4. Test Dark Mode

1. Click the moon icon in header
2. Verify all components switch themes
3. Check color contrast
4. Test toggling back to light mode

---

## Deployment

### Vercel (Recommended)

1. **Install Vercel CLI**:

   ```powershell
   npm install -g vercel
   ```

2. **Deploy**:

   ```powershell
   vercel
   ```

3. **Set Environment Variables**:
   - Go to Vercel dashboard
   - Project → Settings → Environment Variables
   - Add `NEXT_PUBLIC_API_URL`

### Netlify

1. **Build Command**: `npm run build`
2. **Publish Directory**: `.next`
3. **Environment Variables**: Add `NEXT_PUBLIC_API_URL`

### Docker

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

Build and run:

```powershell
docker build -t docuforge-client .
docker run -p 3000:3000 -e NEXT_PUBLIC_API_URL=http://api-url docuforge-client
```

---

## Troubleshooting

### Issue: "Cannot find module 'next'"

**Solution**:

```powershell
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install
```

### Issue: Port 3000 already in use

**Solution 1** - Kill process:

```powershell
Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process
```

**Solution 2** - Use different port:

```powershell
$env:PORT=3001; npm run dev
```

### Issue: "Module not found: Can't resolve 'react-dropzone'"

**Solution**:

```powershell
npm install react-dropzone
```

### Issue: Backend connection failed

**Checklist**:

- [ ] Backend server is running (`http://localhost:8000`)
- [ ] `.env.local` has correct URL
- [ ] No firewall blocking connections
- [ ] CORS is configured in backend

**Test backend**:

```powershell
curl http://localhost:8000/api/analyze
```

### Issue: Styles not loading

**Solution**:

```powershell
# Clear Next.js cache
Remove-Item -Recurse -Force .next
npm run dev
```

### Issue: TypeScript errors

**Solution 1** - Check tsconfig:

```powershell
# Verify tsconfig.json exists
Get-Content tsconfig.json
```

**Solution 2** - Clear build cache:

```powershell
Remove-Item -Recurse -Force .next
Remove-Item next-env.d.ts
npm run dev
```

### Issue: Environment variables not working

**Important**: Environment variables in Next.js must:

1. Start with `NEXT_PUBLIC_` for client-side access
2. Be set before build time
3. Restart dev server after changes

**Solution**:

```powershell
# Kill dev server (Ctrl+C)
# Edit .env.local
# Restart
npm run dev
```

### Issue: Build fails

**Check**:

```powershell
# Clear cache
Remove-Item -Recurse -Force .next

# Reinstall dependencies
Remove-Item -Recurse -Force node_modules
npm install

# Try build again
npm run build
```

---

## Performance Optimization

### 1. Enable Production Mode

Always use `npm run build` + `npm start` for production.

### 2. Optimize Images

- Use WebP format when possible
- Compress images before upload
- Consider image CDN

### 3. Code Splitting

Next.js automatically code-splits. Keep components small and focused.

### 4. Caching

Configure caching headers in `next.config.js`:

```javascript
async headers() {
  return [
    {
      source: '/:all*(svg|jpg|png)',
      headers: [
        {
          key: 'Cache-Control',
          value: 'public, max-age=31536000, immutable',
        },
      ],
    },
  ]
}
```

---

## Updating Dependencies

### Check for updates:

```powershell
npm outdated
```

### Update all packages:

```powershell
npm update
```

### Update specific package:

```powershell
npm install next@latest
```

### Major version updates:

```powershell
npm install next@14 react@18
```

---

## Development Tips

### 1. Fast Refresh

Next.js preserves React state during edits. If broken:

```powershell
# Restart dev server
```

### 2. Error Overlay

Development mode shows detailed errors. Click to open in editor.

### 3. Network Panel

Check browser DevTools → Network to debug API calls.

### 4. React DevTools

Install browser extension for component inspection.

---

## Support & Resources

### Documentation

- Next.js: https://nextjs.org/docs
- React: https://react.dev
- TypeScript: https://typescriptlang.org/docs
- Tailwind: https://tailwindcss.com/docs

### Project Files

- Main README: `README.md`
- Setup Guide: `SETUP.md`
- Component Docs: `COMPONENTS.md`

### Common Commands

```powershell
npm run dev      # Start development server
npm run build    # Build for production
npm start        # Start production server
npm run lint     # Run ESLint
```

---

**Installation Complete!** 🎉

Next: Run `npm run dev` and open http://localhost:3000
