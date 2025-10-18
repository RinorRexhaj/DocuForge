# DocuForge Client - Quick Setup Guide

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```powershell
cd client
npm install
```

This will install:

- Next.js 14 (React framework)
- TypeScript (type safety)
- Tailwind CSS (styling)
- Axios (API calls)
- React Dropzone (file uploads)
- Lucide React (icons)

### Step 2: Configure Environment

```powershell
# Copy the example environment file
copy .env.local.example .env.local
```

**Edit `.env.local`** and set your backend URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Step 3: Start Development Server

```powershell
npm run dev
```

The app will be available at **http://localhost:3000**

## 🧪 Testing Without Backend

1. Upload an image
2. Check "Use mock data (for testing without backend)"
3. Click "Analyze for Tampering"

This generates dummy results to test the UI.

## 🏗️ Production Build

```powershell
# Build for production
npm run build

# Test the production build locally
npm start
```

## 📋 Checklist

- [ ] Node.js 18+ installed
- [ ] Dependencies installed (`npm install`)
- [ ] Environment configured (`.env.local`)
- [ ] Development server running (`npm run dev`)
- [ ] App accessible at http://localhost:3000
- [ ] Backend API running (or mock mode enabled)

## 🔗 Backend Setup

Make sure the backend server is running:

```powershell
cd ../server
python api/start_server.py
```

The backend should be accessible at `http://localhost:8000`

## 🎯 What You Get

### Page Features

- ✅ Modern, clean UI with dark mode
- ✅ Drag & drop file upload
- ✅ Real-time image analysis
- ✅ Interactive results viewer (3 tabs)
- ✅ Analysis history sidebar
- ✅ Downloadable results
- ✅ Mobile responsive

### File Uploads

- Supported formats: JPG, PNG
- Max size: 10MB
- Drag & drop or click to browse
- Image preview before analysis

### Results Display

1. **Heatmap Overlay**: Tampering probability visualization
2. **Tampered Mask**: Binary mask of detected areas
3. **Tampered Regions**: Isolated suspicious regions

## 🛠️ Troubleshooting

### Port Already in Use

If port 3000 is occupied:

```powershell
# Next.js will automatically try port 3001, 3002, etc.
# Or specify a custom port:
$env:PORT=3001; npm run dev
```

### Cannot Connect to Backend

1. Verify backend is running: `http://localhost:8000/docs`
2. Check `.env.local` has correct URL
3. Enable "Use mock data" to test UI

### Module Not Found

```powershell
# Clear cache and reinstall
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install
```

## 📱 Mobile Testing

The app is fully responsive. To test on mobile:

1. Find your local IP: `ipconfig`
2. Access via `http://YOUR_IP:3000` on mobile device
3. Ensure mobile and PC are on same network

## 🎨 UI Components

Built with reusable components:

- **Button**: Primary, secondary, outline, ghost variants
- **Card**: Container with shadow and border
- **Tabs**: Interactive tab switcher
- **LoadingSpinner**: Animated loading state

## 🔒 Security Notes

- File uploads are validated client-side (type & size)
- All API calls use proper error handling
- No sensitive data stored in browser
- CORS configured for local development

## 📊 Performance

- Initial load: < 2s
- File upload: Instant
- Analysis: Depends on backend (typically 2-5s)
- Results display: < 100ms

## 🌙 Dark Mode

Toggle dark mode using the moon/sun icon in the header.

Theme preference is stored in component state (resets on refresh).

## 🎓 Learning Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [TypeScript Guide](https://www.typescriptlang.org/docs)
- [Tailwind CSS](https://tailwindcss.com/docs)

## ✨ Next Steps

1. Upload a test image
2. Analyze it (with or without backend)
3. View results in all three tabs
4. Download result images
5. Check analysis history

---

**Need Help?** Check the main README.md or backend API documentation.
