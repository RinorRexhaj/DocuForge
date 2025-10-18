# DocuForge Client - Frontend Application

A modern, responsive web application for document forgery detection built with Next.js, TypeScript, and Tailwind CSS.

![DocuForge](https://img.shields.io/badge/Next.js-14.2-black?logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.4-blue?logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-3.4-38bdf8?logo=tailwind-css)

## 🌟 Features

- **Drag & Drop Upload**: Intuitive file upload with visual feedback
- **Real-time Analysis**: Instant tampering detection results
- **Interactive Results Viewer**: Toggle between heatmap, mask, and tampered regions
- **Analysis History**: Track all analyzed images with timestamps
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile
- **Dark Mode Support**: Toggle between light and dark themes
- **Download Results**: Export analysis results as PNG images
- **Mock Mode**: Test the UI without a running backend

## 🚀 Getting Started

### Prerequisites

- Node.js 18.x or higher
- npm or yarn package manager

### Installation

1. **Navigate to the client directory**:
   ```bash
   cd client
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.local.example .env.local
   ```
   
   Edit `.env.local` and set your backend API URL:
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

4. **Start the development server**:
   ```bash
   npm run dev
   ```

5. **Open your browser**:
   Navigate to [http://localhost:3000](http://localhost:3000)

## 📦 Build for Production

```bash
# Build the application
npm run build

# Start the production server
npm start
```

## 🎨 Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Custom components with Lucide React icons
- **HTTP Client**: Axios
- **File Upload**: React Dropzone
- **State Management**: React Hooks (useState)

## 📁 Project Structure

```
client/
├── app/
│   ├── components/
│   │   ├── ui/                 # Reusable UI components
│   │   │   ├── Button.tsx
│   │   │   ├── Card.tsx
│   │   │   ├── Tabs.tsx
│   │   │   └── LoadingSpinner.tsx
│   │   ├── FileUpload.tsx      # Drag & drop upload component
│   │   ├── ResultsViewer.tsx   # Analysis results display
│   │   └── FileHistoryList.tsx # Upload history sidebar
│   ├── services/
│   │   └── api.ts              # Backend API integration
│   ├── types/
│   │   └── index.ts            # TypeScript type definitions
│   ├── layout.tsx              # Root layout with metadata
│   ├── page.tsx                # Main application page
│   └── globals.css             # Global styles & Tailwind
├── public/                     # Static assets
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── next.config.js
└── README.md
```

## 🔌 Backend Integration

The frontend expects the backend API to be running at the URL specified in `.env.local`.

### API Endpoint

**POST** `/api/analyze`

**Request**: `multipart/form-data` with image file

**Response**:
```json
{
  "probability": 0.92,
  "heatmap": "data:image/png;base64,...",
  "mask": "data:image/png;base64,...",
  "tampered_regions": "data:image/png;base64,...",
  "prediction": "Tampered"
}
```

### Testing Without Backend

Toggle "Use mock data" in the upload section to test the UI with simulated results.

## 🎯 Key Features Explained

### 1. File Upload Component
- Accepts JPG and PNG images up to 10MB
- Drag-and-drop or click to browse
- Image preview before analysis
- File size validation

### 2. Results Viewer
- **Heatmap**: Shows tampering probability across regions
- **Mask**: Binary representation of tampered areas
- **Tampered Regions**: Isolated view of suspicious areas
- Confidence score with visual progress bar
- Download options for all result images

### 3. Analysis History
- Chronological list of analyzed images
- Quick verdict display (Tampered/Clean)
- Confidence percentage
- Re-analyze capability

### 4. Dark Mode
- Toggle between light and dark themes
- Persists user preference
- Smooth theme transitions

## 🎨 Customization

### Updating Colors

Edit `tailwind.config.ts` to customize the color palette:

```typescript
theme: {
  extend: {
    colors: {
      primary: 'your-color-here',
      // ...
    }
  }
}
```

### Modifying API Endpoint

Update the API URL in `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://your-api-domain.com
```

## 🐛 Troubleshooting

### Module Not Found Errors

If you see errors about missing modules after installation:

```bash
# Delete node_modules and reinstall
rm -rf node_modules
npm install
```

### API Connection Failed

1. Verify the backend server is running
2. Check the API URL in `.env.local`
3. Enable mock mode to test UI functionality
4. Check browser console for detailed error messages

### Build Errors

Ensure you're using Node.js 18 or higher:

```bash
node --version
```

## 📱 Responsive Breakpoints

- **Mobile**: < 640px
- **Tablet**: 640px - 1024px
- **Desktop**: > 1024px

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the DocuForge document analysis suite.

## 🔗 Related

- **Backend API**: See `../server/` directory
- **API Documentation**: `../server/docs/API_README.md`

## 💡 Tips

- Use the mock data toggle to develop UI without backend
- Check the browser console for detailed API errors
- Analysis history is stored in component state (resets on page refresh)
- Large images may take longer to analyze

## 📞 Support

For issues or questions:
- Check the troubleshooting section
- Review backend API logs
- Verify network connectivity

---

Built with ❤️ using Next.js and TypeScript
