# DocuForge Client - Component Documentation

## Component Overview

### Core Components

#### 1. FileUpload.tsx

**Purpose**: Handle image file uploads with drag-and-drop functionality

**Props**:

- `onAnalysisComplete: (result: AnalysisResult, fileName: string) => void` - Callback when analysis finishes
- `onAnalysisStart: () => void` - Callback when analysis begins
- `isLoading: boolean` - Loading state indicator

**Features**:

- Drag-and-drop file upload
- File type validation (JPG, PNG)
- File size validation (max 10MB)
- Image preview
- Mock data mode for testing
- Error handling and display

**Usage**:

```tsx
<FileUpload
  onAnalysisComplete={(result, fileName) => setResult(result)}
  onAnalysisStart={() => setLoading(true)}
  isLoading={isLoading}
/>
```

---

#### 2. ResultsViewer.tsx

**Purpose**: Display analysis results with interactive tabs

**Props**:

- `result: AnalysisResult` - Analysis results from backend

**Features**:

- Tabbed interface (Heatmap, Mask, Tampered Regions)
- Confidence score visualization
- Tampering verdict display
- Image download functionality
- Full-size image viewing
- Explanatory descriptions

**Usage**:

```tsx
<ResultsViewer result={analysisResult} />
```

---

#### 3. FileHistoryList.tsx

**Purpose**: Display history of analyzed files

**Props**:

- `files: UploadedFile[]` - Array of uploaded files
- `onReanalyze: (fileId: string) => void` - Callback for re-analysis

**Features**:

- Chronological file list
- Verdict badges (Tampered/Clean)
- Confidence percentages
- Re-analyze functionality
- Scrollable list

**Usage**:

```tsx
<FileHistoryList
  files={uploadedFiles}
  onReanalyze={(id) => reanalyzeFile(id)}
/>
```

---

### UI Components

#### Button.tsx

Reusable button component with variants

**Props**:

- `variant?: 'primary' | 'secondary' | 'outline' | 'ghost'` - Visual style
- `size?: 'sm' | 'md' | 'lg'` - Size variant
- `fullWidth?: boolean` - Full width button
- `disabled?: boolean` - Disabled state

**Variants**:

- **Primary**: Blue background, white text
- **Secondary**: Gray background
- **Outline**: Bordered button
- **Ghost**: Transparent button

---

#### Card.tsx

Container component with styling

**Props**:

- `children: ReactNode` - Content
- `className?: string` - Additional classes

**Features**:

- Rounded corners
- Shadow
- Dark mode support
- Border styling

---

#### Tabs.tsx

Tab navigation component

**Props**:

- `tabs: Tab[]` - Array of tab objects
- `activeTab: string` - Currently active tab ID
- `onChange: (tabId: string) => void` - Tab change handler

**Usage**:

```tsx
<Tabs
  tabs={[
    { id: "heatmap", label: "Heatmap" },
    { id: "mask", label: "Mask" },
  ]}
  activeTab={activeTab}
  onChange={setActiveTab}
/>
```

---

#### LoadingSpinner.tsx

Animated loading indicator

**Features**:

- Spinning animation
- Dark mode support
- Centered layout

---

## Type Definitions

### AnalysisResult

```typescript
interface AnalysisResult {
  probability: number; // Tampering probability (0-1)
  heatmap: string; // Base64 encoded image
  mask: string; // Base64 encoded image
  tampered_regions: string; // Base64 encoded image
  prediction?: string; // "Tampered" or "Clean"
}
```

### UploadedFile

```typescript
interface UploadedFile {
  id: string; // Unique identifier
  fileName: string; // Original filename
  timestamp: string; // ISO timestamp
  prediction: "Tampered" | "Clean";
  probability: number; // Confidence score
}
```

### ApiResponse

```typescript
interface ApiResponse {
  probability: number;
  heatmap: string;
  mask: string;
  tampered_regions: string;
  prediction: string;
  error?: string;
}
```

---

## Service Functions

### API Service (`services/api.ts`)

#### analyzeImage

```typescript
analyzeImage(file: File): Promise<ApiResponse>
```

Sends image to backend for analysis

**Parameters**:

- `file: File` - Image file to analyze

**Returns**: Promise with analysis results

**Throws**: Error if request fails

---

#### getMockAnalysisResult

```typescript
getMockAnalysisResult(file: File): Promise<ApiResponse>
```

Generates mock analysis results for testing

**Parameters**:

- `file: File` - Image file (for reference)

**Returns**: Promise with mock results

---

## Utility Functions

### helpers.ts

#### formatFileSize

```typescript
formatFileSize(bytes: number): string
```

Converts bytes to human-readable format

#### formatTimestamp

```typescript
formatTimestamp(timestamp: string | Date): string
```

Formats date/time for display

#### validateImageFile

```typescript
validateImageFile(file: File): { valid: boolean; error?: string }
```

Validates image file type and size

#### downloadBase64Image

```typescript
downloadBase64Image(base64Data: string, filename: string): void
```

Downloads base64 image as file

#### getConfidenceLevel

```typescript
getConfidenceLevel(probability: number): {
  level: 'low' | 'medium' | 'high'
  color: string
  description: string
}
```

Categorizes confidence level

---

## State Management

The application uses React hooks for state:

- **Local State**: `useState` for component-level state
- **File Upload**: Managed in FileUpload component
- **Analysis Results**: Passed via props from parent
- **History**: Array state in main page component

### State Flow

```
User uploads file
    ↓
FileUpload component
    ↓
API call (real or mock)
    ↓
Results returned
    ↓
Parent component updates state
    ↓
ResultsViewer displays results
    ↓
History list updated
```

---

## Styling Guidelines

### Tailwind Classes

**Layout**:

- Container: `max-w-7xl mx-auto px-4 sm:px-6 lg:px-8`
- Grid: `grid grid-cols-1 lg:grid-cols-3 gap-8`
- Flex: `flex items-center justify-between`

**Spacing**:

- Padding: `p-4`, `px-6`, `py-8`
- Margin: `mt-4`, `mb-6`, `space-y-4`

**Colors**:

- Primary: `bg-blue-600`, `text-blue-600`
- Success: `bg-green-600`, `text-green-600`
- Error: `bg-red-600`, `text-red-600`
- Neutral: `bg-gray-100`, `text-gray-900`

**Dark Mode**:

- Add `dark:` prefix: `dark:bg-gray-800`, `dark:text-white`

---

## Best Practices

1. **Error Handling**: Always wrap API calls in try-catch
2. **Loading States**: Show loading indicators during async operations
3. **Validation**: Validate file uploads client-side
4. **Accessibility**: Use semantic HTML and ARIA labels
5. **Responsive**: Test on mobile, tablet, desktop
6. **Performance**: Optimize images, lazy load when possible
7. **Type Safety**: Use TypeScript interfaces for all props

---

## Testing Checklist

- [ ] File upload (drag & drop)
- [ ] File upload (click to browse)
- [ ] File validation (wrong type)
- [ ] File validation (too large)
- [ ] Analysis with backend
- [ ] Analysis with mock data
- [ ] Results display (all tabs)
- [ ] Image downloads
- [ ] History list
- [ ] Dark mode toggle
- [ ] Mobile responsive
- [ ] Error handling

---

For more information, see the main README.md
