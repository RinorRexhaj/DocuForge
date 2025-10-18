# Write-Host with colors for better UX
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

Write-ColorOutput "====================================" "Cyan"
Write-ColorOutput "  DocuForge Client Setup Script" "Cyan"
Write-ColorOutput "====================================" "Cyan"
Write-Host ""

# Check if Node.js is installed
Write-ColorOutput "Checking Node.js installation..." "Yellow"
$nodeVersion = node --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "❌ Node.js is not installed!" "Red"
    Write-ColorOutput "Please install Node.js 18+ from https://nodejs.org/" "Red"
    exit 1
}
Write-ColorOutput "✅ Node.js version: $nodeVersion" "Green"
Write-Host ""

# Check if we're in the client directory
if (-not (Test-Path "package.json")) {
    Write-ColorOutput "❌ package.json not found!" "Red"
    Write-ColorOutput "Please run this script from the client directory" "Red"
    exit 1
}

# Install dependencies
Write-ColorOutput "Installing npm dependencies..." "Yellow"
npm install
if ($LASTEXITCODE -ne 0) {
    Write-ColorOutput "❌ npm install failed!" "Red"
    exit 1
}
Write-ColorOutput "✅ Dependencies installed successfully!" "Green"
Write-Host ""

# Setup environment file
Write-ColorOutput "Setting up environment configuration..." "Yellow"
if (-not (Test-Path ".env.local")) {
    if (Test-Path ".env.local.example") {
        Copy-Item ".env.local.example" ".env.local"
        Write-ColorOutput "✅ Created .env.local from example" "Green"
        Write-ColorOutput "⚠️  Please update NEXT_PUBLIC_API_URL in .env.local if needed" "Yellow"
    } else {
        Write-ColorOutput "⚠️  .env.local.example not found, creating default .env.local" "Yellow"
        "NEXT_PUBLIC_API_URL=http://localhost:8000" | Out-File -FilePath ".env.local" -Encoding UTF8
        Write-ColorOutput "✅ Created .env.local with default settings" "Green"
    }
} else {
    Write-ColorOutput "✅ .env.local already exists" "Green"
}
Write-Host ""

# Display next steps
Write-ColorOutput "====================================" "Cyan"
Write-ColorOutput "  Setup Complete! 🎉" "Green"
Write-ColorOutput "====================================" "Cyan"
Write-Host ""
Write-ColorOutput "Next Steps:" "Cyan"
Write-ColorOutput "1. Start the development server:" "White"
Write-ColorOutput "   npm run dev" "Yellow"
Write-Host ""
Write-ColorOutput "2. Open your browser:" "White"
Write-ColorOutput "   http://localhost:3000" "Yellow"
Write-Host ""
Write-ColorOutput "3. Optional: Start the backend server:" "White"
Write-ColorOutput "   cd ../server" "Yellow"
Write-ColorOutput "   python api/start_server.py" "Yellow"
Write-Host ""
Write-ColorOutput "Tip: Enable 'Use mock data' in the UI to test without backend" "Cyan"
Write-Host ""
Write-ColorOutput "For more information, see README.md or SETUP.md" "White"
Write-Host ""
