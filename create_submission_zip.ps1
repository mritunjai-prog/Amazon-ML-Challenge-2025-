# Create Submission ZIP File for ML Challenge 2025
# This script packages all necessary files for competition submission

Write-Host "=" -ForegroundColor Cyan
Write-Host "🎯 Creating ML Challenge 2025 Submission Package" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan

# Define output zip file name
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$zipFileName = "ML_Challenge_2025_Submission_$timestamp.zip"
$zipPath = Join-Path $PSScriptRoot $zipFileName

# Files to include in submission
$filesToInclude = @(
    "solution.ipynb",
    "METHODOLOGY_REPORT.md",
    "README.md",
    "requirements.txt",
    "sample_code.py",
    "Documentation_template.md",
    "dataset\train.csv",
    "dataset\test.csv",
    "dataset\sample_test.csv",
    "dataset\sample_test_out.csv",
    "src\utils.py"
)

# Check if test_out.csv exists (generated after running notebook)
$testOutPath = Join-Path $PSScriptRoot "dataset\test_out.csv"
if (Test-Path $testOutPath) {
    $filesToInclude += "dataset\test_out.csv"
    Write-Host "✓ Found test_out.csv - including predictions" -ForegroundColor Green
} else {
    Write-Host "⚠ test_out.csv not found - run notebook first to generate predictions" -ForegroundColor Yellow
}

# Create temporary directory for clean structure
$tempDir = Join-Path $PSScriptRoot "temp_submission"
if (Test-Path $tempDir) {
    Remove-Item -Path $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null
New-Item -ItemType Directory -Path "$tempDir\dataset" | Out-Null
New-Item -ItemType Directory -Path "$tempDir\src" | Out-Null

Write-Host "`n📦 Copying files..." -ForegroundColor Cyan

# Copy files to temp directory
foreach ($file in $filesToInclude) {
    $sourcePath = Join-Path $PSScriptRoot $file
    if (Test-Path $sourcePath) {
        $destPath = Join-Path $tempDir $file
        Copy-Item -Path $sourcePath -Destination $destPath -Force
        Write-Host "  ✓ $file" -ForegroundColor Gray
    } else {
        Write-Host "  ⚠ $file not found (skipping)" -ForegroundColor Yellow
    }
}

# Create the ZIP file
Write-Host "`n🗜️ Creating ZIP archive..." -ForegroundColor Cyan
if (Test-Path $zipPath) {
    Remove-Item -Path $zipPath -Force
}

Compress-Archive -Path "$tempDir\*" -DestinationPath $zipPath -CompressionLevel Optimal

# Clean up temp directory
Remove-Item -Path $tempDir -Recurse -Force

# Display summary
Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "✅ SUBMISSION PACKAGE CREATED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green
Write-Host "`nZip File: $zipFileName" -ForegroundColor White
Write-Host "Location: $PSScriptRoot" -ForegroundColor White
$zipSize = (Get-Item $zipPath).Length / 1MB
Write-Host "Size: $([math]::Round($zipSize, 2)) MB" -ForegroundColor White

Write-Host "`n📋 Package Contents:" -ForegroundColor Cyan
Write-Host "  • Solution notebook (solution.ipynb)" -ForegroundColor Gray
Write-Host "  • Methodology documentation (METHODOLOGY_REPORT.md)" -ForegroundColor Gray
Write-Host "  • Dataset files (train.csv, test.csv)" -ForegroundColor Gray
Write-Host "  • Predictions (test_out.csv) - if generated" -ForegroundColor Gray
Write-Host "  • Supporting files (README, requirements, utils)" -ForegroundColor Gray

Write-Host "`n🎯 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Run solution.ipynb to generate predictions" -ForegroundColor White
Write-Host "  2. Upload test_out.csv to competition portal" -ForegroundColor White
Write-Host "  3. Upload METHODOLOGY_REPORT.md as documentation" -ForegroundColor White

Write-Host "`n" + "="*60 -ForegroundColor Green
Write-Host "Good luck! 🚀" -ForegroundColor Yellow
Write-Host "="*60 -ForegroundColor Green
