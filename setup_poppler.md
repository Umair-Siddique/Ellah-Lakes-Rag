# Setting up Poppler for PDF OCR

Your PDFs have broken text encoding (glyph codes like `/g38/g75/g88`), so OCR is required.
OCR with Tesseract needs Poppler to convert PDF pages to images first.

## Quick Setup for Windows:

### Step 1: Download Poppler
1. Go to: https://github.com/oschwartz10612/poppler-windows/releases
2. Download the latest release (e.g., `Release-24.08.0-0.zip`)

### Step 2: Extract
1. Extract the ZIP file to `C:\poppler` (or any location you prefer)
2. You should see folders like `C:\poppler\Library\bin\`

### Step 3: Set Environment Variable
Open PowerShell (as Administrator) and run:

```powershell
[System.Environment]::SetEnvironmentVariable('POPPLER_PATH', 'C:\poppler\Library\bin', [System.EnvironmentVariableTarget]::User)
```

**Important:** Replace `C:\poppler\Library\bin` with your actual bin folder path!

### Step 4: Restart Terminal
Close and reopen your PowerShell/terminal for the environment variable to take effect.

### Step 5: Verify Installation
In your new terminal, run:
```powershell
$env:POPPLER_PATH
```
It should display your Poppler bin path.

### Step 6: Run the Script
```bash
python -m preprocess.financial_statements
```

Now OCR will work and extract readable text from your PDFs!

---

## Alternative: Add to System PATH

Instead of setting POPPLER_PATH, you can add Poppler to your system PATH:

1. Open "Edit the system environment variables" from Windows search
2. Click "Environment Variables"
3. Under "User variables", find "Path" and click "Edit"
4. Click "New" and add: `C:\poppler\Library\bin`
5. Click OK on all windows
6. Restart your terminal

Then run the script again.

