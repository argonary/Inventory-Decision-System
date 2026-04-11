$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$root`"; .venv\Scripts\Activate.ps1; uvicorn api.main:app"

Start-Sleep -Seconds 3

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$root`"; .venv\Scripts\Activate.ps1; streamlit run ui/app.py"
