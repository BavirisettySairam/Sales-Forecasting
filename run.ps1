Write-Host "Starting FastAPI Backend..." -ForegroundColor Cyan
Start-Process -NoNewWindow -FilePath ".\gcc_env\python.exe" -ArgumentList "-m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

Write-Host "Starting Streamlit Dashboard..." -ForegroundColor Magenta
.\gcc_env\python.exe -m streamlit run src/dashboard/app.py --server.port 8501
