@echo off
cd /d "%~dp0"
echo Checking dashboard dependencies...
py -c "import streamlit, pandas, numpy, plotly" >nul 2>&1
if errorlevel 1 (
  echo Installing required packages. This may take a few minutes the first time...
  py -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Dependency installation failed. Please check Python and your internet connection.
    pause
    exit /b 1
  )
)
echo Starting SupeRR Selector dashboard...
py -m streamlit run suprr_dashboard.py
pause
