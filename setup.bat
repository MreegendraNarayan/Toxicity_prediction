@echo off
REM Quick Setup and Run Script for Toxicity Prediction Project (Windows)

echo.
echo ============================================
echo 🧬 Toxicity Prediction - Setup and Run
echo ============================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set "python_version=%%i"
echo ✅ Using: %python_version%
echo.

REM Create virtual environment (optional)
set /p create_venv="Create a new virtual environment? (y/n): "
if /i "%create_venv%"=="y" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment created and activated
)

echo.
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo.
echo ============================================
echo Setup Complete! Choose how to run:
echo ============================================
echo.
echo 1) Run Streamlit Web App (Recommended)
echo    Command: streamlit run app.py
echo.
echo 2) Run Training Script
echo    Command: python src/train.py
echo.
echo 3) Run Jupyter Notebooks
echo    Command: jupyter lab notebooks/
echo.
echo 4) Install Package in Development Mode
echo    Command: pip install -e .
echo.

set /p option="Select option (1-4): "

if "%option%"=="1" (
    echo Launching Streamlit app...
    streamlit run app.py
) else if "%option%"=="2" (
    echo Starting training...
    python src/train.py --config src/config.yaml
) else if "%option%"=="3" (
    echo Launching Jupyter Lab...
    jupyter lab notebooks/
) else if "%option%"=="4" (
    echo Installing package in development mode...
    pip install -e .
    echo ✅ Package installed. You can now import: from toxicity import ...
) else (
    echo Invalid option
)

pause
