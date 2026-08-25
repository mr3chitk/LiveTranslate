@echo off
cd /d "%~dp0"
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\.livetranslate-ready" (
    echo [ERROR] Virtual environment setup is incomplete.
    echo Please run install.bat again to finish installing and verifying dependencies.
    echo.
    pause
    exit /b 1
)

set "TRANSLATOR_PATH=C:\Users\PC\LLM-simple\4.translation.bat"
if exist "%TRANSLATOR_PATH%" (
    echo [LLM] Starting Translator...
    start /min "Translator" cmd /c "%TRANSLATOR_PATH%"
)

echo Starting LiveTranslate...
.venv\Scripts\python.exe main.py
if errorlevel 1 (
    echo.
    echo [ERROR] LiveTranslate exited with an error.
    pause
)
