@echo off

:: Quick download custom CT2 models
:: model_custom_download.bat [model_id] [hf/ms]
:: model_custom_download.bat Jim6789/whisper-ja-anime-v0.3-ct2 hf
:: model_custom_download.bat efwkjn/faster-whisper-ja-760M hf
:: model_custom_download.bat TransWithAI/whisper-ja-1.5B-ct2 hf

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found.
    echo Please run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

.venv\Scripts\python.exe  -c "import sys; from model_manager import download_asr_direct; download_asr_direct(sys.argv[1],sys.argv[2])" %1 %2