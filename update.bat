@echo off
cd /d "%~dp0"
set PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%PATH%

echo ========================================
echo   LiveTranslate Updater
echo ========================================
echo.

:: Check git
git --version >nul 2>&1
if errorlevel 1 (
    echo Git not found, attempting to install via winget...
    winget --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Git not found and winget is not available.
        echo Please install Git from https://git-scm.com/downloads
        pause
        exit /b 1
    )
    winget install Git.Git --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [ERROR] Git installation failed.
        pause
        exit /b 1
    )
    :: Refresh PATH
    set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Links;%ProgramFiles%\Git\cmd;%PATH%"
    git --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Git installed but not found in PATH. Please restart and try again.
        pause
        exit /b 1
    )
    echo Git installed successfully.
    echo.
)

:: Pull latest code
echo Pulling latest changes...
git pull
if errorlevel 1 (
    echo.
    echo [ERROR] git pull failed. Check for local conflicts.
    pause
    exit /b 1
)

:: Check venv
if not exist ".venv\Scripts\pip.exe" (
    echo.
    echo Virtual environment not found, running install.bat...
    call install.bat
    exit /b %errorlevel%
)

:: Update dependencies
echo.
echo Updating dependencies...
:: Keep pip's download cache and temp files off the system drive
set "PIP_CACHE_DIR=%~dp0.pip-cache"
set "TMP=%~dp0.tmp"
set "TEMP=%~dp0.tmp"
if not exist "%~dp0.tmp" mkdir "%~dp0.tmp"
del /q ".venv\.livetranslate-ready" >nul 2>&1
.venv\Scripts\pip.exe install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Failed to update dependencies.
    pause
    exit /b 1
)

.venv\Scripts\pip.exe install "yasbd-lib>=0.15,<1.0" --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install yasbd-lib.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -m pip check
if errorlevel 1 (
    echo [ERROR] Installed dependencies are inconsistent.
    pause
    exit /b 1
)

> ".venv\.livetranslate-ready" echo %DATE% %TIME%

rd /s /q "%~dp0.pip-cache" >nul 2>&1
rd /s /q "%~dp0.tmp" >nul 2>&1

echo.
echo ========================================
echo   Update complete!
echo ========================================
echo.
echo Double-click start.bat to launch.
echo.
pause
