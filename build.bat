@echo off
echo ==========================================
echo  Building Autonomous Driving Demo (.exe)
echo ==========================================
echo.

set "PYI=pyinstaller"
set "PIP=pip"
if exist ".venv\Scripts\python.exe" (
    set "PYI=.venv\Scripts\python.exe -m PyInstaller"
    set "PIP=.venv\Scripts\python.exe -m pip"
)

%PYI% --version >nul 2>nul
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    %PIP% install pyinstaller
    echo.
)

echo Running PyInstaller...
%PYI% game.spec --distpath dist --workpath build --clean -y

if %errorlevel% neq 0 (
    echo.
    echo BUILD FAILED. Check the errors above.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo  Build complete!
echo  Output: dist\AutonomousDrivingDemo.exe
echo ==========================================
pause
