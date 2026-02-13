@echo off
echo ==========================================
echo  Building Autonomous Driving Demo (.exe)
echo ==========================================
echo.

where pyinstaller >nul 2>nul
if %errorlevel% neq 0 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller
    echo.
)

echo Running PyInstaller...
pyinstaller game.spec --distpath dist --workpath build --clean -y

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
