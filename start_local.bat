@echo off
setlocal EnableExtensions

cd /d "%~dp0"
title OmniVoice Reader Local Server

set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=

set "PYTHON_BIN=%AUDIOBOOKSTUDIO_PYTHON%"

if not defined PYTHON_BIN (
  if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "PYTHON_BIN=%CONDA_PREFIX%\python.exe"
  )
)

if not defined PYTHON_BIN (
  if exist "I:\conda_envs\omnivoice\python.exe" (
    set "PYTHON_BIN=I:\conda_envs\omnivoice\python.exe"
  )
)

if not defined PYTHON_BIN (
  if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=%CD%\.venv\Scripts\python.exe"
  )
)

if not defined PYTHON_BIN (
  set "PYTHON_BIN=python"
)

set "HOST=%HOST%"
if not defined HOST set "HOST=127.0.0.1"
set "PORT=%PORT%"
if not defined PORT set "PORT=8000"

echo [OmniVoice Reader]
echo Workspace: %CD%
echo Python: %PYTHON_BIN%
echo Host: %HOST%
echo Port: %PORT%
echo.

where "%PYTHON_BIN%" >nul 2>nul
if errorlevel 1 (
  if not exist "%PYTHON_BIN%" (
    echo [Error] Python not found: %PYTHON_BIN%
    echo Set AUDIOBOOKSTUDIO_PYTHON or install the omnivoice conda environment.
    echo.
    pause
    exit /b 1
  )
)

"%PYTHON_BIN%" -X utf8 start_local.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
  echo Service exited normally.
  pause
  exit /b 0
)

echo [Error] start_local.py exited with code %EXIT_CODE%.
echo If you see "address already in use", change PORT or stop the existing process.
echo.
pause
exit /b %EXIT_CODE%
