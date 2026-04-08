@echo off
setlocal

set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=

set PYTHON_BIN=%AUDIOBOOKSTUDIO_PYTHON%

if not defined PYTHON_BIN (
  if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\\python.exe" (
    set PYTHON_BIN=%CONDA_PREFIX%\\python.exe
  )
)

if not defined PYTHON_BIN (
  if exist I:\conda_envs\omnivoice\python.exe (
    set PYTHON_BIN=I:\conda_envs\omnivoice\python.exe
  )
)

if not defined PYTHON_BIN (
  if exist .venv\Scripts\python.exe (
    set PYTHON_BIN=.venv\Scripts\python.exe
  )
)

if not defined PYTHON_BIN (
  set PYTHON_BIN=python
)

"%PYTHON_BIN%" start_local.py
