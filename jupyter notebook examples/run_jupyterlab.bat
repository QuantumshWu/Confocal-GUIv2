@echo off
setlocal
rem ----- current dir is examples/ -----
set "HERE=%~dp0"
set "ROOT=%HERE%\.."

set "VENV_PY=%ROOT%\.venv\Scripts\python.exe"
set "VENV_KSPEC=%ROOT%\.venv\share\jupyter"

if exist "%VENV_PY%" (
  echo [+] Using project venv
  set "JUPYTER_PATH=%VENV_KSPEC%"
  "%VENV_PY%" -m jupyterlab
) else (
  echo [!] .venv not found. Falling back to system JupyterLab...
  jupyter lab
)