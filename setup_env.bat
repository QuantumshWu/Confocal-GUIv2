@echo off
setlocal
rem --- repo root ---
set "ROOT=%~dp0"
pushd "%ROOT%"

rem 1) create venv if missing
if not exist ".venv\Scripts\python.exe" (
  echo [+] Creating virtual environment .venv ...
  python -m venv .venv
)

set "PY=%ROOT%.venv\Scripts\python.exe"

rem 2) ensure pip
echo [+] Bootstrapping pip ...
"%PY%" -m ensurepip --upgrade
"%PY%" -m pip install -U pip setuptools wheel

rem 3) install your deps (wheels preferred to avoid building)
echo [+] Installing project requirements ...
"%PY%" -m pip install --only-binary=:all: -r requirements.txt

rem 4) jupyter components
echo [+] Installing Jupyter components ...
"%PY%" -m pip install jupyterlab ipykernel

rem 5) register exactly ONE kernel inside venv:
rem    internal name = python3 (so it becomes the default),
rem    display name = confocal_gui_env (this is what UI shows)
echo [+] Registering kernel as default ...
"%PY%" -m ipykernel install --prefix "%ROOT%.venv" --name "python3" --display-name "confocal_gui_env"

echo.
echo Done. Start via runjupyterlab.bat
pause