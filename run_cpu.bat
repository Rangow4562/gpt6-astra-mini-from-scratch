@echo off
rem Run any script inside the prepared virtual environment (CPU-friendly defaults).
rem Usage: run_cpu.bat scripts\train_pretrain.py --steps 150 ...
setlocal
cd /d "%~dp0"
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
%PYTHON% %*
endlocal
