@echo off
rem Windows launcher for the Stable Audio 3 plug-in bootstrap.
rem
rem Delegates to bootstrap.ps1 (PowerShell gives us robust quoting and Python
rem here-docs that batch can't do cleanly). Arguments pass straight through:
rem   bootstrap.bat                 build the isolated .venv (no model download)
rem   bootstrap.bat --with-model    also download the gated SA3 Medium weights
rem
rem Overrides are read from the environment by bootstrap.ps1, e.g.:
rem   set SA3_CUDA=cu126 & set SA3_TORCH=2.7.1 & bootstrap.bat
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1" %*
exit /b %errorlevel%
