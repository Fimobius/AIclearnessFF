@echo off
title AIClear
cd /d "%~dp0"
"%~dp0runtime\python.exe" -X utf8 "%~dp0launcher.py"
if errorlevel 1 pause
