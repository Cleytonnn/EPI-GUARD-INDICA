@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Execucao da Suite de Testes Automatizados

if not exist ".venv\Scripts\python.exe" (
    echo [ATENCAO] Execute instalar.bat primeiro.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo =======================================================================
echo Suite de Testes Automatizados - CRIADO POR CLEYTON BAUMGRATZ
echo =======================================================================
echo Executando testes unitarios e de integracao...
python -m pytest tests/ -v
echo.
pause
