@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Servidor de API - Deteccao de EPI [FastAPI]

if exist ".venv\Scripts\python.exe" goto INICIAR_API

echo.
echo =======================================================================
echo [ATENCAO] Ambiente nao instalado!
echo =======================================================================
echo O ambiente virtual .venv ainda nao foi configurado.
echo Execute instalar.bat primeiro.
echo =======================================================================
echo.
pause
exit /b 1

:INICIAR_API
call .venv\Scripts\activate.bat
echo =======================================================================
echo Iniciando Servidor de API FastAPI em http://localhost:8000
echo Documentacao Swagger disponivel em http://localhost:8000/docs
echo Pressione CTRL+C para encerrar o servidor.
echo =======================================================================
echo.

python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

if %ERRORLEVEL% NEQ 0 (
    echo.
    pause
)
