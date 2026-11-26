@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Servidor de API - Deteccao de EPI [FastAPI] - CRIADO POR CLEYTON BAUMGRATZ

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
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Falha ao ativar o ambiente virtual.
    pause
    exit /b 1
)
echo =======================================================================
echo Iniciando Servidor de API FastAPI em http://127.0.0.1:8000
echo Documentacao Swagger disponivel em http://127.0.0.1:8000/docs
echo CRIADO POR CLEYTON BAUMGRATZ
echo Pressione CTRL+C para encerrar o servidor.
echo =======================================================================
echo.

python -m src.api --host 127.0.0.1 --port 8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    pause
)
