@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Sistema de Monitoramento de EPI [Desktop]

if exist ".venv\Scripts\python.exe" goto INICIAR_APP

echo.
echo =======================================================================
echo [ATENCAO] Ambiente nao instalado!
echo =======================================================================
echo O ambiente virtual .venv ainda nao foi configurado.
echo Por favor, de dois cliques primeiro no arquivo:
echo   -^> instalar.bat
echo.
echo Ele instalara todas as dependencias e modelos necessarios automaticamente.
echo =======================================================================
echo.
pause
exit /b 1

:INICIAR_APP
call .venv\Scripts\activate.bat
echo =======================================================================
echo Iniciando Sistema de Monitoramento e Deteccao de EPI...
echo =======================================================================
echo.

python run_desktop.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [AVISO] O aplicativo encerrou com codigo de erro %ERRORLEVEL%.
    pause
)
