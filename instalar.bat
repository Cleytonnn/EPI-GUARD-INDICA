@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Instalador - Sistema de Monitoramento e Deteccao de EPI

echo ===============================================================================
echo        INSTALADOR AUTOMATICO - SISTEMA DE DETECCAO DE EPI [NR-10 / NR-6]
echo ===============================================================================
echo.

:: 1. Verificar instalacao do Python
echo [1/6] Verificando instalacao do Python...
set "PYTHON_CMD="
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 set "PYTHON_CMD=python"

if "%PYTHON_CMD%"=="" (
    where py >nul 2>&1
    if %ERRORLEVEL% EQU 0 set "PYTHON_CMD=py -3"
)

if "%PYTHON_CMD%"=="" goto ERRO_SEM_PYTHON

%PYTHON_CMD% --version
if %ERRORLEVEL% NEQ 0 goto ERRO_VERSAO_PYTHON
echo [OK] Python detectado com sucesso.
echo.
goto CRIAR_VENV

:ERRO_SEM_PYTHON
echo.
echo =======================================================================
echo [ERRO] Python nao foi encontrado no sistema!
echo =======================================================================
echo Para executar este sistema, voce precisa ter o Python 3.10 ou superior.
echo.
echo 1. Baixe o instalador oficial em: https://www.python.org/downloads/
echo 2. ATENCAO: Na primeira tela do instalador, marque a caixa:
echo    "[X] Add python.exe to PATH"
echo 3. Apos instalar o Python, execute este instalar.bat novamente.
echo =======================================================================
echo.
pause
exit /b 1

:ERRO_VERSAO_PYTHON
echo [ERRO] Falha ao verificar a versao do Python.
pause
exit /b 1

:: 2. Criar ambiente virtual isolado (.venv)
:CRIAR_VENV
echo [2/6] Configurando ambiente virtual .venv...
if exist ".venv\Scripts\python.exe" goto VENV_JA_EXISTE

echo Criando ambiente virtual com acesso a pacotes locais...
%PYTHON_CMD% -m venv --system-site-packages .venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Falha ao criar o ambiente virtual .venv.
    pause
    exit /b 1
)
echo [OK] Ambiente virtual criado com sucesso.
goto ATIVAR_VENV

:VENV_JA_EXISTE
echo [OK] Ambiente virtual .venv ja existe.

:: 3. Ativar ambiente virtual
:ATIVAR_VENV
echo.
echo [3/6] Ativando ambiente virtual...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Falha ao ativar o ambiente virtual.
    pause
    exit /b 1
)
echo [OK] Ambiente virtual ativado.
echo.

:: 4. Atualizar pip, setuptools e wheel
echo [4/6] Atualizando pip e ferramentas basicas...
python -m pip install --upgrade pip setuptools wheel --quiet
echo [OK] Ferramentas de pacote atualizadas.
echo.

:: 5. Instalar PyTorch com suporte a Aceleracao por GPU CUDA se houver placa NVIDIA
echo [5/6] Verificando hardware e suporte a Inteligencia Artificial...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 goto VERIFICAR_NVIDIA
goto VERIFICAR_CPU

:VERIFICAR_NVIDIA
echo [HARDWARE] Placa de video NVIDIA detectada!
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] PyTorch com aceleracao GPU CUDA ja esta funcional.
    goto INSTALAR_REQUISITOS
)

echo Instalando PyTorch com aceleracao GPU CUDA - aguarde o download...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if %ERRORLEVEL% NEQ 0 (
    echo [AVISO] Tentando canal alternativo CUDA 12.4...
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)
goto INSTALAR_REQUISITOS

:VERIFICAR_CPU
echo [HARDWARE] Modo padrao CPU.
python -c "import torch" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] PyTorch ja esta instalado.
    goto INSTALAR_REQUISITOS
)
echo Instalando PyTorch para processador CPU...
python -m pip install torch torchvision
goto INSTALAR_REQUISITOS

:INSTALAR_REQUISITOS
echo.
echo Instalando dependencias do projeto requirements.txt...
python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERRO] Ocorreu uma falha ao instalar as dependencias.
    pause
    exit /b 1
)
echo [OK] Todas as dependencias foram instaladas com sucesso.
echo.

:: 6. Baixar modelo base YOLO-Pose se nao existir
echo [6/6] Verificando modelos neurais de Pose e EPI...
if exist "yolov8n-pose.pt" goto MODELO_OK
echo Baixando modelo base yolov8n-pose.pt...
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
goto VALIDACAO

:MODELO_OK
echo [OK] Modelo yolov8n-pose.pt ja presente no diretorio.

:: Validacao final
:VALIDACAO
echo.
echo ===============================================================================
echo                         VALIDACAO DO SISTEMA
echo ===============================================================================
python -c "import torch; dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'PyTorch {torch.__version__} | CUDA Ativo: {torch.cuda.is_available()} | Dispositivo: {dev}')"
python -c "import cv2, customtkinter, ultralytics; print('Modulos essenciais OpenCV, CustomTkinter, YOLO: OK!')"

echo.
echo ===============================================================================
echo                     TUDO PRONTO COM SUCESSO!
echo ===============================================================================
echo Para iniciar a aplicacao desktop:
echo   -^> De dois cliques no arquivo: iniciar.bat
echo.
echo Para iniciar a API REST [FastAPI / Swagger]:
echo   -^> De dois cliques no arquivo: iniciar_api.bat
echo.
echo Para rodar a suite de testes:
echo   -^> De dois cliques no arquivo: testar.bat
echo ===============================================================================
echo.

pause
