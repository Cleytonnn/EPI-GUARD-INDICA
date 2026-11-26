# Usa imagem com CUDA e Python
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Evitar prompts interativos e bytecode solto
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Instalar Python 3.11 com venv e bibliotecas gráficas mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Criar virtualenv dedicado em Python 3.11 para garantir python == pip
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Criar usuário não-root
RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home appuser

# Diretório de trabalho
WORKDIR /app

# Instalar dependências com o pip do Python 3.11
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copiar projeto e ajustar permissões para usuário não-root
COPY --chown=appuser:appuser . .
RUN mkdir -p alertas models gravacoes datasets \
    && chown -R appuser:appuser /app

USER appuser

# Porta da API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Comando padrão: iniciar API
CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]
