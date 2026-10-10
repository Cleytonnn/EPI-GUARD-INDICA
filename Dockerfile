# Usa imagem com CUDA e Python
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Evitar prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instalar Python e dependências do sistema
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Criar link python -> python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Diretório de trabalho
WORKDIR /app

# Copiar requirements e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalar PyTorch com CUDA
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Copiar projeto
COPY . .

# Criar diretórios necessários
RUN mkdir -p alertas models

# Porta da API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando padrão: iniciar API
CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]
