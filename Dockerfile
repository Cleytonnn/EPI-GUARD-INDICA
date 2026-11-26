# Usa imagem com CUDA e Python
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Evitar prompts interativos e arquivos .pyc; logs sem buffer
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Python 3.10 do próprio Ubuntu 22.04 (atende ao requisito >=3.10).
# Antes: instalava python3.11 mas o "pip" do pacote python3-pip pertence ao 3.10,
# então as dependências iam para um interpretador e o CMD rodava em outro (ImportError).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# PyTorch com CUDA ANTES do requirements: senão o ultralytics puxa o torch genérico do PyPI
# (download de vários GB duplicado e possível versão sem CUDA)
RUN python -m pip install --upgrade pip \
    && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Copiar projeto (o .dockerignore evita levar .env, .venv, datasets e gravações para a imagem)
COPY . .

# Usuário sem privilégios de root
RUN useradd --create-home --uid 1000 epiguard \
    && mkdir -p alertas models gravacoes datasets \
    && chown -R epiguard:epiguard /app
USER epiguard

# Porta da API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando padrão: iniciar API (0.0.0.0 é necessário dentro do contêiner)
CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "8000"]
