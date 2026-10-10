# 🛡️ Sistema Corporativo de Detecção de EPI em Tempo Real & Processamento em Lote

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012-EE4C2C.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11%20Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20%2F%20Hexagonal-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-success.svg)]()

Sistema empresarial de inteligência artificial e visão computacional para conformidade de **Equipamentos de Proteção Individual (EPI)**, projetado segundo os padrões de **Clean Architecture (Arquitetura Limpa)**, **Clean Code** e princípios **SOLID**.

Suporta tanto **fluxos de vídeo ao vivo (RTSP, Câmeras IP, Webcams)** quanto **processamento em lote/monitoramento de pastas de arquivos de vídeo (`.mp4`, `.avi`, `.mkv` e arquivos raw HEVC/H.265 `.265`)** organizados pela hierarquia logística de veículos e frotas (`[PLACA] > [DATA] > [VÍDEO]`).

---

## 🏗️ Arquitetura do Sistema (Clean Architecture / Hexagonal)

O projeto segue estritamente a separação em camadas independentes. Regras de negócio e entidades de domínio **não dependem de bibliotecas externas de IA, banco de dados ou frameworks web**. Detalhes tecnológicos (YOLO, PyAV, OpenCV, FastAPI) são adaptadores plugáveis através de interfaces (`Ports & Adapters`).

```mermaid
flowchart TD
    subgraph Presentation ["Camada de Apresentação (Presentation Layer)"]
        CLI["CLI Commands & Visualizer\n(src/presentation/cli)"]
        API["FastAPI REST Endpoints\n(src/presentation/api)"]
    end

    subgraph Container ["Inversão de Controle & DI (IoC)"]
        IoC["src/container.py\n(Dependency Injection Container)"]
    end

    subgraph Core ["Núcleo da Aplicação (Core Layer)"]
        subgraph UseCases ["Casos de Uso (Application / Use Cases)"]
            UC1["ProcessVideoFileUseCase"]
            UC2["ProcessBatchUseCase"]
            UC3["WatchDirectoryUseCase"]
            UC4["AnalyzeFrameUseCase"]
            UC5["ProcessStreamUseCase"]
        end

        subgraph Services ["Serviços de Domínio (Domain Services)"]
            CS["ComplianceService\n(Regras temporais de violação)"]
        end

        subgraph Interfaces ["Portas / Contratos (Interfaces / Ports)"]
            IDetector["IDetector"]
            IVideoReader["IVideoReader"]
            INotifier["INotifier"]
            IAlertStorage["IAlertStorage"]
            IProcessedRepo["IProcessedVideoRepository"]
            IReportRepo["IReportRepository"]
        end

        subgraph Domain ["Entidades & Objetos de Valor (Domain)"]
            Detection["Detection"]
            Violation["Violation"]
            VehicleMeta["VehicleMetadata"]
            PPEType["PPEType"]
        end
    end

    subgraph Infrastructure ["Camada de Infraestrutura (Infrastructure Layer)"]
        YOLO["Ultralytics YOLODetector"]
        Readers["UniversalVideoReader\n(OpenCV + PyAV Raw H.265)"]
        Notifiers["EmailNotifier & CompositeNotifier\n+ RateLimiter"]
        Storage["FileAlertStorage\n(Organização por Placa/Data)"]
        Repos["JsonProcessedRepository\n& CsvReportRepository"]
    end

    Presentation --> UseCases
    UseCases --> Services
    UseCases --> Interfaces
    Services --> Domain
    Interfaces --> Domain
    Infrastructure -.->|Implementa| Interfaces
    IoC --> Presentation
    IoC --> UseCases
    IoC --> Infrastructure
```

### Princípios SOLID Aplicados:
1. **Single Responsibility Principle (SRP):** Cada caso de uso, serviço e repositório possui uma responsabilidade única bem delineada.
2. **Open/Closed Principle (OCP):** Novos canais de alerta (Slack, WhatsApp, Webhooks) ou novos decodificadores de vídeo podem ser conectados sem alterar os casos de uso existentes.
3. **Liskov Substitution Principle (LSP):** Qualquer leitor de vídeo ou notificador que satisfaça o contrato de interface pode ser injetado de forma transparente.
4. **Interface Segregation Principle (ISP):** Interfaces finas e dedicadas (`IDetector`, `IVideoReader`, `INotifier`, `IAlertStorage`, `IProcessedVideoRepository`, `IReportRepository`).
5. **Dependency Inversion Principle (DIP):** O domínio de alto nível depende apenas de abstrações. Implementações concretas residem na camada de infraestrutura e são injetadas pelo `Container`.

---

## 📁 Estrutura de Diretórios

```
INDICADOR/
├── config/
│   └── settings.yaml             # Configurações centralizadas do sistema
├── models/
│   └── best.pt                   # Modelo treinado YOLO11 de EPIs (5.2 MB)
├── src/
│   ├── core/                     # NÚCLEO INDEPENDENTE DE FRAMEWORKS
│   │   ├── domain/               # Entidades, Value Objects e Enums
│   │   │   ├── detection.py      # Dataclass Detection e BoundingBox
│   │   │   ├── violation.py      # Entidade Violation
│   │   │   ├── vehicle_metadata.py # Extração e metadados de frota (Placa/Data/Arquivo)
│   │   │   └── ppe_type.py       # Enum de tipos de EPI e violação
│   │   ├── interfaces/           # Portas (Contratos e Protocolos abstratos)
│   │   │   ├── detector.py       # IDetector
│   │   │   ├── video_reader.py   # IVideoReader
│   │   │   ├── notifier.py       # INotifier
│   │   │   ├── alert_storage.py  # IAlertStorage
│   │   │   ├── processed_repo.py # IProcessedVideoRepository
│   │   │   └── report_repo.py    # IReportRepository
│   │   ├── services/             # Serviços de Domínio
│   │   │   └── compliance_service.py # Regras de persistência de frames consecutivos
│   │   └── use_cases/            # Casos de Uso da Aplicação
│   │       ├── analyze_frame.py      # Análise de um frame individual
│   │       ├── process_video_file.py # Processamento de um arquivo completo
│   │       ├── process_batch.py      # Varredura recursiva de diretórios
│   │       ├── watch_directory.py    # Monitoramento contínuo em tempo real (Watcher)
│   │       └── process_stream.py     # Processamento de streams de câmeras (RTSP/Webcam)
│   │
│   ├── infrastructure/           # ADAPTADORES CONCRETOS (TECNOLOGIAS)
│   │   ├── ai/                   # YOLODetector (Ultralytics + CUDA)
│   │   ├── video/                # UniversalVideoReader (OpenCV + PyAV HEVC Raw Bitstream)
│   │   ├── notifications/        # EmailNotifier, CompositeNotifier, RateLimiter
│   │   ├── storage/              # FileAlertStorage (Salva evidências anotadas por placa/data)
│   │   └── repositories/         # CsvReportRepository, JsonProcessedRepository
│   │
│   ├── presentation/             # CAMADA DE ENTRADA E APRESENTAÇÃO
│   │   ├── api/                  # FastAPI REST API (Rotas, Schemas Pydantic, Factory)
│   │   └── cli/                  # CLI Commands e Renderizador Visual OpenCV
│   │
│   ├── config/                   # Carregador de configurações tipadas (Settings)
│   ├── container.py              # Container de Injeção de Dependências (IoC)
│   ├── main.py                   # Entrypoint CLI (100% retrocompatível)
│   ├── video_batch.py            # Entrypoint Batch (100% retrocompatível)
│   └── api.py                    # Entrypoint REST API (100% retrocompatível)
│
├── tests/                        # SUÍTE DE TESTES AUTOMATIZADOS
│   ├── unit/                     # Testes de Domínio, Metadados e Serviços
│   └── integration/              # Testes de Casos de Uso, Pipeline e API
├── alertas/                      # Evidências geradas (Placa/Data/Imagens)
├── scripts/                      # Utilitários de treinamento e benchmarks
├── docker-compose.yml            # Orquestração de containers com suporte a GPU
├── Dockerfile                    # Container Docker com PyTorch e FFmpeg
├── requirements.txt              # Dependências do projeto
└── README.md
```

---

## ⚡ Suporte a Arquivos HEVC/H.265 Brutos (`.265`)

Muitos DVRs veiculares e câmeras industriais geram arquivos com extensão `.265` contendo **raw byte-streams H.265 (HEVC)** sem contêiner MP4/MKV e com pré-alocação de disco de tamanho fixo. O `UniversalVideoReader` implementado possui:
1. **Verificação preventiva de arquivos corrompidos/vazios:** Detecta blocos compostos integralmente por bytes nulos (`0x00`).
2. **Decodificação de Baixo Nível via PyAV:** Inicializa o decodificador diretamente com o demuxer `hevc`, contornando falhas normais do `cv2.VideoCapture`.
3. **Fallback Automático para OpenCV:** Se o arquivo for um container convencional (MP4, AVI), chaveia automaticamente para a biblioteca mais otimizada.

---

## 🚀 Como Executar

### 🖥️ 1. Interface Gráfica Desktop (Recomendado)

O sistema possui uma aplicação Desktop moderna com tema escuro (Dark Mode), suporte a **Drag & Drop (arrastar e soltar arquivos)**, player integrado em tempo real e monitoramento de frota:

```bash
python run_desktop.py
# ou:
python -m src.desktop
```

#### Recursos da Interface Desktop:
- 📥 **Arrastar & Soltar Vídeos (Drag & Drop):** Solte arquivos `.mp4` ou `.265` diretamente na janela para análise instantânea com a IA.
- 📺 **Player Frame a Frame:** Visualize os trabalhadores sendo rastreados com caixas coloridas (Verde = EPI OK, Vermelho = Violação) e contador de FPS na GPU.
- 🏷️ **Detecção Automática de Placa e Data:** Extrai os metadados do veículo automaticamente e permite edição manual.
- 📂 **Varredura em Lote & Watch Mode:** Aponte para a pasta de gravações e processe centenas de vídeos ou deixe o monitoramento contínuo ativado.
- 🚨 **Galeria de Evidências:** Visualize as fotos capturadas com a tarja de auditoria e abra a planilha CSV com 1 clique.
- 📹 **Transmissão ao Vivo:** Teste webcams e câmeras IP (RTSP) diretamente na interface.

---

### 2. Processamento em Lote via Terminal (Placa > Data > Arquivo)

A IA inspeciona pastas locais ou compartilhadas na rede (`[pasta] / [PLACA] / [DATA] / [video]`):

```bash
# Processar recursivamente todos os vídeos novos
python -m src.video_batch --dir ./gravacoes

# Ou através do entrypoint principal:
python -m src.main --batch-dir ./gravacoes
```

#### Modo Observador Contínuo (`--watch`):
Fica monitorando o diretório e analisa novos arquivos à medida que são descarregados pelos veículos:
```bash
python -m src.video_batch --dir ./gravacoes --watch --interval 10
```

#### Processar um arquivo específico (`.mp4` ou `.265`):
```bash
python -m src.video_batch --file ./gravacoes/ABC1D23/2026-10-04/video_01.265
```

---

### 2. Monitoramento de Câmeras ao Vivo (RTSP / Webcam)

```bash
# Webcam padrão (source 0)
python -m src.main

# Câmera IP via stream RTSP
python -m src.main --source "rtsp://usuario:senha@192.168.1.100:554/stream"

# Modo Headless / Produção (sem abrir janela de interface gráfica)
python -m src.main --source 0 --no-display
```

---

### 3. API REST FastAPI (Integração Corporativa)

Inicia o servidor de inferência e compliance para consumo via HTTP:

```bash
python -m src.api --host 0.0.0.0 --port 8000
```
- **Documentação Swagger Interativa:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Documentação Redoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Healthcheck:** `GET /health`
- **Estatísticas do Sistema & GPU:** `GET /stats`
- **Envio de Arquivo multipart:** `POST /analyze/upload`
- **Envio de Imagem em Base64:** `POST /analyze/base64`

---

## 🧪 Suíte de Testes Automatizados

O projeto conta com suíte completa de testes unitários e de integração em `tests/`:

```bash
# Executar todos os testes com saída verbosa
pytest tests -v

# Executar apenas testes unitários
pytest tests/unit -v

# Executar validação completa do sistema
python scratch/test_full.py
```

---

## 📊 Entregáveis e Evidências Geradas

1. **Evidências Visuais com Tarja de Auditoria:**
   Salvas em `alertas/[PLACA]/[DATA]/` contendo imagem em alta resolução com caixas delimitadoras coloridas (Verde = EPI Presente; Vermelho = Violação) e tarja superior detalhando Placa, Data, Câmera e Minutagem.
2. **Relatório Consolidado CSV (`relatorio_infracoes.csv`):**
   Gera log com timestamp, placa, vídeo, instante em segundos e tipo de infração para integração com BI e planilhas gerenciais.
3. **Controle de Idempotência (`videos_processados.json`):**
   Garante que vídeos já auditados não sejam processados repetidamente, economizando tempo e uso de GPU.

---

## ⚙️ Configuração Centralizada (`config/settings.yaml`)

Todas as opções operacionais são configuradas via YAML com tipagem estrita:
```yaml
detection:
  model_path: "models/best.pt"
  confidence: 0.45
  iou_threshold: 0.45
  imgsz: 640
  track: true

compliance:
  min_violation_frames: 5
  required_ppe:
    - "helmet"
    - "gloves"
    - "vest"

batch:
  supported_extensions:
    - ".mp4"
    - ".avi"
    - ".mkv"
    - ".265"
    - ".hevc"
  output_csv: "relatorio_infracoes.csv"
  processed_db: "videos_processados.json"
```
