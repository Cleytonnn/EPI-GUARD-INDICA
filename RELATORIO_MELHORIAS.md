# Relatório de Revisão — EPI GUARD (PROJETO CLEYTONN)

**Data da revisão:** 26/11/2026
**Escopo:** todo o código em `src/`, `tests/`, `scripts/`, configurações (`config/`, `.env.example`), scripts Windows (`.bat`), `Dockerfile` e `docker-compose.yml`.
**Onde estão as correções:** esta pasta `MELHORIAS/` é uma **cópia completa do projeto já corrigida**. O projeto original **não foi alterado**. O arquivo `ALTERACOES.diff` mostra, linha a linha, tudo o que mudou em relação ao original.

---

## 1. Resumo

| Item | Original | Versão em MELHORIAS |
|---|---|---|
| Testes automatizados | **49 passaram, 5 falharam/erro** (4 falhas + 1 erro de coleta) | **70 passaram, 0 falharam** (3 ignorados por falta de Tkinter/VLC no ambiente de teste) |
| Lint (ruff) | Sem erros | Sem erros, código formatado |
| Erros corrigidos | — | **38** (9 críticos, 10 altos, 19 médios/baixos) |
| Testes novos | — | **18 testes de regressão** em `tests/unit/test_melhorias.py`, um para cada erro importante |

Todos os erros críticos listados abaixo foram **reproduzidos no código original** antes de serem corrigidos (não são suposições).

**Os 3 problemas mais graves:**
1. As listas do `settings.yaml` (EPIs obrigatórios, classes de infração, extensões de vídeo) **eram ignoradas**. O sistema auditava "óculos" e "botas" em vez de "Uniforme RF", mesmo com o YAML correto.
2. Os modelos `models/*.pt` da pasta são **ponteiros do Git LFS** (132 bytes), não os pesos reais. O programa **não iniciava** (erro ao carregar o modelo).
3. Vídeos `.265` recuperados pelo VLC eram lidos com **0 quadros** e marcados como "processados" sem nenhuma análise.

---

## 2. Erros encontrados e corrigidos

### 2.1 Críticos (resultado errado ou sistema não funciona)

| # | Erro | Onde | Efeito | Correção |
|---|---|---|---|---|
| C1 | `AppSettings.load()` filtrava as chaves com `hasattr(Classe, chave)`. Campos com `default_factory` (listas) não existem como atributo da classe e eram **descartados em silêncio** | `src/config/settings.py` | `required_ppe`, `violation_classes`, `file_extensions` e `cors_origins` do YAML nunca eram usados. Exigia `goggles`/`boots` e não `uniform_rf` | Leitura via `dataclasses.fields()`, com aviso no log para chaves desconhecidas |
| C2 | `models/epi_best.pt` e `models/best.pt` são ponteiros de texto do Git LFS. O container só verificava se o arquivo existia | `src/container.py` | `Container()` quebrava com `TypeError` e o app, a API e os testes não iniciavam | Novo `model_utils.py` detecta o ponteiro LFS. O sistema volta ao modo Pose com aviso claro (`git lfs pull`). O `instalar.bat` também avisa |
| C3 | O backend `vlc-remux` abria o vídeo, mas `read()` só tratava `opencv` e `pyav` | `universal_reader.py` | `.265` recuperado pelo VLC tinha **0 quadros analisados** e era marcado como processado | `read()` passou a tratar `vlc-remux` |
| C4 | O temporário do remux usava só o nome do arquivo (`vlc_clean_CH1.mp4`) e o VLC aceitava um arquivo antigo que já existisse | `universal_reader.py`, `vlc_service.py` | Dois `CH1.265` de placas diferentes podiam ler **o vídeo do outro veículo** | Nome único (uuid), saída antiga apagada antes do remux, código de retorno verificado, temporário apagado no `release()` |
| C5 | Quadro sem detecções não zerava os contadores (o conjunto do quadro anterior não era limpo) | `compliance_service.py` | Infração confirmada **sem** os N quadros consecutivos exigidos | O conjunto é limpo antes do decaimento |
| C6 | Sem tracking, várias pessoas com a mesma infração compartilhavam a chave `(None, no_helmet)` e o contador subia N vezes por quadro | `compliance_service.py` | Com 2 pessoas, a confirmação de 3 quadros acontecia no 2º quadro | Cada chave conta no máximo 1 vez por quadro |
| C7 | O filtro de civis usava `track_id in civil_tracks`. Sem tracking, todos têm `None` | `compliance_service.py` | **Um pedestre no quadro silenciava as infrações de todos os trabalhadores** | O filtro só considera IDs reais |
| C8 | `HybridEPIDetector` (backend padrão) não tinha `is_cabin` | `hybrid_detector.py` | O modo cabine/motorista (CH2, DMS…) **nunca era ativado** no modo híbrido | Propriedade `is_cabin` repassada ao Pose |
| C9 | `HybridEPIDetector` não tinha `model_path` | `hybrid_detector.py`, `routes.py` | `GET /health` dava **erro 500** e o healthcheck do Docker marcava o contêiner como "unhealthy" | `model_path` no híbrido e `/health` tolerante |

### 2.2 Altos (segurança, concorrência, perda de dados)

| # | Erro | Onde | Correção |
|---|---|---|---|
| A1 | API **sem autenticação**, escutando em `0.0.0.0`, com CORS `*` e `allow_credentials=True` (combinação inválida e insegura) | `api/app.py`, `routes.py`, `settings.py` | `API_KEY` opcional no `.env` (header `X-API-Key`, comparação em tempo constante). Host padrão `127.0.0.1`. CORS com origens explícitas e sem credenciais quando a origem é `*` |
| A2 | Upload e Base64 **sem limite de tamanho** (risco de estouro de memória). O Base64 aceitava lixo | `routes.py` | Limite `api.max_upload_mb` (10 MB) → HTTP 413, e `b64decode(validate=True)` → HTTP 400 |
| A3 | Rotas `async` rodavam a inferência no event loop, então uma imagem travava todas as outras requisições | `routes.py` | `run_in_threadpool` |
| A4 | O mesmo detector (com estado de tracking) era usado ao mesmo tempo por threads da API, das câmeras HTTP e do Desktop, sem trava | `container.py` | Novo `ThreadSafeDetector` (decorator com `RLock`) |
| A5 | `/analyze` compartilhava **um único** `ComplianceService` entre todas as câmeras e clientes, e uma imagem avulsa **sempre** voltava `is_compliant=true` | `analyze_frame.py` | Um serviço por `camera_id` (LRU), `is_compliant` considera as infrações visíveis no quadro e o novo campo `frame_violations` |
| A6 | `camera_api` criava o `ComplianceService` sem `infer_missing_ppe`, o que gerava **inferência duplicada** e falsos alertas no modo híbrido | `camera_api.py` | Usa `container.create_compliance_service()` (regra única) |
| A7 | Câmeras HTTP: violações sem metadados, então a chave do rate limit colidia entre câmeras e as evidências caíam soltas em `alertas/`. Câmeras acima de `max_workers` (10) nunca eram monitoradas | `camera_api_client.py` | Metadados com `camera_id`, pool ≥ nº de câmeras, `try/except` por ciclo, parada imediata (`Event`), estatísticas com trava |
| A8 | Senhas e tokens em texto puro no `cameras.yaml`, e 2 câmeras de exemplo vinham **habilitadas** com endereços falsos | `config/cameras.yaml` | Suporte a `${VARIAVEL}` (valor vem do `.env`) e todos os exemplos desabilitados |
| A9 | Caminho das evidências montado com placa/`camera_id` sem sanitizar (`../../`), e evidências com o mesmo nome eram sobrescritas | `file_alert_storage.py` | Sanitização dos nomes e sufixo `_1`, `_2`… em vez de sobrescrever |
| A10 | Um vídeo com defeito **interrompia o lote inteiro** e matava o watcher | `process_batch.py`, `watch_directory.py` | `try/except` por vídeo, resumo com nº de falhas, parada responsiva do watcher |

### 2.3 Médios e baixos

| # | Erro | Correção |
|---|---|---|
| M1 | `videos_processados.json` era sobrescrito direto: se o programa fechasse no meio, o arquivo corrompia e **tudo era reprocessado** | Gravação atômica (temporário + `os.replace`) |
| M2 | O CSV de infrações aceitava nomes começando com `=`, `+`, `-`, `@` (fórmula executada ao abrir no Excel) | Células neutralizadas, cabeçalho recriado se o arquivo sumir, trava de escrita |
| M3 | Caminhos de saída (`alertas`, CSV, JSON, coleta) dependiam da pasta de onde o programa era aberto | Ancorados na raiz do projeto |
| M4 | Erro de sintaxe no `settings.yaml` era engolido em silêncio | Log de erro explícito |
| M5 | E-mail: sem timeout SMTP (thread presa para sempre), sem suporte à porta 465 e sem vários destinatários | `timeout=30`, `SMTP_SSL` na 465, lista separada por vírgula |
| M6 | `RateLimiter` sem trava, usando o relógio do sistema e com crescimento sem limite | Trava, `time.monotonic()`, limpeza periódica |
| M7 | Híbrido: `_worker_grace` crescia sem limite em streams longos (vazamento de memória) | Limpeza de IDs antigos |
| M8 | Webcam usava `CAP_DSHOW` (só existe no Windows), então não abria no Linux/Docker | `CAP_DSHOW` só no Windows |
| M9 | Desktop: um `after()` por quadro, sem controle. Com a UI mais lenta que a câmera, a fila do Tk crescia (atraso e memória) | Descarta a prévia enquanto a anterior não foi desenhada |
| M10 | Aprendizado Ativo: `review()` aceitava IDs com `../`, e `export_dataset()` apagava (`rmtree`) qualquer destino, inclusive a própria coleta | Validação do ID e recusa de destinos perigosos |
| M11 | `python -m src.api` carregava os modelos **duas vezes** (o uvicorn reimportava o módulo) | Passa o objeto `app` quando não há `--reload` |
| M12 | `iniciar_api.bat` subia em `0.0.0.0` com `--reload` (modo desenvolvimento) | `127.0.0.1` sem `--reload` |
| M13 | `instalar.bat`: `%ERRORLEVEL%` dentro de `( … )` fazia a detecção do `py` **nunca funcionar**, e a versão 3.10+ não era verificada | Rótulos/`goto` e checagem da versão |
| M14 | `instalar.bat` criava o venv com `--system-site-packages` (conflito com pacotes globais) | venv isolado |
| M15 | Dockerfile instalava o Python 3.11, mas o `pip` pertencia ao 3.10: as dependências iam para um interpretador e o app rodava em outro | Python 3.10 do Ubuntu 22.04 com `python -m pip` |
| M16 | Dockerfile instalava o torch do PyPI (via ultralytics) antes do torch CUDA, rodava como root e **copiava o `.env` para a imagem** | Torch CUDA primeiro, usuário sem root e novo `.dockerignore` |
| M17 | `docker-compose` montava `./.env`: se o arquivo não existisse, o Docker criava uma **pasta** `.env` | `env_file` opcional (requer Docker Compose 2.24+) e `config` montada como somente leitura |
| M18 | Testes Desktop/VLC **falhavam** (em vez de serem ignorados) sem Tkinter ou display | `pytest.importorskip` e skip em `TclError` |
| M19 | Container: `"best.pt" in model_file` também casava com nomes como `epi_best.pt`, e um `backend` inválido não era tratado | Checagem por "pose" no nome e backend validado |

---

## 3. Arquivos alterados e criados

**Alterados (34):** `src/config/settings.py`, `src/container.py`, `src/api.py`, `src/camera_api.py`, `src/core/services/compliance_service.py`, `src/core/use_cases/{analyze_frame,process_batch,watch_directory}.py`, `src/infrastructure/ai/hybrid_detector.py`, `src/infrastructure/cctv/camera_api_client.py`, `src/infrastructure/notifications/{email_notifier,rate_limiter}.py`, `src/infrastructure/repositories/{csv_report_repo,json_processed_repo}.py`, `src/infrastructure/storage/file_alert_storage.py`, `src/infrastructure/training/active_learning_collector.py`, `src/infrastructure/video/{camera_stream,universal_reader,vlc_service}.py`, `src/presentation/api/{app,routes,schemas}.py`, `src/presentation/desktop/views/{live_view,file_analysis_view}.py`, `tests/unit/{test_desktop_app,test_vlc_service}.py`, `config/{settings,cameras}.yaml`, `.env.example`, `Dockerfile`, `docker-compose.yml`, `instalar.bat`, `iniciar_api.bat`, `README.md`.

**Criados (4):**
- `src/infrastructure/ai/model_utils.py`: detecta ponteiros do Git LFS e pesos inválidos.
- `src/infrastructure/ai/thread_safe_detector.py`: acesso ao detector serializado entre threads.
- `tests/unit/test_melhorias.py`: 18 testes de regressão.
- `.dockerignore`.

---

## 4. Mudanças de comportamento que você precisa saber

1. **O `settings.yaml` agora vale de verdade.** Com `required_ppe: [helmet, gloves, uniform_rf, vest]`, no backend `pose` serão cobrados *vest* **e** *uniform_rf*, o que pode gerar alertas duplicados de tronco. No backend `hybrid` (padrão) isso não acontece. Se usar `pose`, remova `vest`/`no_vest` do YAML.
2. **A API agora escuta só em `127.0.0.1` por padrão.** Para acessar de outra máquina, use `--host 0.0.0.0` **e** defina `API_KEY` no `.env`. O Docker já passa `0.0.0.0`.
3. **`/analyze/*` responde `is_compliant=false`** quando há infração visível no quadro, mesmo sem confirmação temporal. O novo campo `frame_violations` mostra quantas.
4. **Câmeras de exemplo do `cameras.yaml` vêm desabilitadas.** Coloque as senhas no `.env` (ex.: `CAM_SENHA=...`) e use `${CAM_SENHA}` no YAML.
5. **Modelos LFS:** enquanto `models/epi_best.pt` for ponteiro, o sistema roda **só com Pose/heurística** (com aviso no log). Para usar o modelo treinado: `git lfs install` e `git lfs pull`.

---

## 5. Recomendações não aplicadas (exigem decisão de arquitetura)

- **A Clean Architecture está quebrada em dois pontos:** `core/use_cases/process_stream.py` importa `infrastructure.video.CameraStream` e `presentation.cli.visualizer`, e `process_video_file.py` importa o visualizer. O ideal é injetar um "renderizador" por interface (porta), como já é feito com o detector.
- **Lógica duplicada no Desktop:** `file_analysis_view.py` e `live_view.py` reimplementam o laço de `ProcessVideoFileUseCase`/`ProcessLiveStreamUseCase`. A aba "Ao Vivo" **não aplica o ComplianceService**: conta infrações brutas e não salva evidência. O certo é reaproveitar os casos de uso com callback de quadro.
- **Tracking entre câmeras:** o `ThreadSafeDetector` evita a corrupção por concorrência, mas o ByteTrack continua único. IDs de câmeras diferentes se misturam. Para muitas câmeras, o ideal é um detector (ou tracker) por câmera.
- **Falha silenciosa de inferência:** os detectores retornam `[]` quando o modelo dá erro, e isso é interpretado como "nenhuma infração". Vale expor um contador de erros no `/health`/`/stats`.
- **O botão "Parar" do lote** (Desktop) só para o watcher. O lote comum roda até o fim, porque `ProcessBatchUseCase` não tem cancelamento.
- **Dependências sem versão máxima** (`ultralytics>=8.3`, `numpy>=1.24`…): gerar um `requirements.lock` (ex.: `pip freeze`) para instalações reprodutíveis.

---

## 6. Como validar

```powershell
cd "PROJETO CLEYTONN\MELHORIAS"
.\instalar.bat          # ou use o .venv já existente
.\testar.bat            # esperado: 70 passed (alguns "skipped" se faltar VLC)
```

Testes executados nesta revisão: Python 3.13, ultralytics 8.4, torch 2.15 (CPU), Linux sem display. Os testes de interface Tkinter e VLC foram ignorados nesse ambiente e devem rodar normalmente no Windows.
