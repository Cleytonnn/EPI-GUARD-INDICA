"""Script de treinamento do modelo YOLO para detecção de EPI.

Treina (fine-tune) um modelo YOLO no dataset Construction-PPE
da Ultralytics, que contém classes para EPI usado e ausente.

Uso:
    python scripts/train_model.py
    python scripts/train_model.py --model yolo11s.pt --epochs 200
    python scripts/train_model.py --dataset roboflow --api-key SUA_CHAVE
"""

import argparse
import logging
import sys
from pathlib import Path

# Configura encoding UTF-8 para evitar erros de emoji no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_ultralytics(
    data: str = "construction-ppe.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    project: str = "runs/train",
    name: str = "ppe_detection",
):
    """Treina o modelo YOLO com o dataset Construction-PPE da Ultralytics.

    Este dataset possui 1.416 imagens e 11 classes:
    - EPI usado: helmet, gloves, vest, boots, goggles
    - EPI ausente: no_helmet, no_gloves, no_boots, no_goggle
    - Outros: Person, none

    Args:
        model_name: Modelo base para fine-tuning.
        epochs: Número de épocas de treinamento.
        imgsz: Tamanho da imagem de entrada.
        batch: Tamanho do batch.
        device: Dispositivo ('', '0', 'cpu'). Vazio = auto-detectar.
        project: Diretório do projeto de treinamento.
        name: Nome do experimento.
    """
    from ultralytics import YOLO

    logger.info("=" * 60)
    logger.info("🏋️ TREINAMENTO DO MODELO DE DETECÇÃO DE EPI")
    logger.info("=" * 60)
    logger.info(f"   Modelo base: {model_name}")
    logger.info(f"   Épocas: {epochs}")
    logger.info(f"   Tamanho de imagem: {imgsz}")
    logger.info(f"   Batch size: {batch}")
    logger.info(f"   Dispositivo: {device or 'auto'}")
    logger.info("=" * 60)

    # Carregar modelo base
    logger.info(f"📥 Carregando modelo base: {model_name}")
    model = YOLO(model_name)

    # Treinar modelo YOLO
    logger.info(f"🚀 Iniciando treinamento com dataset: {data}...")
    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=project,
        name=name,
        patience=20,  # Early stopping após 20 épocas sem melhora
        save=True,
        save_period=10,  # Salvar checkpoint a cada 10 épocas
        plots=True,  # Gerar gráficos de treinamento
        verbose=True,
    )

    # Copiar melhor modelo para a pasta models/
    best_model_path = Path(project) / name / "weights" / "best.pt"
    if best_model_path.exists():
        dest = Path(__file__).resolve().parent.parent / "models" / "best.pt"
        dest.parent.mkdir(exist_ok=True)

        import shutil

        shutil.copy2(best_model_path, dest)
        logger.info(f"✅ Melhor modelo copiado para: {dest}")
    else:
        logger.warning("⚠️ Modelo 'best.pt' não encontrado. Verifique o treinamento.")

    logger.info("=" * 60)
    logger.info("🏁 TREINAMENTO CONCLUÍDO!")
    logger.info(f"   Resultados em: {Path(project) / name}")
    logger.info("=" * 60)

    return results


def train_roboflow(
    api_key: str,
    workspace: str,
    project_name: str,
    version: int = 1,
    model_name: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
):
    """Treina o modelo com um dataset do Roboflow Universe.

    Útil para adicionar classes extras como macacão (coverall),
    máscara facial, protetor auricular, etc.

    Args:
        api_key: Chave da API do Roboflow.
        workspace: Nome do workspace no Roboflow.
        project_name: Nome do projeto no Roboflow.
        version: Versão do dataset.
        model_name: Modelo base para fine-tuning.
        epochs: Número de épocas.
        imgsz: Tamanho da imagem.
        batch: Tamanho do batch.
        device: Dispositivo de treinamento.
    """
    from roboflow import Roboflow
    from ultralytics import YOLO

    logger.info("📥 Baixando dataset do Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("yolov8")

    logger.info(f"📁 Dataset baixado em: {dataset.location}")

    # Treinar
    model = YOLO(model_name)
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project="runs/train",
        name="ppe_roboflow",
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copiar melhor modelo
    best_path = Path("runs/train/ppe_roboflow/weights/best.pt")
    if best_path.exists():
        dest = Path(__file__).resolve().parent.parent / "models" / "best_roboflow.pt"
        dest.parent.mkdir(exist_ok=True)
        import shutil

        shutil.copy2(best_path, dest)
        logger.info(f"✅ Modelo Roboflow salvo em: {dest}")

    return results


def validate_model(model_path: str = "models/best.pt"):
    """Valida o modelo treinado no dataset de teste.

    Args:
        model_path: Caminho para o modelo treinado.
    """
    from ultralytics import YOLO

    logger.info(f"📊 Validando modelo: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data="construction-ppe.yaml")

    logger.info("=" * 60)
    logger.info("📊 RESULTADOS DA VALIDAÇÃO")
    logger.info(f"   mAP50: {metrics.box.map50:.4f}")
    logger.info(f"   mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"   Precisão: {metrics.box.mp:.4f}")
    logger.info(f"   Recall: {metrics.box.mr:.4f}")
    logger.info("=" * 60)

    return metrics


def export_model(
    model_path: str = "models/best.pt",
    format: str = "onnx",
):
    """Exporta o modelo para formato otimizado.

    Args:
        model_path: Caminho para o modelo treinado.
        format: Formato de exportação ('onnx', 'engine', 'openvino').
    """
    from ultralytics import YOLO

    logger.info(f"📦 Exportando modelo para formato: {format}")
    model = YOLO(model_path)
    exported = model.export(format=format)
    logger.info(f"✅ Modelo exportado: {exported}")
    return exported


def parse_args():
    """Processa argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Treinamento do modelo de detecção de EPI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python scripts/train_model.py                                    # Treinar com dataset padrão
  python scripts/train_model.py --model yolo11s.pt --epochs 200    # Modelo maior, mais épocas
  python scripts/train_model.py --device 0                         # Usar GPU 0
  python scripts/train_model.py --validate models/best.pt          # Validar modelo
  python scripts/train_model.py --export models/best.pt --format onnx  # Exportar para ONNX
  python scripts/train_model.py --dataset roboflow --api-key CHAVE --workspace ws --project proj
        """,
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Caminho do arquivo YAML do dataset customizado (ex: datasets/custom_ppe/data.yaml)",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo base (default: yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Número de épocas (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", default="", help="Dispositivo: '', '0', 'cpu' (default: auto)")
    parser.add_argument(
        "--dataset", default="ultralytics", choices=["ultralytics", "roboflow"], help="Fonte do dataset"
    )
    parser.add_argument("--api-key", default=None, help="API key do Roboflow")
    parser.add_argument("--workspace", default=None, help="Workspace do Roboflow")
    parser.add_argument("--project", default=None, help="Projeto do Roboflow")
    parser.add_argument("--version", type=int, default=1, help="Versão do dataset Roboflow")
    parser.add_argument("--validate", default=None, metavar="MODEL_PATH", help="Validar um modelo treinado")
    parser.add_argument("--export", default=None, metavar="MODEL_PATH", help="Exportar modelo para formato otimizado")
    parser.add_argument("--format", default="onnx", help="Formato de exportação (default: onnx)")

    return parser.parse_args()


def main():
    """Função principal."""
    args = parse_args()

    # Modo validação
    if args.validate:
        validate_model(args.validate)
        return

    # Modo exportação
    if args.export:
        export_model(args.export, args.format)
        return

    # Modo treinamento
    if args.dataset == "roboflow":
        if not args.api_key or not args.workspace or not args.project:
            logger.error("❌ Para dataset Roboflow, forneça: --api-key, --workspace, --project")
            sys.exit(1)
        train_roboflow(
            api_key=args.api_key,
            workspace=args.workspace,
            project_name=args.project,
            version=args.version,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    else:
        dataset_yaml = args.data if args.data else "construction-ppe.yaml"
        train_ultralytics(
            data=dataset_yaml,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()
