"""Utilitário para extrair frames de vídeos para enriquecer o dataset de treinamento.

Uso:
    python scripts/extract_frames.py --video "caminho/do/video.mp4" --interval 2
    python scripts/extract_frames.py --dir "gravacoes" --interval 3
"""

import argparse
import sys
from pathlib import Path
import cv2

# Configura encoding UTF-8 para evitar erros de emoji no terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Garante acesso aos módulos do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.infrastructure.video.universal_reader import UniversalVideoReader


def extract_frames(
    input_path: str,
    output_dir: str = "datasets/raw_frames",
    interval_seconds: float = 2.0,
    max_frames_per_video: int = 150,
):
    """Extrai frames espaçados no tempo de arquivos de vídeo para anotação."""
    in_path = Path(input_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        videos = [in_path]
    elif in_path.is_dir():
        exts = {".mp4", ".265", ".h265", ".hevc", ".avi", ".mkv"}
        videos = [p for p in in_path.rglob("*") if p.suffix.lower() in exts]
    else:
        print(f"❌ Caminho inválido: {in_path}")
        return

    print(f"🎬 {len(videos)} vídeo(s) encontrado(s) para extração.")
    total_extracted = 0

    for v_idx, video_file in enumerate(videos, 1):
        reader = UniversalVideoReader(video_file)
        if not reader.is_opened():
            print(f"⚠️ Não foi possível abrir: {video_file.name}")
            continue

        fps = reader.fps or 25.0
        frame_step = max(1, int(fps * interval_seconds))
        frame_idx = 0
        extracted_this_video = 0
        video_stem = video_file.stem

        print(f"[{v_idx}/{len(videos)}] Processando {video_file.name} (a cada {interval_seconds}s via {reader.backend})...")

        while True:
            ret, frame = reader.read()
            if not ret or frame is None:
                break

            if frame_idx % frame_step == 0:
                out_filename = f"{video_stem}_f{frame_idx:06d}.jpg"
                out_filepath = out_dir / out_filename
                cv2.imwrite(str(out_filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_this_video += 1
                total_extracted += 1

                if extracted_this_video >= max_frames_per_video:
                    break

            frame_idx += 1

        reader.release()
        print(f"   -> {extracted_this_video} frames extraídos.")

    print("=" * 60)
    print(f"✅ Extração concluída! {total_extracted} imagens salvas em:")
    print(f"   {out_dir}")
    print("👉 Agora você pode subir essas imagens para o Roboflow para rotular e treinar!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrair frames de vídeos para dataset YOLO")
    parser.add_argument("--video", "-v", help="Caminho para arquivo de vídeo único")
    parser.add_argument("--dir", "-d", help="Diretório com múltiplos vídeos")
    parser.add_argument("--interval", "-i", type=float, default=2.0, help="Intervalo em segundos entre cada frame")
    parser.add_argument("--output", "-o", default="datasets/raw_frames", help="Pasta de saída das imagens")
    parser.add_argument("--max", "-m", type=int, default=150, help="Máximo de frames por vídeo")

    args = parser.parse_args()
    target = args.video or args.dir or "gravacoes"
    extract_frames(
        input_path=target,
        output_dir=args.output,
        interval_seconds=args.interval,
        max_frames_per_video=args.max,
    )

