"""Utilitários para validar arquivos de pesos (.pt) antes de carregá-los."""

from __future__ import annotations

from pathlib import Path

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def is_git_lfs_pointer(path: str | Path) -> bool:
    """Indica se o arquivo é apenas um ponteiro de texto do Git LFS (modelo não baixado)."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size > 1024:
            return False
        with open(p, "rb") as fh:
            return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def is_usable_weights(path: str | Path) -> bool:
    """O arquivo existe, não está vazio e não é ponteiro do Git LFS."""
    p = Path(path)
    try:
        return p.is_file() and p.stat().st_size > 0 and not is_git_lfs_pointer(p)
    except OSError:
        return False
