"""Entrypoint CLI para a Interface Gráfica Desktop."""

from src.container import Container
from src.presentation.desktop import run_desktop_app


def main():
    container = Container()
    run_desktop_app(container=container)


if __name__ == "__main__":
    main()
