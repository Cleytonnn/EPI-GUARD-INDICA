"""Notificador de alertas via Email (SMTP)."""

from __future__ import annotations

import logging
import os
import smtplib
import threading
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.core.domain.violation import Violation

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Implementa INotifier enviando e-mails com imagem anexada."""

    def __init__(
        self,
        sender: str | None = None,
        password: str | None = None,
        receiver: str | None = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        enabled: bool = True,
    ):
        self.sender = sender
        self.password = password
        self.receiver = receiver
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.enabled = enabled and bool(sender and password and receiver)
        self.timeout = 30  # segundos; sem timeout uma conexão SMTP travada prendia a thread para sempre
        self._sent_count = 0
        self._count_lock = threading.Lock()

        if not self.enabled:
            logger.debug("ℹ️ Notificador de e-mail desabilitado ou sem credenciais completas.")

    def notify(self, violation: Violation, image_path: str | None = None) -> bool:
        """Envia o e-mail de alerta em uma thread separada para não travar a pipeline."""
        if not self.enabled:
            return False

        thread = threading.Thread(
            target=self._send_sync,
            args=(violation, image_path),
            daemon=True,
        )
        thread.start()
        return True

    def _send_sync(self, violation: Violation, image_path: str | None) -> bool:
        """Execução síncrona do envio SMTP."""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = self.receiver

            meta = violation.metadata
            plate_info = f" [Placa: {meta.plate}]" if meta and meta.plate else ""
            msg["Subject"] = f"🚨 Alerta EPI{plate_info} - {violation.description}"

            timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
            lines = [
                "ALERTA DE VIOLAÇÃO DE EPI",
                "=" * 40,
                f"Tipo: {violation.description}",
            ]
            if meta:
                if meta.plate:
                    lines.append(f"Placa do Veículo: {meta.plate}")
                if meta.record_date:
                    lines.append(f"Data da Gravação: {meta.record_date}")
                if meta.video_name:
                    lines.append(f"Arquivo de Vídeo: {meta.video_name}")
                if meta.video_time_str:
                    lines.append(f"Tempo no Vídeo: {meta.video_time_str}")

            lines.extend(
                [
                    f"Data/Hora Processamento: {timestamp}",
                    f"Confiança: {violation.confidence:.1%}",
                    f"ID do Trabalhador: {violation.track_id or 'N/A'}",
                    f"Frames Consecutivos: {violation.consecutive_frames}",
                    "",
                    "A evidência fotográfica com as marcações está em anexo.",
                    "---",
                    "Sistema de Detecção de EPI - Arquitetura Limpa",
                ]
            )

            body = "\n".join(lines)
            msg.attach(MIMEText(body, "plain", "utf-8"))

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    mime_img = MIMEImage(f.read(), _subtype="jpeg")
                    mime_img.add_header(
                        "Content-Disposition",
                        "attachment",
                        filename=os.path.basename(image_path),
                    )
                    msg.attach(mime_img)

            receivers = [r.strip() for r in str(self.receiver).replace(";", ",").split(",") if r.strip()]
            if int(self.smtp_port) == 465:  # SMTPS (SSL direto) — antes só funcionava STARTTLS na 587
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, receivers, msg.as_string())
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                    server.starttls()
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, receivers, msg.as_string())

            with self._count_lock:
                self._sent_count += 1
            logger.info(f"📧 E-mail de alerta enviado com sucesso para {self.receiver}")
            return True

        except Exception as e:
            logger.error(f"❌ Falha no envio de e-mail SMTP: {e}")
            return False

    @property
    def total_sent(self) -> int:
        return self._sent_count
