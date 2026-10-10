"""Sistema de alertas para violações de EPI.

Gerencia o envio de alertas por email e salvamento de imagens
quando violações de segurança são detectadas, com rate limiting
para evitar excesso de notificações.
"""

import os
import cv2
import time
import logging
import smtplib
import threading
from datetime import datetime
from typing import Dict, Optional
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class RateLimiter:
    """Controle de taxa de envio de alertas.
    
    Evita o envio excessivo de alertas para o mesmo tipo de
    violação e/ou mesmo trabalhador dentro de um período.
    """

    def __init__(self, cooldown_seconds: int = 60):
        """Inicializa o rate limiter.
        
        Args:
            cooldown_seconds: Tempo mínimo (em segundos) entre alertas do mesmo tipo.
        """
        self.cooldown = cooldown_seconds
        self._last_sent: Dict[str, float] = {}

    def can_send(self, alert_key: str) -> bool:
        """Verifica se um alerta pode ser enviado.
        
        Args:
            alert_key: Chave única do alerta (ex: 'no_helmet_track_5')
            
        Returns:
            True se o cooldown já passou, False caso contrário.
        """
        now = time.time()
        last = self._last_sent.get(alert_key, 0)
        if now - last >= self.cooldown:
            self._last_sent[alert_key] = now
            return True
        remaining = self.cooldown - (now - last)
        logger.debug(f"⏱️ Rate limit ativo para '{alert_key}'. Próximo envio em {remaining:.0f}s")
        return False

    def reset(self):
        """Reseta todos os timers."""
        self._last_sent.clear()


class AlertManager:
    """Gerenciador de alertas de violação de EPI.
    
    Responsável por salvar imagens de violações localmente
    e enviar notificações por email com screenshots anexados.
    """

    def __init__(
        self,
        save_images: bool = True,
        save_dir: str = "alertas",
        send_email: bool = False,
        cooldown_seconds: int = 60,
        image_quality: int = 85,
    ):
        """Inicializa o gerenciador de alertas.
        
        Args:
            save_images: Se True, salva imagens de violações localmente.
            save_dir: Diretório para salvar as imagens.
            send_email: Se True, envia alertas por email.
            cooldown_seconds: Intervalo mínimo entre alertas do mesmo tipo.
            image_quality: Qualidade JPEG das imagens salvas (1-100).
        """
        self.save_images = save_images
        self.save_dir = save_dir
        self.send_email = send_email
        self.image_quality = image_quality
        self.rate_limiter = RateLimiter(cooldown_seconds)

        # Configuração de email (carregada do .env)
        self._email_sender: Optional[str] = None
        self._email_password: Optional[str] = None
        self._email_receiver: Optional[str] = None
        self._smtp_server: str = "smtp.gmail.com"
        self._smtp_port: int = 587

        # Criar diretório de alertas
        if save_images:
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"📁 Imagens de alerta serão salvas em: {os.path.abspath(save_dir)}")

        # Carregar configuração de email
        if send_email:
            self._load_email_config()

        # Contadores
        self._alerts_sent = 0
        self._images_saved = 0

    def _load_email_config(self):
        """Carrega configuração de email das variáveis de ambiente."""
        load_dotenv()

        self._email_sender = os.getenv("EMAIL_SENDER")
        self._email_password = os.getenv("EMAIL_PASSWORD")
        self._email_receiver = os.getenv("EMAIL_RECEIVER")
        self._smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self._smtp_port = int(os.getenv("SMTP_PORT", "587"))

        if not self._email_sender or not self._email_password:
            logger.error(
                "❌ Credenciais de email não configuradas! "
                "Configure EMAIL_SENDER e EMAIL_PASSWORD no arquivo .env"
            )
            self.send_email = False
            return

        if not self._email_receiver:
            logger.error("❌ EMAIL_RECEIVER não configurado no arquivo .env")
            self.send_email = False
            return

        logger.info(f"📧 Email configurado: {self._email_sender} → {self._email_receiver}")

    def process_violation(self, frame, violation, metadata: Optional[Dict] = None) -> Optional[str]:
        """Processa uma violação detectada.
        
        Args:
            frame: Frame do vídeo (numpy array BGR) com as anotações.
            violation: Objeto Violation do módulo compliance.
            metadata: Dicionário opcional com metadados (placa, data, video_name, video_time, etc.).

        Returns:
            Caminho da imagem salva (se salva), ou None.
        """
        # Criar chave única para rate limiting
        prefix = ""
        if metadata:
            placa = metadata.get("placa", "")
            video_name = metadata.get("video_name", "")
            if placa:
                prefix = f"{placa}_"
            elif video_name:
                prefix = f"{video_name}_"

        alert_key = f"{prefix}{violation.violation_type}_track_{violation.track_id}"

        # Verificar rate limit
        if not self.rate_limiter.can_send(alert_key):
            return None

        placa_info = f" [Placa: {metadata.get('placa')}]" if (metadata and metadata.get("placa")) else ""
        logger.info(f"🚨 Processando alerta: {violation.description}{placa_info}")

        # Salvar imagem localmente
        image_path = None
        if self.save_images:
            image_path = self._save_violation_image(frame, violation, metadata=metadata)

        # Enviar email (em thread separada para não bloquear)
        if self.send_email and image_path:
            thread = threading.Thread(
                target=self._send_email_alert,
                args=(image_path, violation, metadata),
                daemon=True,
            )
            thread.start()

        return image_path

    def _save_violation_image(self, frame, violation, metadata: Optional[Dict] = None) -> Optional[str]:
        """Salva a imagem da violação no disco.
        
        Args:
            frame: Frame do vídeo.
            violation: Dados da violação.
            metadata: Metadados contextuais (placa, data, video_name, etc.).
            
        Returns:
            Caminho do arquivo salvo ou None em caso de erro.
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            violation_type = violation.violation_type.replace(" ", "_")
            track_str = f"_id{violation.track_id}" if violation.track_id is not None else ""

            target_dir = self.save_dir

            # Se houver metadados de placa/data, organizar em subpastas ou nome do arquivo
            if metadata and (metadata.get("placa") or metadata.get("data")):
                placa = metadata.get("placa", "sem_placa")
                data_rec = metadata.get("data", "sem_data")
                v_time = str(metadata.get("video_time", "00-00-00")).replace(":", "-")
                v_name = Path(metadata.get("video_name", "video")).stem

                # Organizar por pasta alertas/PLACA/DATA/
                target_dir = os.path.join(self.save_dir, placa, data_rec)
                os.makedirs(target_dir, exist_ok=True)
                filename = f"alerta_{v_name}_{v_time}_{violation_type}{track_str}.jpg"
            else:
                os.makedirs(target_dir, exist_ok=True)
                filename = f"alerta_{timestamp}_{violation_type}{track_str}.jpg"

            filepath = os.path.join(target_dir, filename)

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
            cv2.imwrite(filepath, frame, encode_params)

            self._images_saved += 1
            logger.info(f"💾 Imagem salva: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"❌ Erro ao salvar imagem: {e}")
            return None

    def _send_email_alert(self, image_path: str, violation, metadata: Optional[Dict] = None) -> None:
        """Envia alerta por email com a imagem da violação anexada.
        
        Executado em thread separada para não bloquear o loop principal.
        """
        try:
            # Montar email
            msg = MIMEMultipart()
            msg["From"] = self._email_sender
            msg["To"] = self._email_receiver

            # Assunto com placa se disponível
            if metadata and metadata.get("placa"):
                msg["Subject"] = f"🚨 Alerta EPI [Placa: {metadata['placa']}] - {violation.description}"
            else:
                msg["Subject"] = f"🚨 Alerta EPI: {violation.description}"

            # Corpo do email
            timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")

            body_lines = [
                "ALERTA DE VIOLAÇÃO DE EPI",
                "=" * 40,
                "",
                f"Tipo: {violation.description}",
            ]

            if metadata:
                if metadata.get("placa"):
                    body_lines.append(f"Placa do Veículo: {metadata['placa']}")
                if metadata.get("data"):
                    body_lines.append(f"Data da Gravação: {metadata['data']}")
                if metadata.get("video_name"):
                    body_lines.append(f"Arquivo de Vídeo: {metadata['video_name']}")
                if metadata.get("video_time"):
                    body_lines.append(f"Tempo no Vídeo: {metadata['video_time']}")

            body_lines.extend([
                f"Data/Hora Processamento: {timestamp}",
                f"Confiança: {violation.confidence:.1%}",
                f"ID do Trabalhador: {violation.track_id or 'N/A'}",
                f"Frames Consecutivos: {violation.consecutive_frames}",
                "",
                "A imagem da evidência com anotações está anexada a este email.",
                "",
                "---",
                "Sistema de Detecção de EPI - Alerta Automático",
            ])

            body = "\n".join(body_lines)
            msg.attach(MIMEText(body, "plain", "utf-8"))

            # Anexar imagem
            with open(image_path, "rb") as img_file:
                mime_image = MIMEImage(img_file.read(), _subtype="jpeg")
                mime_image.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(image_path),
                )
                msg.attach(mime_image)

            # Enviar
            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._email_sender, self._email_password)
                server.sendmail(
                    self._email_sender,
                    self._email_receiver,
                    msg.as_string(),
                )

            self._alerts_sent += 1
            logger.info(f"📧 Email de alerta enviado para {self._email_receiver}")

        except smtplib.SMTPAuthenticationError:
            logger.error(
                "❌ Falha na autenticação SMTP. Verifique suas credenciais.\n"
                "   Para Gmail, use uma Senha de App: "
                "https://myaccount.google.com/apppasswords"
            )
        except Exception as e:
            logger.error(f"❌ Erro ao enviar email: {e}")

    def get_stats(self) -> Dict[str, int]:
        """Retorna estatísticas dos alertas."""
        return {
            "emails_enviados": self._alerts_sent,
            "imagens_salvas": self._images_saved,
        }

    def test_email(self) -> bool:
        """Envia um email de teste para verificar a configuração.
        
        Returns:
            True se o email foi enviado com sucesso.
        """
        if not self.send_email:
            logger.error("❌ Email não está configurado.")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self._email_sender
            msg["To"] = self._email_receiver
            msg["Subject"] = "✅ Teste - Sistema de Detecção de EPI"

            body = (
                "Este é um email de teste do Sistema de Detecção de EPI.\n\n"
                "Se você recebeu este email, a configuração está correta!\n\n"
                f"Servidor SMTP: {self._smtp_server}:{self._smtp_port}\n"
                f"Remetente: {self._email_sender}\n"
                f"Destinatário: {self._email_receiver}\n"
            )
            msg.attach(MIMEText(body, "plain", "utf-8"))

            with smtplib.SMTP(self._smtp_server, self._smtp_port) as server:
                server.starttls()
                server.login(self._email_sender, self._email_password)
                server.sendmail(
                    self._email_sender,
                    self._email_receiver,
                    msg.as_string(),
                )

            logger.info("✅ Email de teste enviado com sucesso!")
            return True

        except Exception as e:
            logger.error(f"❌ Falha no teste de email: {e}")
            return False
