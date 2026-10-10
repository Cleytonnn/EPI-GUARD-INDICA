"""Enums e constantes de domínio para EPIs e violações."""

from enum import Enum


class PPEClass(str, Enum):
    """Classes de EPIs válidas no domínio."""
    HELMET = "helmet"
    GLOVES = "gloves"
    VEST = "vest"
    BOOTS = "boots"
    GOGGLES = "goggles"
    PERSON = "Person"
    NONE = "none"

    @property
    def label_pt(self) -> str:
        translations = {
            self.HELMET: "Capacete ✅",
            self.GLOVES: "Luvas ✅",
            self.VEST: "Colete ✅",
            self.BOOTS: "Botas ✅",
            self.GOGGLES: "Óculos ✅",
            self.PERSON: "Pessoa",
            self.NONE: "---",
        }
        return translations.get(self, str(self.value))


class ViolationClass(str, Enum):
    """Classes de ausência de EPIs (violações de segurança)."""
    NO_HELMET = "no_helmet"
    NO_GLOVES = "no_gloves"
    NO_BOOTS = "no_boots"
    NO_GOGGLE = "no_goggle"
    NO_VEST = "no_vest"

    @property
    def label_pt(self) -> str:
        translations = {
            self.NO_HELMET: "Sem Capacete 🔴",
            self.NO_GLOVES: "Sem Luvas 🔴",
            self.NO_BOOTS: "Sem Botas 🔴",
            self.NO_GOGGLE: "Sem Óculos de Proteção 🔴",
            self.NO_VEST: "Sem Colete 🔴",
        }
        return translations.get(self, str(self.value))

    @property
    def related_ppe(self) -> str:
        mapping = {
            self.NO_HELMET: "helmet",
            self.NO_GLOVES: "gloves",
            self.NO_BOOTS: "boots",
            self.NO_GOGGLE: "goggles",
            self.NO_VEST: "vest",
        }
        return mapping.get(self, "desconhecido")
