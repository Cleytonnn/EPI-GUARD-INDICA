"""Limiares de decisão da detecção de EPIs (fonte única de verdade).

Os valores foram calibrados empiricamente sobre as gravações reais de CFTV de
veículo (1920x1080 @ 15 fps) disponíveis no projeto: capacetes verde/laranja/
amarelo/branco, uniforme RF NR-10 (bege com pala laranja e camisa laranja) e
luvas de vaqueta. Ao trocar de câmera/modelo, revalide-os com
``scripts/benchmark_stability.py`` antes de alterar.

Nomenclatura: ``*_CONF`` = confiança mínima de um detector; ``*_RATIO`` =
proporção de pixels/área; os demais são descritos individualmente.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Fusão Pose + modelo de EPIs (hybrid_detector)
# --------------------------------------------------------------------------- #
# Confiança mínima do modelo para promover um "civil" a trabalhador (EPI inequívoco).
CIVIL_PROMOTION_HELMET_CONF = 0.58
CIVIL_PROMOTION_UNIFORM_CONF = 0.60

# Capacete visto pela Pose com confiança >= este valor não é derrubado por um
# ``no_helmet`` isolado do modelo externo cuja confiança seja < MODEL_NO_HELMET_OVERRIDE_CONF
# (o modelo erra capacetes verde-escuros em perfil/costas).
POSE_HELMET_TRUST_CONF = 0.26
MODEL_NO_HELMET_OVERRIDE_CONF = 0.70

# Evidência mínima do modelo para considerar colete/uniforme e luvas (fallback sem Pose).
MODEL_VEST_MIN_CONF = 0.45
MODEL_GLOVES_MIN_CONF = 0.45
MODEL_NO_GLOVES_MIN_CONF = 0.55

# --------------------------------------------------------------------------- #
# Heurísticas anatômicas por cor (pose_detector)
# --------------------------------------------------------------------------- #
# Caixas aninhadas com interseção/menor-área >= este valor são duplicatas.
NESTED_BOX_OVERLAP_RATIO = 0.58

# Pessoa só é "funcionário" pelo capacete se a pontuação da heurística for >= este valor.
WORKER_HELMET_SCORE = 0.48

# Close-up real de motorista na cabine (altura/largura mínimas relativas ao quadro
# e base da caixa próxima à borda inferior). Evita promover pedestres próximos.
CABIN_MIN_HEIGHT_RATIO = 0.62
CABIN_MIN_WIDTH_RATIO = 0.25
CABIN_MIN_BOTTOM_RATIO = 0.85

# Luvas: proporção mínima de couro claro (vaqueta) ou cor de alta visibilidade na mão.
GLOVE_VAQUETA_MIN_RATIO = 0.34
GLOVE_HIVIS_MIN_RATIO = 0.25
# Luva só vence a pele quando sua proporção supera ``pele * GLOVE_SKIN_DOMINANCE``.
GLOVE_SKIN_DOMINANCE = 0.90
# Mão ocluída (manga laranja RF ou sombra/calça escura): resultado indeterminado.
GLOVE_OCCLUSION_ORANGE_RATIO = 0.35
GLOVE_OCCLUSION_DARK_RATIO = 0.65
