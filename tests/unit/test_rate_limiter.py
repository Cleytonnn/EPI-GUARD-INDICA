"""Testes unitários do RateLimiter."""

from src.infrastructure.notifications.rate_limiter import RateLimiter


def test_rate_limiter_allows_first_and_blocks_immediate_repeats():
    limiter = RateLimiter(cooldown_seconds=60)
    key = "cam1_no_helmet_1"

    assert limiter.can_send(key) is True
    assert limiter.can_send(key) is False

    # Outra chave deve ser permitida
    other_key = "cam1_no_gloves_1"
    assert limiter.can_send(other_key) is True


def test_rate_limiter_reset():
    limiter = RateLimiter(cooldown_seconds=60)
    key = "test_key"
    assert limiter.can_send(key) is True
    assert limiter.can_send(key) is False

    limiter.reset()
    assert limiter.can_send(key) is True
