from __future__ import annotations

import importlib
import textwrap


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")










def test_get_provider_max_output_tokens_resolution(monkeypatch, tmp_path):
    """Per-model cap wins over provider-level; unknown → None (leave untouched).

    Regression for the fallback max_tokens carry-over: when the primary's
    output budget (e.g. an Anthropic proxy pinning 200000) is not re-clamped
    to the fallback model's cap, a smaller-cap provider (Groq 8192/16384)
    rejects every request with a deterministic output-cap 400. The clamp in
    try_activate_fallback relies on this resolver returning the fallback's
    real cap; None must mean "no config → don't touch max_tokens".
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, """\
        providers:
          groq:
            models:
              qwen/qwen3.8-27b:
                max_output_tokens: 8192
              openai/gpt-oss-120b:
                max_output_tokens: 32768
          capped-provider:
            max_output_tokens: 4096
            models:
              some-model: {}
          legacy-key:
            models:
              m1:
                max_tokens: 5000
        """)

    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)

    resolve = to_mod.get_provider_max_output_tokens

    # Per-model max_output_tokens wins.
    assert resolve("groq", "qwen/qwen3.8-27b") == 8192
    assert resolve("groq", "openai/gpt-oss-120b") == 32768
    # Provider-level fallback when the model has no own cap.
    assert resolve("capped-provider", "some-model") == 4096
    # ``max_tokens`` accepted as an alias for ``max_output_tokens``.
    assert resolve("legacy-key", "m1") == 5000
    # Unknown provider / unknown model with no provider-level cap → None.
    assert resolve("nonexistent-xyz", "whatever") is None
    assert resolve("groq", "unknown-model-abc") is None
    # Empty provider id → None (guard).
    assert resolve("", "qwen/qwen3.8-27b") is None


def test_anthropic_adapter_honors_timeout_kwarg():
    """build_anthropic_client(timeout=X) overrides the default read timeout."""
    pytest = __import__("pytest")
    pytest.importorskip("anthropic")  # skip if optional SDK missing
    from agent.anthropic_adapter import build_anthropic_client

    c_default = build_anthropic_client("sk-ant-dummy", None)
    c_custom = build_anthropic_client("sk-ant-dummy", None, timeout=45.0)
    c_invalid = build_anthropic_client("sk-ant-dummy", None, timeout=-1)

    # Custom overrides the read timeout; invalid falls back to the default;
    # the connect timeout is unaffected by the override.
    assert c_custom.timeout.read == 45.0
    assert c_invalid.timeout.read == c_default.timeout.read != 45.0
    assert c_custom.timeout.connect == c_default.timeout.connect


def test_resolved_api_call_timeout_priority(monkeypatch, tmp_path):
    """AIAgent._resolved_api_call_timeout() honors config > env > default priority."""
    # Isolate HERMES_HOME
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")

    # Case A: config wins over env var
    _write_config(tmp_path, """\
        providers:
          openrouter:
            request_timeout_seconds: 77
            models:
              openai/gpt-4o-mini:
                timeout_seconds: 42
        """)
    monkeypatch.setenv("HERMES_API_TIMEOUT", "999")

    from run_agent import AIAgent
    agent = AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # Per-model override wins
    assert agent._resolved_api_call_timeout() == 42.0

    # Provider-level (different model, no per-model override)
    agent.model = "some/other-model"
    assert agent._resolved_api_call_timeout() == 77.0

    # Case B: no config → env wins
    _write_config(tmp_path, "")
    # Clear the cached config load
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)
    import run_agent as ra_mod
    importlib.reload(ra_mod)

    agent2 = ra_mod.AIAgent(
        model="some/model",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent2._resolved_api_call_timeout() == 999.0




