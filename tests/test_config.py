import os
from pathlib import Path

from tor import config


def test_phase_defaults_to_zero(monkeypatch):
    monkeypatch.delenv("TOR_PHASE", raising=False)
    assert config.get_settings().phase == "0"


def test_load_secrets_sets_env_without_overwriting(monkeypatch, tmp_path):
    f = tmp_path / "s.env"
    f.write_text("HF_TOKEN=abc\nTABPFN_TOKEN=xyz\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("TABPFN_TOKEN", "keep")
    config._SECRETS.clear()
    config.load_secrets(f)
    assert os.environ["HF_TOKEN"] == "abc"
    assert os.environ["TABPFN_TOKEN"] == "keep"
    config._SECRETS.clear()


def test_redact_masks_secret_values(monkeypatch, tmp_path):
    f = tmp_path / "s.env"
    f.write_text("HF_TOKEN=abc\nTABPFN_TOKEN=xyz\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    config._SECRETS.clear()
    config.load_secrets(f)
    assert config.redact("token abc here") == "token *** here"
    config._SECRETS.clear()


def test_cache_dir_falls_back_when_data_unwritable(monkeypatch):
    monkeypatch.delenv("TOR_CACHE", raising=False)
    real = os.access
    monkeypatch.setattr(
        os, "access",
        lambda p, m, *a, **k: False if str(p).startswith("/data/hadar") else real(p, m, *a, **k),
    )
    s = config.get_settings()
    assert s.cache_dir == s.repo_root / "cache"
