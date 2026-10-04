"""config.py reads its overridable knobs from the environment at import.

Each test spawns a fresh interpreter so the module is imported exactly once
under a controlled environment (config runs _load_dotenv() at import).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _probe(expr: str, **env: str) -> str:
    e = dict(os.environ)
    e.update(env)
    out = subprocess.run(
        [sys.executable, "-c", f"import config; print({expr})"],
        cwd=ROOT, env=e, capture_output=True, text=True, check=True,
        stdin=subprocess.DEVNULL)
    return out.stdout.strip()


def test_data_defaults_to_repo_data_dir():
    env = {k: v for k, v in os.environ.items() if k != "ARL_DATA_DIR"}
    out = subprocess.run(
        [sys.executable, "-c", "import config; print(config.DATA)"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
        stdin=subprocess.DEVNULL)
    assert Path(out.stdout.strip()) == (ROOT / "data").resolve()


def test_data_dir_override_is_absolute(tmp_path):
    rel = "out_dir_for_test"
    got = Path(_probe("config.DATA", ARL_DATA_DIR=rel))
    assert got.is_absolute()
    assert got == (ROOT / rel).resolve()


def test_ensure_dirs_creates_missing_override(tmp_path):
    target = tmp_path / "fresh" / "nested"
    assert not target.exists()
    _probe("config.ensure_dirs() or config.DATA", ARL_DATA_DIR=str(target))
    assert target.is_dir()


def test_date_window_defaults_and_overrides():
    assert _probe("config.DATE_FROM") == "202607010000"
    assert _probe("config.DATE_TO") == "202608260000"
    assert _probe("config.DATE_FROM", ARL_DATE_FROM="202701010000") == "202701010000"
    assert _probe("config.DATE_TO", ARL_DATE_TO="202702010000") == "202702010000"


def test_user_agent_has_no_email():
    ua = _probe("config.USER_AGENT")
    assert "@" not in ua
    assert "olbrain" not in ua.lower()
    assert ua.startswith("autoresearchrl/")
