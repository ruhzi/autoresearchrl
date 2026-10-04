"""Paper text contains arrows, Greek, CJK. A cp1252 console must not crash
the stage scripts when they print it. arl.util makes stdout/stderr UTF-8
with replacement on import."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNIPPET = (
    "import sys; sys.path.insert(0, r'%s'); "
    "from arl.util import log; "
    "print('stdout → α 中'); log('stderr → α 中')"
) % ROOT


def test_unicode_survives_cp1252_console():
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0")
    out = subprocess.run([sys.executable, "-c", SNIPPET], cwd=ROOT, env=env,
                         capture_output=True, stdin=subprocess.DEVNULL)
    assert out.returncode == 0, out.stderr.decode("utf-8", "replace")
    assert b"UnicodeEncodeError" not in out.stderr
    assert out.stdout.decode("utf-8").startswith("stdout ")
    assert out.stderr.decode("utf-8").startswith("stderr ")
