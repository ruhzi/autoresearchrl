"""Central configuration for the autoresearchrl dataset pipeline.

Everything tunable lives here so the scripts stay thin.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Runs at import, before anything below reads os.getenv. A variable already
    set in the real environment is never overwritten, so an explicit
    `$env:OPENROUTER_API_KEY = ...` still overrides the file -- handy for
    one-off runs against a different key or model.

    Deliberately tiny: no python-dotenv dependency, no interpolation, no export
    keyword. Just enough to keep the key off the command line.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

# ------------------------------------------------------------------- models
#
# Every LLM in the pipeline: one line per stage/role. Edit here.
# Verify slugs against your account:  python scripts/00_list_models.py

# stage 4: eligibility screen (3 booleans)
MODEL_GATE = os.getenv("ARL_MODEL_GATE",
                       "~deepseek/deepseek-v4-flash-latest")
# stage 5: RLM root orchestrator
MODEL_RLM_ROOT = os.getenv("ARL_MODEL_RLM_ROOT",
                           "anthropic/claude-sonnet-5")
# stage 5: enumerator subagents
MODEL_ENUM_SUB = os.getenv("ARL_MODEL_ENUM_SUB",
                           "anthropic/claude-sonnet-5")
# stage 6: verbatim retrieval agent
MODEL_RETRIEVE = os.getenv("ARL_MODEL_RETRIEVE",
                           "z-ai/glm-5.2")
# stage 7: training-prompt composer
MODEL_COMPOSE = os.getenv("ARL_MODEL_COMPOSE",
                          "anthropic/claude-sonnet-5")
# stage 8: final audit judge
MODEL_SAMPLE_VERIFY = os.getenv("ARL_MODEL_SAMPLE_VERIFY",
                                "anthropic/claude-sonnet-5")

POOL_JSON = DATA / "pool.json"
POOL_PARTIAL_JSON = DATA / "pool_partial.json"   
SAMPLE_JSON = DATA / "sample.json"
SRC_DIR = DATA / "src"
PARSED_DIR = DATA / "parsed"
EXTRACT_DIR = DATA / "extracted"
LABELS_JSONL = DATA / "samples.jsonl"
COMPOSED_DIR = DATA / "composed"        
VERIFIED_DIR = DATA / "verified"        
TRAIN_JSONL = DATA / "train_samples.jsonl"
# The pipeline's only hand-maintained inputs: audited corrections, replayed on
# every rebuild so `run_all.py` cannot revert them. See arl/overrides.py.
LABEL_OVERRIDES = DATA / "label_overrides.json"
PROMPT_OVERRIDES = DATA / "prompt_overrides.json"
REPORT_JSON = DATA / "report.json"
FETCH_REPORT = DATA / "fetch_report.json"
SCREEN_DIR = DATA / "screen"          
CANDIDATES_DIR = DATA / "candidates"  
CANDIDATES_PARTIAL_DIR = DATA / "candidates_partial"

# ---------------------------------------------------------------- arXiv query

CATEGORIES = ["cs.LG", "cs.CL", "cs.AI", "cs.CV"]

DATE_FROM = "202607010000"
DATE_TO = "202608260000"   # exclusive end: covers submissions through 25 Aug 2026

POOL_TARGET = int(os.getenv("ARL_POOL_TARGET", "1600"))   # arXiv pool size; raised with the window so the
                                                           # newer week is added, not swapped in for the older one
POOL_PAGE = 220                                            # API page size
# THE number of papers the pipeline processes. Raising it grows the corpus:
# data/sample.json is a manifest of papers already paid for, so the draw keeps
# all of them and tops up to here. Lowering it deletes nothing. See
# arl.arxiv_api.draw_sample.
SAMPLE_SIZE = int(os.getenv("ARL_SAMPLE_SIZE", "50"))
SAMPLE_SEED = int(os.getenv("ARL_SAMPLE_SEED", "20260812"))


API_DELAY_S = 3.5
EPRINT_DELAY_S = 4.0

USER_AGENT = os.getenv(
    "ARL_USER_AGENT",
    "autoresearchrl/0.1 (research dataset build; ruhaan.chopra@olbrain.com)",
)

# ---------------------------------------------------------------- OpenRouter

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

def api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set.\n\n"
            f"  Easiest: edit {ROOT / '.env'} and put your key on the\n"
            "  OPENROUTER_API_KEY line. That file is gitignored and is read\n"
            "  automatically at startup.\n\n"
            "  Or via the shell (overrides .env):\n"
            "    $env:OPENROUTER_API_KEY = 'sk-or-v1-...'"
        )
    return key

TRAJECTORIES = os.getenv("ARL_TRAJECTORIES", "1") == "1"
TRAJ_DIR = DATA / "trajectories"

REQUEST_TIMEOUT_S = 300
MAX_RETRIES = 4
LLM_TEMPERATURE = float(os.getenv("ARL_LLM_TEMPERATURE", "0.0"))

TOOL_RESULT_MAX_CHARS = int(os.getenv("ARL_TOOL_RESULT_MAX_CHARS", "20000"))

NET_WAIT_S = int(os.getenv("ARL_NET_WAIT_S", "3600"))
MAX_WORKERS = int(os.getenv("ARL_MAX_WORKERS", "8"))


REASONING_MODE = os.getenv("ARL_REASONING_MODE", "off")


def reasoning_param() -> dict | None:
    """Map REASONING_MODE to the OpenRouter `reasoning` body param."""
    mode = REASONING_MODE.strip().lower()
    if mode == "off":
        return {"enabled": False}
    if mode in ("low", "medium", "high"):
        return {"effort": mode}
    return None

RLM_FAST_PATH = os.getenv("ARL_RLM_FAST_PATH", "1") == "1"

RLM_MAX_ROOT_TURNS = int(os.getenv("ARL_RLM_MAX_ROOT_TURNS", "24"))
RLM_ROOT_MAX_TOKENS = int(os.getenv("ARL_RLM_ROOT_MAX_TOKENS", "4096"))
RLM_ENUM_MAX_TOKENS = int(os.getenv("ARL_RLM_ENUM_MAX_TOKENS", "16000"))

RLM_SUBAGENT_MAX_CHARS = int(os.getenv("ARL_RLM_SUBAGENT_MAX_CHARS", "60000"))

RLM_PEEK_CHARS = int(os.getenv("ARL_RLM_PEEK_CHARS", "2000"))
RLM_GREP_MAX_HITS = int(os.getenv("ARL_RLM_GREP_MAX_HITS", "20"))
RLM_GREP_CONTEXT_CHARS = int(os.getenv("ARL_RLM_GREP_CONTEXT_CHARS", "200"))

RETRIEVE_MAX_TURNS = int(os.getenv("ARL_RETRIEVE_MAX_TURNS", "24"))
RETRIEVE_MAX_TOKENS = int(os.getenv("ARL_RETRIEVE_MAX_TOKENS", "8192"))

RETRIEVE_READ_CHARS = int(os.getenv("ARL_RETRIEVE_READ_CHARS", "8000"))


COMPOSE_MAX_TOKENS = int(os.getenv("ARL_COMPOSE_MAX_TOKENS", "3072"))
COMPOSE_RETRIES = int(os.getenv("ARL_COMPOSE_RETRIES", "2"))

VERIFY_MAX_TOKENS = int(os.getenv("ARL_VERIFY_MAX_TOKENS", "2048"))


MAX_TABLE_CHARS = 60_000
MAX_CONTEXT_CHARS = 60_000



def ensure_dirs() -> None:
    for d in (DATA, SRC_DIR, PARSED_DIR, EXTRACT_DIR, SCREEN_DIR,
              CANDIDATES_DIR, CANDIDATES_PARTIAL_DIR, COMPOSED_DIR,
              VERIFIED_DIR):
        d.mkdir(parents=True, exist_ok=True)
