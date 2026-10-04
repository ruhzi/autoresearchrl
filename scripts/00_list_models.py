"""List OpenRouter models so we configure real slugs instead of guessing.

    python scripts/00_list_models.py                # anthropic models
    python scripts/00_list_models.py openai gemini  # filter by substring(s)
    python scripts/00_list_models.py --all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from arl.openrouter import list_models  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_all = "--all" in sys.argv
    filters = [a.lower() for a in args] or ["anthropic"]

    models = list_models()
    print(f"{len(models)} models in the OpenRouter catalogue\n")

    rows = []
    for m in models:
        mid = m.get("id", "")
        if not show_all and not any(f in mid.lower() for f in filters):
            continue
        pricing = m.get("pricing") or {}
        try:
            pin = float(pricing.get("prompt", 0)) * 1e6
            pout = float(pricing.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            pin = pout = 0.0
        params = m.get("supported_parameters") or []
        rows.append((
            mid,
            m.get("context_length") or 0,
            pin,
            pout,
            "structured" if "structured_outputs" in params
            else ("resp_fmt" if "response_format" in params else "-"),
        ))

    rows.sort()
    print(f"{'model id':<50} {'ctx':>9} {'$/Min':>8} {'$/Mout':>8}  json")
    print("-" * 90)
    for mid, ctx, pin, pout, js in rows:
        print(f"{mid:<50} {ctx:>9} {pin:>8.2f} {pout:>8.2f}  {js}")

    print("\nCurrently configured (one line per stage, see config.py):")
    roles = (("gate (stage 4)", config.MODEL_GATE),
             ("rlm root (stage 5)", config.MODEL_RLM_ROOT),
             ("enumerator (stage 5)", config.MODEL_ENUM_SUB),
             ("retrieve (stage 6)", config.MODEL_RETRIEVE),
             ("compose (stage 7)", config.MODEL_COMPOSE),
             ("verify (stage 8)", config.MODEL_SAMPLE_VERIFY))
    for name, slug in roles:
        print(f"  {name:<22} {slug}")
    have = {r[0] for r in rows} if show_all else {m.get("id") for m in models}
    for name, slug in roles:
        mark = "ok" if slug in have else "NOT FOUND in catalogue -- fix it"
        print(f"    {name}: {slug} -> {mark}")
    print("\nStructured-output support matters: the extractor must be a model "
          "with 'structured' in the json column.")


if __name__ == "__main__":
    main()
