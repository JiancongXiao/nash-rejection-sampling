"""Run COMAL's published log-probability script with Qwen-only compatibility.

Some COMAL revisions import both Qwen and Tulu dataset helpers even though the
checked-in ``data_utils.py`` only contains the Qwen helpers.  The unconditional
import prevents the supported Qwen path from starting.  This entry point adds
guarded, never-used Tulu symbols and then executes the upstream script without
changing its Qwen data, model, or log-probability logic.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType


class _UnavailableTuluDataset:
    def __init__(self, *_args, **_kwargs) -> None:
        raise RuntimeError(
            "This COMAL checkout does not provide PreferenceBaseTuluDataset; "
            "use --model_type qwen or install a COMAL revision with Tulu support."
        )


def _unavailable_tulu_collate(*_args, **_kwargs):
    raise RuntimeError(
        "This COMAL checkout does not provide collate_preference_base_tulu; "
        "use --model_type qwen or install a COMAL revision with Tulu support."
    )


def install_missing_tulu_symbols(data_utils: ModuleType) -> tuple[str, ...]:
    """Supply guards for stale unconditional imports in upstream COMAL."""

    installed = []
    if not hasattr(data_utils, "PreferenceBaseTuluDataset"):
        data_utils.PreferenceBaseTuluDataset = _UnavailableTuluDataset
        installed.append("PreferenceBaseTuluDataset")
    if not hasattr(data_utils, "collate_preference_base_tulu"):
        data_utils.collate_preference_base_tulu = _unavailable_tulu_collate
        installed.append("collate_preference_base_tulu")
    return tuple(installed)


def selected_model_type(argv: list[str]) -> str:
    if "--model_type" not in argv:
        return "qwen"
    index = argv.index("--model_type")
    if index + 1 >= len(argv):
        raise ValueError("--model_type requires a value")
    return argv[index + 1]


def main() -> None:
    comal_root = Path(os.environ["NASHRS_COMAL_ROOT"]).resolve()
    upstream = comal_root / "get_logprobs.py"
    if not upstream.is_file():
        raise FileNotFoundError(f"missing upstream COMAL script: {upstream}")

    sys.path.insert(0, str(comal_root))
    data_utils = importlib.import_module("data_utils")
    installed = install_missing_tulu_symbols(data_utils)
    if installed and selected_model_type(sys.argv[1:]) != "qwen":
        raise RuntimeError(
            "The compatibility entry point can only fill stale imports for the "
            "Qwen path; this COMAL checkout has no functional Tulu dataset helpers."
        )
    if installed:
        print(
            "COMAL compatibility: guarded missing, unused Tulu symbols: "
            + ", ".join(installed),
            file=sys.stderr,
        )
    runpy.run_path(str(upstream), run_name="__main__")


if __name__ == "__main__":
    main()
