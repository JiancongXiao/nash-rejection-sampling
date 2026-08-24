"""Write OpenRLHF global-step logs as JSON Lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nashrs.metrics_export import export_step_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = export_step_metrics(args.log, args.output)
    print(
        json.dumps(
            {
                "steps_exported": len(records),
                "output": str(args.output),
                "last_step": records[-1]["step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
