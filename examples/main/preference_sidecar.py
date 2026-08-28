"""JSON-lines preference-oracle sidecar for dependency-isolated Main runs."""

from __future__ import annotations

import argparse
import json
import sys

from nashrs import build_preference_oracle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components-json", required=True)
    args = parser.parse_args()
    components = json.loads(args.components_json)
    oracle = build_preference_oracle(components, device="cuda")
    print(
        "NASHRS_READY\t" + json.dumps({"component_count": len(oracle.components)}),
        flush=True,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        probabilities = oracle.compare(
            request["prompts"], request["left"], request["right"]
        )
        print(
            "NASHRS_RESULT\t"
            + json.dumps([float(value) for value in probabilities]),
            flush=True,
        )


if __name__ == "__main__":
    main()
