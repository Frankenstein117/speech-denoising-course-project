from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluate import score_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one enhanced speech directory.")
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scores = score_directory(args.clean_dir, args.candidate_dir, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(scores.drop(columns=["file", "stage"]).mean().round(4))


if __name__ == "__main__":
    main()
