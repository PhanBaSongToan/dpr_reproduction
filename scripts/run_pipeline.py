"""Entry point placeholder for the reproducibility pipeline."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPR reproduction stages.")
    parser.add_argument("--stage", choices=["preprocess", "bm25", "train", "index", "evaluate", "all"], default="all")
    parser.parse_args()
    print("Pipeline scaffold ready. Configure data and model before running a full experiment.")


if __name__ == "__main__":
    main()
