from __future__ import annotations

import argparse
from pathlib import Path

from app.pipeline import InternshipApplicationPipeline, load_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated Internship Application Pipeline")
    parser.add_argument("--brochure", required=True, help="Path to internship brochure PDF")
    parser.add_argument("--cv", required=True, help="Path to master CV markdown/text file")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config()
    pipeline = InternshipApplicationPipeline(config)

    pipeline.run(
        brochure_pdf=Path(args.brochure),
        master_cv_path=Path(args.cv),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
